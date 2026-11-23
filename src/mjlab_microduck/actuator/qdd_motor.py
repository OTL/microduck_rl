"""QDD biped motor model with a torque-speed envelope (driver-side PD, MIT mode).

The plain QDD actuators (robot/qdd_biped_constants.py) only clamp the torque to
the motor peak. That lets a policy learn dynamic motions (jumps) the motors
cannot do: the GIM6010-8 runs out of voltage as it speeds up. This actuator is an
explicit PD (the drivers' MIT loop) whose output is clipped to the envelope of the
robot repo's analysis/hop.py ``tmax()`` (docs/jump-feasibility.md):

  GIM6010-8 (hip_yaw / hip_roll / hip_pitch / knee), output side:
    |tau| <= t_lim (GDK468 MIT range, 11 Nm widened); while MOTORING (tau * w > 0)
    the FOC voltage limit without field weakening
        (R Iq + Ke w)^2 + (w_e L Iq)^2 <= (V_bus / sqrt3)^2,  tau = Kt Iq
    Kt 0.47 Nm/A, R 0.48 ohm (data sheet value taken as per phase: conservative),
    Ke = Kt / 1.5, L 368 uH, 14 pole pairs, gear 8 (w_e = 8 * 14 * w).
    22 V: 11 Nm at 0, 7.6 at 10 rad/s, 5.6 at 15, 4.1 at 20, 2.0 at 30.
  GIM3510-8 (ankle_pitch): 6.28 Nm stall, linear to 0 at 42.4 rad/s x V / 24.
  Braking (tau * w <= 0): the full t_lim (the motor is back-driven).

Per-env randomisation (``randomize_qdd_motor_envelope``): bus voltage (battery
state of charge and sag) and a strength factor (Kt / magnet / driver current
calibration, +-10 %) that scales the envelope (not the driver's clamp t_lim).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity

# GIM6010-8 (robot repo analysis/hop.py)
KT_6010 = 0.47  # Nm/A, output side
R_PH_6010 = 0.48  # ohm
KE_6010 = KT_6010 / 1.5  # V s/rad (24 V / sqrt3 / 44 rad/s = 0.315)
L_PH_6010 = 368e-6  # H
POLES_6010 = 14
GEAR_6010 = 8
# GIM3510-8
STALL_3510 = 6.28  # Nm
W0_3510_AT_24V = 42.4  # rad/s no-load at 24 V

V_BUS_NOMINAL = 22.0


def gim6010_motoring_limit(w: torch.Tensor, v_bus: torch.Tensor) -> torch.Tensor:
    """Largest motoring torque [Nm] of a GIM6010-8 at output speed w [rad/s] and bus
    voltage v_bus [V] (FOC voltage limit, no field weakening; before the t_lim clamp)."""
    aw = torch.abs(w)
    vm = v_bus / math.sqrt(3.0)
    e = KE_6010 * aw
    x = aw * (GEAR_6010 * POLES_6010 * L_PH_6010)
    a = R_PH_6010**2 + x * x
    b = 2.0 * R_PH_6010 * e
    c = e * e - vm * vm
    disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
    iq = (-b + torch.sqrt(disc)) / (2.0 * a)
    return torch.where(e < vm, KT_6010 * torch.clamp(iq, min=0.0), torch.zeros_like(iq))


def gim3510_motoring_limit(w: torch.Tensor, v_bus: torch.Tensor) -> torch.Tensor:
    w0 = W0_3510_AT_24V * v_bus / 24.0
    return torch.clamp(STALL_3510 * (1.0 - torch.abs(w) / w0), min=0.0)


def motoring_limit(motor: str, w: torch.Tensor, v_bus: torch.Tensor) -> torch.Tensor:
    if motor == "6010":
        return gim6010_motoring_limit(w, v_bus)
    if motor == "3510":
        return gim3510_motoring_limit(w, v_bus)
    raise ValueError(f"unknown motor {motor!r}")


@dataclass(kw_only=True)
class QddMotorActuatorCfg(IdealPdActuatorCfg):
    """Explicit PD + torque-speed envelope. ``effort_limit`` = driver clamp t_lim."""

    motor: str = "6010"  # "6010" (GIM6010-8) or "3510" (GIM3510-8)
    v_bus: float = V_BUS_NOMINAL

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> "QddMotorActuator":
        return QddMotorActuator(self, entity, target_ids, target_names)


class QddMotorActuator(IdealPdActuator[QddMotorActuatorCfg]):
    def __init__(self, cfg: QddMotorActuatorCfg, entity: Entity, target_ids: list[int], target_names: list[str]):
        super().__init__(cfg, entity, target_ids, target_names)
        self.v_bus: torch.Tensor | None = None  # [B, 1]
        self.strength: torch.Tensor | None = None  # [B, 1]
        self._vel: torch.Tensor | None = None

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        n = data.nworld
        self.v_bus = torch.full((n, 1), float(self.cfg.v_bus), device=device)
        self.strength = torch.ones((n, 1), device=device)
        self._vel = torch.zeros((n, len(self._target_names)), device=device)

    def apply_delay(self, cmd: ActuatorCmd) -> ActuatorCmd:
        """Delay the whole MIT command (position target, velocity target, torque
        feed-forward) by the same lag: they travel in one CAN frame on the robot.
        (mjlab's default delays only the position target.)"""
        if self._delay_buffer is None:
            return cmd
        n = cmd.position_target.shape[-1]
        self._delay_buffer.append(torch.cat((cmd.position_target, cmd.velocity_target, cmd.effort_target), dim=-1))
        out = self._delay_buffer.compute()
        return dataclasses.replace(
            cmd, position_target=out[..., :n], velocity_target=out[..., n : 2 * n], effort_target=out[..., 2 * n :]
        )

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self._vel is not None
        self._vel[:] = cmd.vel
        return super().compute(cmd)

    def available(self, effort: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(lo, hi) torque bounds at joint speed vel (per env / joint)."""
        assert self.force_limit is not None and self.v_bus is not None and self.strength is not None
        t_lim = self.force_limit
        mot = torch.minimum(t_lim, self.strength * motoring_limit(self.cfg.motor, vel, self.v_bus))
        hi = torch.where(vel > 0.0, mot, t_lim)  # positive torque motors when w > 0
        lo = -torch.where(vel < 0.0, mot, t_lim)
        return lo, hi

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        assert self._vel is not None
        lo, hi = self.available(effort, self._vel)
        return torch.clamp(effort, min=lo, max=hi)

    def set_envelope(self, env_ids, v_bus: torch.Tensor | None = None, strength: torch.Tensor | None = None) -> None:
        assert self.v_bus is not None and self.strength is not None
        if v_bus is not None:
            self.v_bus[env_ids] = v_bus.reshape(-1, 1)
        if strength is not None:
            self.strength[env_ids] = strength.reshape(-1, 1)


def _bus_power_terms(entity: "Entity", tau: torch.Tensor, w: torch.Tensor, small: torch.Tensor):
    """(A, Bq, P_small, V) per env for the bus current model of tasks/mdp.qdd_bus_current:
    GIM6010-8 power = tau w + 1.5 R (tau / Kt)^2 (A = sum tau w, Bq = sum of the copper
    term, both over the 6010 joints), GIM3510-8 = tau w / 0.7 motoring, x 0.7 braking."""
    act = next((a for a in entity.actuators if isinstance(a, QddMotorActuator) and a.v_bus is not None), None)
    v = act.v_bus[:, 0] if act is not None else torch.full((tau.shape[0],), V_BUS_NOMINAL, device=tau.device)
    kt = KT_6010 * act.strength if act is not None else KT_6010
    big = ~small
    pw = tau * w
    a = torch.where(big, pw, 0.0).sum(1)
    bq = torch.where(big, 1.5 * R_PH_6010 * torch.square(tau / kt), 0.0).sum(1)
    psm = torch.where(small, torch.where(pw > 0, pw / 0.7, pw * 0.7), 0.0).sum(1)
    return a, bq, psm, v


def install_bus_current_limiter(entity: "Entity", limit_a: float) -> None:
    """Software bus current limiter (robot side: the controller scales the GIM6010-8 torque
    commands so the predicted bus current stays below the LM5069 trip, 48.5 A min).
    Every physics substep, after all actuators computed their torques, the hip / knee
    (GIM6010-8) torques are scaled by one factor s in [0, 1] per env so that the predicted
    bus current (same model as tasks/mdp.qdd_bus_current) is <= limit_a. The ankles and
    arms (GIM3510-8, ~1 A) are not scaled. Patches the entity's _apply_actuator_controls;
    the last factor is kept in entity.bus_limiter_scale."""
    if getattr(entity, "_bus_limiter_installed", False):
        return
    names = list(entity.joint_names)
    orig = entity._apply_actuator_controls
    st: dict = {}

    def apply() -> None:
        orig()
        data = entity._data
        acts = [a for a in entity._custom_actuators if isinstance(a, QddMotorActuator)]
        if not acts:
            return
        w = data.joint_vel
        dev = w.device
        if not st:
            st["small"] = torch.tensor([n.endswith(("ankle_pitch", "shoulder_pitch", "elbow")) for n in names], device=dev)
            st["idx"] = [(a.ctrl_ids, a.target_ids, data.indexing.ctrl_ids[a.ctrl_ids]) for a in acts]
        tau = torch.zeros_like(w)
        for _, jids, gids in st["idx"]:
            tau[:, jids] = data.data.ctrl[:, gids]
        a, bq, psm, v = _bus_power_terms(entity, tau, w, st["small"])
        plim = limit_a * v
        c = psm - plim
        disc = torch.clamp(a * a - 4 * bq * c, min=0.0)
        s_quad = (-a + torch.sqrt(disc)) / torch.clamp(2 * bq, min=1e-9)
        s_lin = torch.where(a > 1e-9, -c / torch.clamp(a, min=1e-9), torch.ones_like(a))
        s = torch.where(bq > 1e-6, s_quad, s_lin).clamp(0.0, 1.0)
        s = torch.where(a + bq + psm > plim, s, torch.ones_like(s))
        entity.__dict__["bus_limiter_scale"] = s
        scale = torch.where(st["small"], torch.ones_like(tau), s[:, None])
        for cids, jids, _ in st["idx"]:
            data.write_ctrl(tau[:, jids] * scale[:, jids], cids)

    entity._apply_actuator_controls = apply
    entity.__dict__["_bus_limiter_installed"] = True
