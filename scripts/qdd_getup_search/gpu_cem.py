"""GPU keyframe search (CEM on MuJoCo Warp): a both-feet get-up for the QDD biped with arms.

Each candidate is a left/right-mirrored keyframe motion: from wherever the fall left
the joints, ramp through NK keys (hip pitch, knee, ankle, shoulder [, elbow]) to the
default pose. With --asym each side has its own keys, plus hip roll. Every candidate is scored on NS real falls from the fall library
(prone or supine), with floor friction x0.6-1.2 and kp x0.9-1.1 randomised, so one
generation is POP*NS parallel worlds. Score: ends standing (bonus), torso up and at
standing height, plus the best "upright x height" reached on the way (shaping).

The arms are a design study (OTL asked for arms so mochi can get up with both feet
together): two shoulder-pitch joints on the torso sides, optionally an elbow each.
Run on the GPU box (yatai):

  uv run python scripts/qdd_getup_search/gpu_cem.py --face prone --arm-len 0.20 \
      --elbow --iters 150 --out logs/getup_arms/prone_L20_elbow.npz

The CAD v2 arm of the training model (25 cm = upper 10 + fore 15, plates at y 145 /
178.5 mm, GIM3510-8 on both joints, elbow + bends forward):

  uv run python scripts/qdd_getup_search/gpu_cem.py --face supine --cad \
      --iters 150 --out logs/getup_arms/supine_cad.npz

Plain-MuJoCo replay / video: scripts/qdd_getup_search/arms_replay.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot import qdd_biped_constants as rc

DT = 0.002


@dataclass
class ArmCfg:
    length: float = 0.20  # shoulder axis -> hand tip (upper + fore arm with an elbow)
    elbow: bool = False
    shoulder: tuple = (0.0, 0.103, 0.112)  # torso frame, outside the side wall
    sh_range: tuple = (-3.3, 1.2)  # 0 hanging down, -pi/2 forward, + backward
    el_range: tuple = (0.0, 2.36)  # 0 straight, + bends the forearm forward
    kp: float = 30.0
    kd: float = 0.6
    effort: float = 5.6  # GIM4310-10 peak
    mass_motor: float = 0.227
    mass_link: float = 0.06
    # --cad: the CAD v2 arm (robot/qdd_biped_constants.py ARM_*); None = the old study
    upper: float | None = None  # shoulder -> elbow (None: length / 2)
    upper_y: float = 0.045  # link capsule y from the shoulder
    fore_y: float | None = None  # forearm capsule y (None: upper_y)
    mass_upper: float | None = None  # (None: mass_link)
    mass_fore: float | None = None
    mass_shoulder_motor: float | None = None  # (None: mass_motor)
    sh_armature: float = 1.5e-3  # GIM4310-10 (--cad: GIM3510-8)


def arm_cfg(a) -> ArmCfg:
    """ArmCfg from the search args; --cad uses the CAD v2 arm the training model has
    (25 cm = upper 10 + fore 15, plates at y 145 / 178.5 mm, GIM3510-8 on both)."""
    sh = (-3.3, a.sh_back)
    if not getattr(a, "cad", False):
        return ArmCfg(length=a.arm_len, elbow=a.elbow, sh_range=sh)
    return ArmCfg(
        length=rc.ARM_LENGTH,
        elbow=True,
        sh_range=sh,
        el_range=rc.ARM_ELBOW_RANGE,
        kp=rc.ARM_KP,
        kd=rc.ARM_KD,
        effort=rc.ARM_PEAK,
        upper=rc.ARM_UPPER,
        upper_y=rc.ARM_UPPER_Y,
        fore_y=rc.ARM_FORE_Y,
        mass_upper=rc.ARM_UPPER_MASS,
        mass_fore=rc.ARM_FORE_MASS,
        mass_shoulder_motor=rc.ARM_MOTOR["mass"] + rc.ARM_SHOULDER_MOUNT_MASS,
        sh_armature=rc.ARM_MOTOR["armature"],
    )


def build_model(arm: ArmCfg) -> mujoco.MjModel:
    """Get-up collision model + floor + arms (bodies after the legs: qpos 7..16 legs, then arms)."""
    spec = rc.get_qdd_biped_getup_spec()
    torso = spec.body(rc.TORSO_BODY)
    for s, nm in ((1, "l"), (-1, "r")):
        b = torso.add_body(
            name=f"{nm}_arm",
            pos=[arm.shoulder[0], s * arm.shoulder[1], arm.shoulder[2]],
        )
        b.add_joint(
            name=f"{nm}_shoulder",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            range=list(arm.sh_range),
            armature=arm.sh_armature,
            damping=0.05,
            frictionloss=0.05,
        )
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.0265, 0.02, 0],
            pos=[0, s * 0.02, 0],
            quat=[0.7071, 0.7071, 0, 0],
            mass=arm.mass_motor if arm.mass_shoulder_motor is None else arm.mass_shoulder_motor,
            contype=0,
            conaffinity=0,
            group=2,
        )
        y = s * arm.upper_y
        yf = y if arm.fore_y is None else s * arm.fore_y
        if arm.elbow:
            up = arm.length / 2 if arm.upper is None else arm.upper
            b.add_geom(
                name=f"{nm}_upper_arm_collision",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, y, 0, 0, y, -up],
                size=[0.014, 0, 0],
                mass=arm.mass_link if arm.mass_upper is None else arm.mass_upper,
            )
            f = b.add_body(name=f"{nm}_forearm", pos=[0, 0, -up])
            f.add_joint(
                name=f"{nm}_elbow",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, -1, 0],  # + bends the forearm forward (+x)
                range=list(arm.el_range),
                armature=0.00064,
                damping=0.05,
                frictionloss=0.05,
            )
            f.add_geom(  # elbow motor (ankle class, GIM3510-8)
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.023, 0.026, 0],
                pos=[0, y, 0],  # CAD v2: elbow motor centre y 42 mm = the upper-arm plate
                quat=[0.7071, 0.7071, 0, 0],
                mass=0.26,
                contype=0,
                conaffinity=0,
                group=2,
            )
            f.add_geom(
                name=f"{nm}_forearm_collision",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, yf, 0, 0, yf, -(arm.length - up)],
                size=[0.014, 0, 0],
                mass=arm.mass_link if arm.mass_fore is None else arm.mass_fore,
            )
        else:
            b.add_geom(
                name=f"{nm}_arm_collision",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, y, 0, 0, y, -arm.length],
                size=[0.014, 0, 0],
                mass=arm.mass_link,
            )
    for g in spec.geoms:
        if not g.name.endswith("_collision"):
            if g.group == 2 or g.contype == 0:
                g.contype = g.conaffinity = 0
            continue
        g.contype, g.conaffinity = 1, 1
        g.condim, g.priority = 3, 1
        g.friction = [0.8 if g.name in rc.FOOT_GEOMS else rc.BODY_FRICTION, 0.02, 0.01]
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.01],
        contype=1,
        conaffinity=1 | rc.GETUP_FLOOR_BIT,
    )
    spec.option.timestep = DT
    return spec.compile()


def leg_gains():
    kp, kd, eff = [], [], []
    for n in rc.QDD_JOINT_NAMES:
        a = (
            rc.QDD_ACTUATOR_HIP_YAW_ROLL
            if n.endswith(("hip_yaw", "hip_roll"))
            else rc.QDD_ACTUATOR_HIP_PITCH_KNEE
            if n.endswith(("hip_pitch", "knee"))
            else rc.QDD_ACTUATOR_ANKLE
        )
        kp.append(a.stiffness)
        kd.append(a.damping)
        eff.append(a.effort_limit)
    return kp, kd, eff


# No-load speeds (rad/s), robo robot/model.py MOTORS: GIM6010-8 420 rpm, GIM3510-8 405 rpm,
# GIM4310-10 228 rpm. The torque available in the direction the joint already turns falls
# linearly to 0 at this speed (back-EMF); braking keeps the full torque.
VMAX_LEG = [44.0, 44.0, 44.0, 44.0, 42.4] * 2
VMAX_SHOULDER, VMAX_ELBOW = 23.9, 42.4


def torque_limits(qd, eff, vmax):
    """(lower, upper) torque bounds with the linear torque-speed limit (numpy or torch)."""
    if isinstance(qd, np.ndarray):
        up = eff * np.clip(1 - np.maximum(qd, 0) / vmax, 0, 1)
        lo = -eff * np.clip(1 - np.maximum(-qd, 0) / vmax, 0, 1)
    else:
        up = eff * torch.clamp(1 - qd.clamp(min=0) / vmax, 0, 1)
        lo = -eff * torch.clamp(1 - (-qd).clamp(min=0) / vmax, 0, 1)
    return lo, up


def vmax_list(elbow: bool, cad: bool = False) -> list[float]:
    sh = VMAX_ELBOW if cad else VMAX_SHOULDER  # --cad: GIM3510-8 on the shoulder too
    arm = [sh, VMAX_ELBOW] if elbow else [sh]
    return VMAX_LEG + arm * 2


def default_legs() -> np.ndarray:
    k = math.radians(rc.CROUCH_KNEE_DEG)
    return np.array([0.0, 0.0, -k / 2, k, -k / 2] * 2)


def fall_starts(face: str, n: int, seed: int) -> np.ndarray:
    """Real falls (push / stride / drop) lying on the front (prone) or back (supine): qpos (n, 17)."""
    lib = np.load(str(rc.QDD_GETUP_STATES_NPZ))
    kinds = list(lib["kinds"])
    w, x, y, z = lib["root_quat"].T
    prone = 2 * (x * z - w * y) < 0
    sel = np.isin(lib["kind"], [kinds.index(k) for k in ("push", "stride", "drop")])
    sel &= prone if face == "prone" else ~prone
    ids = np.random.default_rng(seed).choice(np.flatnonzero(sel), n, replace=False)
    q = np.zeros((n, 17))
    q[:, 2] = lib["root_z"][ids] + 0.003
    q[:, 3:7] = lib["root_quat"][ids]
    q[:, 7:] = lib["joint_pos"][ids]
    return q


class Search:
    def __init__(self, a, device):
        from mjlab.sim.sim import Simulation, SimulationCfg

        self.a, self.dev = a, device
        self.arm = arm_cfg(a)
        self.m = build_model(self.arm)
        self.nj = self.m.nq - 7
        self.npk = n_per_key(a)  # params per key
        self.W = a.pop * a.ns
        self.sim = Simulation(
            self.W, SimulationCfg(nconmax=64, njmax=256), self.m, device
        )
        self.sim.expand_model_fields(("geom_friction",))
        kp, kd, eff = leg_gains()
        na = self.nj - 10
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
        self.kp = t(kp + [self.arm.kp] * na)
        self.kd = t(kd + [self.arm.kd] * na)
        self.eff = t(eff + [self.arm.effort] * na)
        self.vmax = t(vmax_list(a.elbow, getattr(a, "cad", False)))
        self.lo = t(self.m.jnt_range[1:, 0])
        self.hi = t(self.m.jnt_range[1:, 1])
        self.q0 = t(np.r_[default_legs(), np.zeros(na)])
        st = fall_starts(a.face, a.n_starts, a.seed)
        self.starts = t(np.c_[st, np.zeros((len(st), na))])
        self.fr0 = self.sim.get_default_field("geom_friction").clone()
        self.foot_b = [self.m.body(f"{s}_foot").id for s in "lr"]

    def keys(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(P, NK*npk + NK) -> joint targets (P, NK, nj) and durations (P, NK)."""
        P, NK, k = X.shape[0], self.a.nk, self.npk
        p = X[:, : NK * k].reshape(P, NK, k)
        z = torch.zeros(P, NK, device=self.dev)
        if getattr(self.a, "asym", False):
            # each side its own [hip_roll, hip_pitch, knee, ankle, shoulder (, elbow)]
            h = k // 2
            sides = []
            for s in (p[..., :h], p[..., h:]):
                sides.append(torch.stack([z, s[..., 0], s[..., 1], s[..., 2], s[..., 3]], -1))
            q = [sides[0], sides[1], p[..., 4:h], p[..., h + 4 :]]
        else:
            leg = torch.stack([z, z, p[..., 0], p[..., 1], p[..., 2]], -1)
            legr = leg.clone()
            q = [leg, legr, p[..., 3:4], p[..., 3:4]]
            if self.a.elbow:
                q = [leg, legr, p[..., 3:4], p[..., 4:5], p[..., 3:4], p[..., 4:5]]
        # joint order: l leg (5), r leg (5), then l arm chain, r arm chain
        q = torch.cat(q, -1)
        q = torch.clamp(q, self.lo, self.hi)
        hp = q[..., [2, 7]]
        hpd = torch.rad2deg(hp)
        kmax = torch.deg2rad(
            torch.where(
                hpd > 0,
                80.0,
                torch.where(
                    hpd > -15,
                    90.0,
                    torch.where(hpd > -30, 100.0, torch.where(hpd > -45, 110.0, 120.0)),
                ),
            )
        )
        q[..., [3, 8]] = torch.minimum(q[..., [3, 8]], kmax)
        durs = torch.clamp(
            X[:, NK * k : NK * k + NK], getattr(self.a, "min_dur", 0.05), 1.5
        )
        return q, durs

    @torch.no_grad()
    def evaluate(
        self,
        X: torch.Tensor,
        start_ids: torch.Tensor,
        gen: torch.Generator,
        record=None,
    ):
        a, W, ns = self.a, self.W, self.a.ns
        P = X.shape[0]
        keys, durs = self.keys(X)  # (P, NK, nj), (P, NK)
        keys = keys.repeat_interleave(ns, 0)
        durs = durs.repeat_interleave(ns, 0)  # (W, ...)
        d = self.sim.data
        qs = self.starts[start_ids.repeat(P)]
        self.sim.reset()
        d.qpos[:] = qs
        d.qvel[:] = 0
        fr = (
            torch.empty(W, device=self.dev).uniform_(0.6, 1.2, generator=gen)
            if a.dr
            else torch.ones(W, device=self.dev)
        )
        self.sim.model.geom_friction[:] = self.fr0.unsqueeze(0) * torch.stack(
            [fr, torch.ones_like(fr), torch.ones_like(fr)], -1
        ).unsqueeze(1)
        kps = (
            torch.empty(W, device=self.dev).uniform_(0.9, 1.1, generator=gen)
            if a.dr
            else torch.ones(W, device=self.dev)
        ).unsqueeze(1) * self.kp
        self.sim.forward()
        cur = d.qpos[:, 7:].clone()
        prep = a.prep
        allk = torch.cat(
            [keys, self.q0.expand(W, 1, -1)], 1
        )  # K0..K(NK-1), default pose
        bounds = torch.cumsum(
            durs, 1
        )  # segment j: allk[j] -> allk[j+1], ends at bounds[j] (after prep)
        T = prep + durs.sum(1).max().item() + a.hold
        n = int(T / DT)
        prog = torch.zeros(W, device=self.dev)
        spin = torch.zeros(W, device=self.dev)  # peak torso angular speed
        rows = []
        ar = torch.arange(W, device=self.dev)
        zero = torch.zeros(W, 1, device=self.dev)
        for i in range(n):
            t = (i + 1) * DT
            if t <= prep:
                tgt = cur + (t / prep) * (allk[:, 0] - cur)
            else:
                tp = t - prep
                k = (tp >= bounds).sum(1)  # 0..NK
                kc = k.clamp(max=a.nk - 1)
                start = torch.cat([zero, bounds], 1)[ar, kc]
                frac = ((tp - start) / durs[ar, kc]).clamp(0, 1)
                frac = torch.where(k >= a.nk, torch.ones_like(frac), frac)
                tgt = allk[ar, kc] + frac.unsqueeze(1) * (
                    allk[ar, kc + 1] - allk[ar, kc]
                )
            q, qd = d.qpos[:, 7:], d.qvel[:, 6:]
            tau = kps * (tgt - q) - self.kd * qd
            if a.speed_limit:
                lo, hi = torque_limits(qd, self.eff, self.vmax)
                d.qfrc_applied[:, 6:] = torch.maximum(torch.minimum(tau, hi), lo)
            else:
                d.qfrc_applied[:, 6:] = torch.clamp(tau, -self.eff, self.eff)
            self.sim.step()
            if i % 10 == 0:
                up = d.xmat[:, 1, 2, 2]
                spin = torch.maximum(spin, d.qvel[:, 3:6].norm(dim=1))
                prog = torch.maximum(
                    prog, up * torch.clamp(d.qpos[:, 2] / rc.STAND_BASE_Z, max=1.0)
                )
                if record is not None and i % 10 == 0:
                    rows.append(d.qpos[record].cpu().numpy().copy())
        up = d.xmat[:, 1, 2, 2]
        z = d.qpos[:, 2]
        bad = ~torch.isfinite(d.qpos).all(1)
        ok = (up > 0.9) & ((z - rc.STAND_BASE_Z).abs() < 0.03) & ~bad
        s = (
            up
            + 1
            - torch.clamp((z - rc.STAND_BASE_Z).abs() / 0.05, max=3)
            - 0.5 * d.qvel[:, :6].norm(dim=1).clamp(max=6)
            + 2 * prog
            - a.spin_w * torch.clamp(spin - a.spin_free, min=0)
            + 2 * ok.float()
        )
        s = torch.where(bad, torch.full_like(s, -10.0), s)
        if record is not None:
            return np.array(rows), ok[record].item()
        return s.view(P, ns).mean(1), ok.view(P, ns).float().mean(1)


def n_per_key(a) -> int:
    """Search parameters per keyframe: mirrored = one side, --asym = both sides + hip roll."""
    side = 5 if a.elbow else 4
    return 2 * (side + 1) if getattr(a, "asym", False) else side


def sym_to_asym(x: np.ndarray, nk: int, elbow: bool) -> np.ndarray:
    """A mirrored parameter vector as the same motion in --asym form (hip roll 0)."""
    side = 5 if elbow else 4
    k = x[: nk * side].reshape(nk, side)
    one = np.c_[np.zeros(nk), k]
    return np.r_[np.c_[one, one].ravel(), x[nk * side :]]


def seed_keys(face: str, nk: int, elbow: bool) -> np.ndarray:
    d0 = [-0.39, 0.79, -0.39, 0.0]
    if face == "prone":  # push the chest up with the arms, tuck the legs, table, stand
        ks = [
            [0, 0, 0, -1.57],
            [-0.8, 0.8, 0, -1.57],
            [-1.57, 0.2, 0.0, -1.57],
            [-1.5, 0.4, -0.2, -1.2],
            [-1.6, 1.2, -0.6, -0.8],
            [-1.2, 1.4, -0.6, 0],
            d0,
            d0,
        ]
    else:  # sit up on the arms behind, feet in, push forward over the feet
        ks = [
            [-1.0, 1.8, -0.6, 0.6],
            [-1.0, 1.8, -0.6, 1.2],
            [-1.3, 2.0, -0.7, 1.0],
            [-1.6, 2.09, -0.75, 0.6],
            [-1.6, 2.09, -0.75, 0.2],
            [-1.2, 1.4, -0.6, 0],
            d0,
            d0,
        ]
    ks = ks[:nk]
    if elbow:
        ks = [k + [0.0] for k in ks]
    return np.r_[np.concatenate(ks), [0.5] * nk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", default="prone", choices=("prone", "supine"))
    ap.add_argument("--arm-len", type=float, default=0.20)
    ap.add_argument("--elbow", action="store_true")
    ap.add_argument(
        "--cad", action="store_true", help="the CAD v2 arm of the training model (sets --arm-len 0.25 --elbow)"
    )
    ap.add_argument(
        "--sh-back", type=float, default=2.2, help="shoulder range backward (rad)"
    )
    ap.add_argument("--nk", type=int, default=7)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument(
        "--ns", type=int, default=8, help="falls per candidate per generation"
    )
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--prep", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--spin-free", type=float, default=6.0, help="torso rad/s that costs nothing"
    )
    ap.add_argument(
        "--spin-w", type=float, default=0.3, help="score per rad/s above --spin-free"
    )
    ap.add_argument(
        "--no-speed-limit",
        dest="speed_limit",
        action="store_false",
        help="torque clip only (results before 2026-10-18 03:00)",
    )
    ap.add_argument(
        "--no-dr",
        dest="dr",
        action="store_false",
        help="no friction / kp randomisation",
    )
    ap.add_argument(
        "--asym",
        action="store_true",
        help="left and right keys searched separately, plus hip roll",
    )
    ap.add_argument("--init", type=Path, help="start from a result (.npz)")
    ap.add_argument("--init-which", default="best", choices=("best", "mu"))
    ap.add_argument("--sd0", type=float, default=0.4, help="initial key spread (rad)")
    ap.add_argument("--mu-lr", type=float, default=0.7, help="CEM mean step toward the elite")
    ap.add_argument("--sd-floor", type=float, default=0.005)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cad:
        a.arm_len, a.elbow = rc.ARM_LENGTH, True
    S = Search(a, a.device)
    gen = torch.Generator(device=a.device)
    gen.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.init:
        f = np.load(a.init)
        mu = f[a.init_which] if a.init_which in f and f[a.init_which].ndim else f["mu"]
    else:
        mu = seed_keys(a.face, a.nk, a.elbow)
    nkey = a.nk * S.npk
    if a.asym and mu.size == a.nk * (5 if a.elbow else 4) + a.nk:
        mu = sym_to_asym(mu, a.nk, a.elbow)
    assert mu.size == nkey + a.nk, (mu.size, nkey + a.nk)
    sd = np.r_[[a.sd0] * nkey, [0.15] * a.nk]
    best = (-1e9, None, 0.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for it in range(a.iters):
        t0 = time.time()
        X = mu + sd * rng.normal(size=(a.pop, mu.size))
        X[0] = mu
        if best[1] is not None:  # elitism: the best so far is re-scored every generation
            X[1] = best[1]
        ids = torch.tensor(rng.choice(a.n_starts, a.ns, replace=False), device=a.device)
        s, ok = S.evaluate(
            torch.tensor(X, dtype=torch.float32, device=a.device), ids, gen
        )
        s, ok = s.cpu().numpy(), ok.cpu().numpy()
        el = np.argsort(s)[-max(8, a.pop // 16) :]
        mu = (1 - a.mu_lr) * mu + a.mu_lr * X[el].mean(0)
        sd = 0.5 * sd + 0.5 * X[el].std(0) + a.sd_floor
        if s[el[-1]] > best[0]:
            best = (s[el[-1]], X[el[-1]].copy(), ok[el[-1]])
        print(
            f"{it} best {s.max():.3f} ok(best) {ok[el[-1]]:.2f} ok(elite) {ok[el].mean():.2f} mu-ok {ok[0]:.2f} {time.time() - t0:.1f}s",
            flush=True,
        )
        np.savez(
            a.out,
            mu=mu,
            sd=sd,
            best=best[1],
            best_score=best[0],
            best_ok=best[2],
            args=json.dumps(
                {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
            ),
        )
    print("done", a.out)


if __name__ == "__main__":
    main()
