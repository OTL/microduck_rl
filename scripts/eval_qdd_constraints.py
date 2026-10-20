"""Check a QDD biped (with arms) policy against the robot's hardware limits, on CPU.

    uv run python scripts/eval_qdd_constraints.py walk  policy.onnx [--vx 0.3]
    uv run python scripts/eval_qdd_constraints.py rough policy.onnx [--vx 0.3]
    uv run python scripts/eval_qdd_constraints.py hop   policy.onnx [--armswing]
    uv run python scripts/eval_qdd_constraints.py getup policy.onnx [--kinds push bridge_supine ...]

Every physics substep (the hook on env.sim.step) records, per env:
  - motor bus current (mdp.qdd_bus_current, randomised bus voltage / strength); the largest
    5 ms average is what the LM5069 hot-swap sees (limit 48.5 A minimum since robo PR #49;
    training aims at <= 35 A for the battery sag)
  - the applied ankle torques: RMS over the run after the first second, the worse ankle of
    each env, and the heat ratio (RMS / rated)^2 (rated: GIM3510-8 1.0 Nm, or GIM6010-8
    5.0 Nm when QDD_ANKLE=GIM6010-8, robot/qdd_biped_constants.ANKLE_RATED_NM)
  - the arm bending moment at the shoulder / elbow joints (mdp.arm_bending_moment; torque
    sensors added to the scene here if the task has none)
plus the task outcome: walking speed and falls; hop foot clearance and falls; get-up
CoM rise above the start and peak per-foot landing force in m g; get-up
stood fraction (torso within 15 deg of upright, height within 3 cm of standing at the end).

The ONNX files are exported with batch 1: rows are run one at a time.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import qdd_biped_arms as A

from mjlab_microduck.robot.qdd_biped_constants import ANKLE_RATED_NM, QDD_ANKLE

BUS_LIMIT_A = 35.0
SKIP_S = 1.0  # spawn transient, left out of the RMS and the peaks (walking / rough)
TASKS = {
    "walk": A.VELOCITY_ARMS_TASK_ID,
    "rough": A.ROUGH_ARMS_TASK_ID,
    "hop": A.JUMP_ARMS_TASK_ID,
    "getup": A.GETUP_CHAIN_ARMS_TASK_ID,
}


def _add_moment_sensors(cfg) -> None:
    from mjlab.sensor import BuiltinSensorCfg, ObjRef

    have = {getattr(s, "prefixed_name", s.name) for s in cfg.scene.sensors}
    add = tuple(
        BuiltinSensorCfg(
            name=f"{s}_{j}_moment", sensor_type="torque", obj=ObjRef(type="site", name=f"{s}_{j}_site", entity="robot")
        )
        for s in "lr"
        for j in ("shoulder", "elbow")
        if f"robot/{s}_{j}_moment" not in have
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + add


def make_cfg(kind: str, a):
    task = A.JUMP_ARMSWING_TASK_ID if (kind == "hop" and a.armswing) else TASKS[kind]
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = a.n
    cfg.curriculum = {}
    if kind in ("walk", "rough"):
        tw = cfg.commands["twist"]
        tw.ranges.lin_vel_x = (a.vx, a.vx)
        tw.ranges.lin_vel_y = (0.0, 0.0)
        tw.ranges.ang_vel_z = (0.0, 0.0)
        for f in ("rel_standing_envs", "rel_turn_in_place_envs", "rel_forward_envs", "rel_heading_envs", "zero_command_prob"):
            if hasattr(tw, f):
                setattr(tw, f, 0.0)
        tw.resampling_time_range = (1.0e9, 1.0e9)
        cfg.episode_length_s = a.seconds
        cfg.events.pop("push_robot", None)
        if kind == "rough" and cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.max_init_terrain_level = a.terrain_level
    elif kind == "hop":
        cfg.commands["skill"].rsi_fraction = 0.0
    else:
        r = cfg.events["reset_getup"].params
        r["kind_probs"] = {k: 1.0 / len(a.kinds) for k in a.kinds}
        r["standing_prob"] = 0.0
        r["fixed_order"] = False
        r["frontiers"] = {}
    _add_moment_sensors(cfg)
    return cfg, task


def run(kind: str, onnx_path: str, a) -> dict:
    import onnxruntime as ort

    torch.manual_seed(a.seed)
    cfg, task = make_cfg(kind, a)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    sess = ort.InferenceSession(onnx_path)
    iname = sess.get_inputs()[0].name
    robot = env.scene["robot"]
    ankle_ids = [i for i, n in enumerate(robot.joint_names) if n.endswith("ankle_pitch")]
    sensors = A.ARM_MOMENT_SENSORS
    sub = dict(bus=[], ank=[], mom=[], sole=[], comz=[])
    sites = None
    if kind == "hop":
        from mjlab_microduck.tasks.qdd_skill import FOOT_HALF_LENGTH_M, FOOT_HALF_WIDTH_M

        sites = [robot.find_sites(n)[0][0] for n in ("l_foot_site", "r_foot_site")]
    orig = env.sim.step

    def hook():
        orig()
        sub["bus"].append(microduck_mdp.qdd_bus_current(env, robot).clone())
        sub["ank"].append(robot.data.qfrc_actuator[:, ankle_ids].clone())
        sub["mom"].append(microduck_mdp.arm_bending_moment(env, sensors).clone())
        if sites is not None:
            sub["sole"].append(microduck_mdp.sole_lowest_z(robot, sites, FOOT_HALF_LENGTH_M, FOOT_HALF_WIDTH_M).clone())
            sub["comz"].append(microduck_mdp.robot_com(env, robot)[0][:, 2].clone())

    env.sim.step = hook
    obs, _ = env.reset()
    if kind == "getup":
        gt = env.event_manager.get_term_cfg("reset_getup").func
        idx = gt.index.clamp(min=0)
        front = (gt.prone[idx] & ~gt.is_path[idx]).clone()
    if kind == "hop":
        steps = env.command_manager.get_term("skill").n - 1
    else:
        steps = int(round(cfg.episode_length_s / env.step_dt)) - 2
    B = a.n
    alive = torch.ones(B, dtype=torch.bool)
    vx_sum = torch.zeros(B)
    n_vx = 0
    t_skip = round(SKIP_S / env.step_dt)
    land_mg = torch.zeros(B)
    if kind == "hop":
        feet = env.scene["feet_ground_contact"]
        weight = microduck_mdp._robot_mass(env, robot) * 9.81
    for k in range(steps):
        x = obs["actor"].numpy().astype(np.float32)
        act = torch.tensor(np.concatenate([sess.run(None, {iname: x[i : i + 1]})[0] for i in range(B)]))
        obs, _, term, trunc, _ = env.step(act)
        alive &= ~term
        if kind == "hop":  # per-foot peak vertical force over the substeps of this step
            fh = feet.data.force_history
            f = fh[..., 2].abs().amax(dim=-1) if fh is not None else feet.data.force[..., 2].abs()
            land_mg = torch.maximum(land_mg, torch.nan_to_num(f).amax(dim=1) / weight)
        if k >= t_skip:  # body-frame forward speed (a slow yaw drift does not count against it)
            vx_sum += robot.data.root_link_lin_vel_b[:, 0]
            n_vx += 1
    out = dict(task=task, onnx=onnx_path, n=B, ankle=QDD_ANKLE, ankle_rated_nm=ANKLE_RATED_NM)
    if kind in ("walk", "rough"):
        out["vx_cmd"] = a.vx
        out["vx_mean"] = float((vx_sum / max(n_vx, 1))[alive].mean()) if alive.any() else float("nan")
        out["fell"] = float((~alive).float().mean())
    elif kind == "getup":
        up = microduck_mdp.qdd_torso_up(env)
        z = microduck_mdp.qdd_root_height(env)
        from mjlab_microduck.robot.qdd_biped_constants import STAND_BASE_Z

        ok = (up > math.cos(math.radians(15))) & ((z - STAND_BASE_Z).abs() < 0.03)
        out["stood"] = float(ok.float().mean())
        out["stood_front"] = float(ok[front].float().mean()) if front.any() else float("nan")
        out["n_front"] = int(front.sum())
    env.sim.step = orig
    dt = env.physics_dt
    skip = round((SKIP_S if kind in ("walk", "rough") else 0.3) / dt)
    bus = torch.stack(sub["bus"], 1)[:, skip:]
    nav = max(1, round(0.005 / dt))
    bus5 = torch.nn.functional.conv1d(bus.reshape(B, 1, -1), torch.ones(1, 1, nav) / nav).reshape(B, -1)
    ank = torch.stack(sub["ank"], 1)[:, skip:]  # [B, T, 2]
    ank_rms = torch.sqrt(torch.square(ank).mean(dim=1)).amax(dim=1)  # worse ankle per env
    mom = torch.stack(sub["mom"], 1)[:, skip:]
    q = lambda t, p: float(torch.quantile(t.flatten().float(), p))  # noqa: E731
    pk = bus5.amax(dim=1)
    out.update(
        bus5_peak_median_a=q(pk, 0.5),
        bus5_peak_max_a=float(pk.max()),
        bus5_over_limit_frac=float((pk > BUS_LIMIT_A).float().mean()),
        bus_mean_a=float(bus.mean()),
        ankle_rms_nm_mean=float(ank_rms.mean()),
        ankle_rms_nm_max=float(ank_rms.max()),
        ankle_heat_ratio_mean=float((ank_rms / ANKLE_RATED_NM).square().mean()),
        ankle_heat_ratio_max=float((ank_rms / ANKLE_RATED_NM).square().max()),
        arm_moment_p99_nm=q(mom, 0.99),
        arm_moment_max_nm=float(mom.max()),
        arm_moment_env_peak_median_nm=q(mom.amax(dim=1), 0.5),
    )
    if sites is not None:
        sole = torch.stack(sub["sole"], 1)[:, skip:]
        clear = sole.amax(dim=1)
        out["foot_clear_cm_median"] = 100 * q(clear, 0.5)
        comz = torch.stack(sub["comz"], 1)
        rise = comz.amax(dim=1) - comz[:, 0]
        out["com_rise_cm_median"] = 100 * q(rise, 0.5)
        out["landing_mg_median"] = q(land_mg, 0.5)
        out["landing_mg_max"] = float(land_mg.max())
        out["fell"] = float((~alive).float().mean())
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(TASKS))
    ap.add_argument("onnx")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--terrain-level", type=int, default=5)
    ap.add_argument("--armswing", action="store_true", help="hop: the arm-swing task")
    ap.add_argument("--kinds", nargs="*", default=["push", "stride", "drop", "crouch", "bridge_supine", "path"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = run(a.kind, a.onnx, a)
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
