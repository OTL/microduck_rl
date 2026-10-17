#!/usr/bin/env python3
"""Generate the fallen-state library for the QDD biped get-up task.

Every state is the robot ALREADY LYING STILL on a flat floor after a fall that
the real controller would produce, simulated in plain MuJoCo (CPU) with the
get-up collision model (robot/qdd_biped_constants.py get_qdd_biped_getup_spec):

  push   standing in the default pose with the walking PD gains, pushed at the
         torso (0.8-2.0 m/s, random direction). Like the controller: PD holds
         until the fall check fires (tilt > 35 deg and |gyro| > 3 rad/s, or
         tilt > 50 deg), then damping mode (Kp 0, Kd 0.5) until it lies still.
  stride as push, but from a random mid-stride pose (joints +-0.35 rad around
         the default, torso tilted up to 15 deg) and a weaker push: falls
         while walking.
  drop   any orientation, any joint angles within the limits (knee <=
         knee_max), dropped from 0.45 m in damping mode: odd poses (on the side,
         legs folded under the torso, ...).

A state is kept when it is at rest (|qvel| small), finite and within the joint
limits. Saved: root height, root quaternion (w, x, y, z), joint angles in
QDD_JOINT_NAMES order and the kind. The env randomises the yaw at reset.

  uv run python scripts/qdd_getup_states.py --n 3000 \
      --out src/mjlab_microduck/robot/qdd_biped/getup_states.npz
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.robot import qdd_biped_constants as rc

DT = 0.001
KD_DAMP = 0.5  # controller [motion] damping_kd
FALL_TILT_DEG, FALL_GYRO, MAX_TILT_DEG = 35.0, 3.0, 50.0  # controller [safety]
REST_QVEL = 0.1  # max |qvel| to count as lying still (0.2 s)
SLOW_QVEL = 0.5  # ... or still this slow after 6 s (creeping legs): kept, velocities zeroed
KINDS = ("push", "stride", "drop")
KIND_P = (0.4, 0.3, 0.3)


def build_model() -> mujoco.MjModel:
    spec = rc.get_qdd_biped_getup_spec()
    for g in spec.geoms:
        if not g.name.endswith("_collision"):
            g.contype = g.conaffinity = 0
            continue
        g.condim = 3
        g.priority = 1
        is_foot = g.name in rc.FOOT_GEOMS
        g.friction = [0.8 if is_foot else rc.BODY_FRICTION, 0.02, 0.01]
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.01],
        contype=1,
        conaffinity=1 | rc.GETUP_FLOOR_BIT,
    )
    spec.option.timestep = DT
    return spec.compile()


def _gains(m: mujoco.MjModel):
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
    return np.array(kp), np.array(kd), np.array(eff)


def _default_q() -> np.ndarray:
    k = math.radians(rc.CROUCH_KNEE_DEG)
    one = [0.0, 0.0, -k / 2, k, -k / 2]
    return np.array(one * 2)


def _quat_from_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _tilt_deg(d: mujoco.MjData) -> float:
    R = d.xmat[1].reshape(3, 3)  # torso = body 1
    return math.degrees(math.acos(np.clip(R[2, 2], -1.0, 1.0)))


def _knee_ok(q: np.ndarray, tol: float) -> bool:
    for s in (0, 5):
        if q[s + 3] > rc.knee_max(q[s + 2]) + tol:
            return False
    return True


def simulate(seed: int, kind: str, m: mujoco.MjModel):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    kp, kd, eff = _gains(m)
    lo, hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
    q0 = _default_q()
    qadr = 7

    if kind == "drop":
        q = rng.uniform(lo, hi)
        for s in (0, 5):
            q[s + 3] = min(q[s + 3], rc.knee_max(q[s + 2]) - 0.02)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        d.qpos[:3] = [0, 0, 0.45]
        d.qpos[3:7] = quat
        d.qpos[qadr:] = q
        d.qvel[:6] = rng.uniform(-1, 1, 6) * [0.3, 0.3, 0.3, 2, 2, 2]
        damping, target = True, q.copy()
    else:
        target = q0.copy()
        if kind == "stride":
            target = q0 + rng.uniform(-0.35, 0.35, 10)
            target = np.clip(target, lo + 0.02, hi - 0.02)
            for s in (0, 5):
                target[s + 3] = float(np.clip(target[s + 3], 0.1, rc.knee_max(target[s + 2]) - 0.02))
        d.qpos[:3] = [0, 0, rc.STAND_BASE_Z + 0.002]
        tilt_axis = rng.normal(size=3) * [1, 1, 0]
        tilt = math.radians(rng.uniform(0, 15)) if kind == "stride" else 0.0
        d.qpos[3:7] = (
            _quat_from_axis_angle(tilt_axis + [1e-9, 0, 0], tilt) if tilt > 0 else [1, 0, 0, 0]
        )
        d.qpos[qadr:] = q0
        # settle 0.3 s on the default pose, then move to the stride target
        damping = False
        for i in range(300):
            tau = np.clip(kp * (q0 - d.qpos[qadr:]) - kd * d.qvel[6:], -eff, eff)
            d.qfrc_applied[6:] = tau
            mujoco.mj_step(m, d)
        v = rng.uniform(0.8, 2.0) if kind == "push" else rng.uniform(0.3, 1.5)
        ang = rng.uniform(0, 2 * math.pi)
        d.qvel[0] += v * math.cos(ang)
        d.qvel[1] += v * math.sin(ang)
        d.qvel[3:6] += rng.uniform(-1, 1, 3)

    t_rest = 0.0
    for i in range(int(6.0 / DT)):
        q, qd = d.qpos[qadr:], d.qvel[6:]
        if not damping:
            tilt = _tilt_deg(d)
            gyro = np.linalg.norm(d.qvel[3:6])
            if (tilt > FALL_TILT_DEG and gyro > FALL_GYRO) or tilt > MAX_TILT_DEG:
                damping = True
        if damping:
            tau = np.clip(-KD_DAMP * qd, -eff, eff)
        else:
            tau = np.clip(kp * (target - q) - kd * qd, -eff, eff)
        d.qfrc_applied[6:] = tau
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            simulate.reason = "nan"
            return None
        if damping and np.abs(d.qvel).max() < REST_QVEL:
            t_rest += DT
            if t_rest > 0.2:
                break
        else:
            t_rest = 0.0
    if not damping:
        simulate.reason = "did not fall"
        return None
    if t_rest <= 0.2 and np.abs(d.qvel).max() > SLOW_QVEL:
        simulate.reason = f"moving (max |qvel| {np.abs(d.qvel).max():.2f})"
        return None
    q = d.qpos[qadr:].copy()
    if np.any(q < lo - 0.05) or np.any(q > hi + 0.05):
        simulate.reason = "joint range"
        return None
    if not _knee_ok(q, 0.1):
        simulate.reason = "knee_max"
        return None
    return float(d.qpos[2]), d.qpos[3:7].copy(), q, _tilt_deg(d)


def _work(args):
    seed, kind = args
    m = build_model()
    return kind, simulate(seed, kind, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(rc.QDD_GETUP_STATES_NPZ))
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    jobs = [(a.seed * 100000 + i, KINDS[rng.choice(3, p=KIND_P)]) for i in range(int(a.n * 1.3))]
    with mp.Pool(a.workers or None) as pool:
        res = pool.map(_work, jobs, chunksize=8)
    rows = [(k, r) for k, r in res if r is not None][: a.n]
    z = np.array([r[0] for _, r in rows], np.float32)
    quat = np.array([r[1] for _, r in rows], np.float32)
    q = np.array([r[2] for _, r in rows], np.float32)
    tilt = np.array([r[3] for _, r in rows], np.float32)
    kind = np.array([KINDS.index(k) for k, _ in rows], np.int8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, root_z=z, root_quat=quat, joint_pos=q, tilt_deg=tilt, kind=kind,
        kinds=np.array(KINDS), joint_names=np.array(rc.QDD_JOINT_NAMES),
    )
    print(f"{len(rows)} states ({len(res) - sum(r is None for _, r in res)} usable of {len(res)}) -> {a.out}")
    for i, k in enumerate(KINDS):
        sel = kind == i
        if sel.any():
            print(f"  {k:6s} {sel.sum():5d}  tilt {np.percentile(tilt[sel], [10, 50, 90]).round(0)} deg  "
                  f"z {np.percentile(z[sel], [10, 50, 90]).round(3)} m")
    up = tilt < 30
    print(f"  still upright-ish (tilt < 30 deg): {up.mean():.1%}")


if __name__ == "__main__":
    main()
