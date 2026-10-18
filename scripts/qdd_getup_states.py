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
  path   states along get-up motions found by a keyframe search
         (scripts/qdd_getup_search/ -> robot/qdd_biped/getup_demo_traj.npz:
         qpos / qvel every 20 ms), WITH their velocities, mirrored left/right
         at random and with small joint / velocity noise: reverse curriculum
         along a motion known to work (prone -> one leg kicks back, torso up
         into a split -> legs together -> standing).
  crouch reverse-curriculum starts (part way up, not a fall): standing, the PD
         drives to a random deep crouch / kneel (knee 60-120 deg) with the torso
         tilted 0-25 deg and a small push, and holds it (no damping) until it
         rests: squatting, kneeling, sitting on the floor. Gives the policy
         on-policy data near the end of a get-up (AGENTS.md: reverse
         curriculum for "learns the start, never the last mile").

Velocities (root_vel: world-frame linear + angular, joint_vel) are zero
except for "path" states. A simulated state is kept when it is at rest (|qvel| small), finite and within the joint
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
KINDS = ("push", "stride", "drop", "crouch", "path", "bridge_prone", "bridge_supine")
KIND_P = (0.3, 0.25, 0.2, 0.25)  # simulated kinds; "path" rows are added on top (--n-path)
SIM_KINDS = KINDS[:4]
FALL_KINDS = ("push", "stride", "drop")
DEMO_DT = 0.02  # s between demo frames (scripts/qdd_getup_search/export_traj.py)
# --arms: the model with arms (robot/qdd_biped_constants.py add_arms); joints =
# legs then l/r shoulder pitch + elbow. Set in main() before the worker pool forks.
ARMS = False


def joint_names() -> tuple:
    return rc.QDD_ARMS_JOINT_NAMES if ARMS else rc.QDD_JOINT_NAMES


def nj() -> int:
    return len(joint_names())


def build_model() -> mujoco.MjModel:
    spec = rc.get_qdd_biped_getup_arms_spec() if ARMS else rc.get_qdd_biped_getup_spec()
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
    for n in joint_names():
        a = (
            rc.QDD_ACTUATOR_ARM
            if n.endswith(("shoulder_pitch", "elbow"))
            else rc.QDD_ACTUATOR_HIP_YAW_ROLL
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
    return np.array(one * 2 + [0.0] * (nj() - 10))


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
    elif kind == "crouch":
        knee = math.radians(rng.uniform(60, 120))
        target = np.zeros(nj())
        target[10:] = rng.uniform(-0.3, 0.3, nj() - 10)  # arms (if any) near hanging
        for s in (0, 5):
            hp = -knee / 2 + rng.uniform(-0.5, 0.3)
            target[s : s + 5] = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), hp, knee, -knee / 2 + rng.uniform(-0.3, 0.3)]
        target = np.clip(target, lo + 0.02, hi - 0.02)
        for s in (0, 5):
            target[s + 3] = min(target[s + 3], rc.knee_max(target[s + 2]) - 0.02)
        tilt_axis = rng.normal(size=3) * [1, 1, 0] + [1e-9, 0, 0]
        d.qpos[:3] = [0, 0, rc.STAND_BASE_Z + 0.002]
        d.qpos[3:7] = _quat_from_axis_angle(tilt_axis, math.radians(rng.uniform(0, 25)))
        d.qpos[qadr:] = q0
        d.qvel[:3] = rng.uniform(-0.15, 0.15, 3)
        damping = False
    else:
        target = q0.copy()
        if kind == "stride":
            target = q0 + rng.uniform(-0.35, 0.35, nj())
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
        if not damping and kind != "crouch":
            tilt = _tilt_deg(d)
            gyro = np.linalg.norm(d.qvel[3:6])
            if (tilt > FALL_TILT_DEG and gyro > FALL_GYRO) or tilt > MAX_TILT_DEG:
                damping = True
        if damping:
            tau = np.clip(-KD_DAMP * qd, -eff, eff)
        else:
            tgt = q0 + min(1.0, i * DT / 0.5) * (target - q0) if kind == "crouch" else target
            tau = np.clip(kp * (tgt - q) - kd * qd, -eff, eff)
        d.qfrc_applied[6:] = tau
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            simulate.reason = "nan"
            return None
        if (damping or kind == "crouch") and i > 500 and np.abs(d.qvel).max() < REST_QVEL:
            t_rest += DT
            if t_rest > 0.2:
                break
        else:
            t_rest = 0.0
    if not damping and kind != "crouch":
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
    return float(d.qpos[2]), d.qpos[3:7].copy(), q, _tilt_deg(d), np.zeros(6), np.zeros(nj())


def _swap_lr(j: np.ndarray) -> np.ndarray:
    """Joint vector mirrored: legs swapped with hip yaw / roll negated, arms
    (pitch axes) swapped."""
    l, r = j[:5].copy(), j[5:10].copy()
    for leg in (l, r):
        leg[0] *= -1  # hip yaw
        leg[1] *= -1  # hip roll
    out = np.r_[r, l]
    if len(j) > 10:
        out = np.r_[out, j[12:14], j[10:12]]
    return out


def _mirror(qpos: np.ndarray, qvel_world: np.ndarray):
    """Left/right mirror image (reflection y -> -y) of a state."""
    q = qpos.copy()
    q[1] *= -1
    w, x, y, z = q[3:7]
    q[3:7] = [w, -x, y, -z]
    q[7:] = _swap_lr(q[7:])
    v = qvel_world.copy()
    v[1] *= -1  # linear y
    v[3] *= -1  # angular x
    v[5] *= -1  # angular z
    v[6:] = _swap_lr(v[6:])
    return q, v


def path_remain_s(demo: np.ndarray, dt: float = DEMO_DT) -> np.ndarray:
    """Per demo frame: the demo time left until standing [s]. Demo 0 ends
    standing; every other demo (the supine roll) ends lying prone, at the start
    of demo 0, so demo 0's whole length is added."""
    out = np.zeros(len(demo), np.float32)
    n0 = int(np.sum(demo == 0))
    for k in np.unique(demo):
        idx = np.flatnonzero(demo == k)
        # the arms demo is a supine get-up alone (demo 1, no demo 0): ends standing
        out[idx] = (len(idx) - 1 - np.arange(len(idx))) * dt + (0 if k == 0 else max(n0 - 1, 0) * dt)
    return out


def sample_path_states(path: Path, n: int, seed: int, m: mujoco.MjModel):
    """n states along the demo trajectories (velocities in the world frame),
    random mirror, joint noise 0.03 rad, velocity scale 0.8-1.2. Each row ends
    with the demo time left until standing (path_remain_s)."""
    rng = np.random.default_rng(seed)
    dd = np.load(path)
    Q, V = dd["qpos"].astype(np.float64), dd["qvel"].astype(np.float64)
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    demo = dd["demo"] if "demo" in dd.files else np.zeros(len(Q), np.int8)
    frames = [np.flatnonzero(demo == k) for k in np.unique(demo)]
    remain = path_remain_s(demo)
    rows = []
    tries = 0
    while len(rows) < n:
        tries += 1
        if tries > 20 * n:
            raise SystemExit(f"path states: only {len(rows)} of {n} without self-penetration")
        i = rng.choice(frames[rng.integers(len(frames))])  # each demo equally often
        q, v = Q[i].copy(), V[i].copy()
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R = d.xmat[1].reshape(3, 3)
        vw = v.copy()
        vw[3:6] = R @ v[3:6]  # free-joint angular velocity: body -> world frame
        if rng.random() < 0.5:
            q, vw = _mirror(q, vw)
        q[7:] = np.clip(q[7:] + rng.normal(0, 0.03, nj()), lo, hi)
        for s in (0, 5):
            q[7 + s + 3] = min(q[7 + s + 3], rc.knee_max(q[7 + s + 2]))
        vw *= rng.uniform(0.8, 1.2)
        d.qpos[:] = q
        q[2] += _floor_lift(m, d)  # a demo made on another arm length can go into the floor
        d.qpos[:] = q
        if _self_penetration(m, d) > 0.003:
            continue  # e.g. the 20 cm demo's forearm in the thigh with the 25 cm arms
        mujoco.mj_kinematics(m, d)
        rows.append(
            (float(q[2]), q[3:7].copy(), q[7:].copy(), _tilt_deg(d), vw[:6].copy(), vw[6:].copy(), float(remain[i]), int(i))
        )
    return rows


def _floor_lift(m: mujoco.MjModel, d: mujoco.MjData, tol: float = 0.002) -> float:
    """How far to raise the root so that no geom goes more than ``tol`` into
    the floor (0 when none does)."""
    mujoco.mj_forward(m, d)
    pen = 0.0
    for c in d.contact[: d.ncon]:
        if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0:
            pen = min(pen, float(c.dist))
    return max(0.0, -pen - tol)


def _self_penetration(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """Deepest robot-robot penetration [m] (0 when none)."""
    mujoco.mj_forward(m, d)
    pen = 0.0
    for c in d.contact[: d.ncon]:
        if m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0:
            pen = min(pen, float(c.dist))
    return -pen


def _yaw_free(quat: np.ndarray) -> np.ndarray:
    """quat with its heading removed: rotated about world z so that the
    torso's up axis (body z, horizontal when lying) points along world +x."""
    w, x, y, z = quat
    bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    yaw = math.atan2(bz[1], bz[0]) if np.hypot(bz[0], bz[1]) > 1e-3 else 0.0
    q = _quat_mul(_quat_from_axis_angle([0, 0, 1], -yaw), quat)
    return q / np.linalg.norm(q)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def face_of(quat: np.ndarray) -> str:
    """"prone" (chest down) or "supine" (back down), by the body x (forward)
    axis' world z; side-lying goes to the nearer one."""
    w, x, y, z = quat
    return "prone" if 2 * (x * z - w * y) < 0 else "supine"


def bridge_state(anchor_q: np.ndarray, fall_q: np.ndarray, alpha: float, m: mujoco.MjModel):
    """A lying state ``alpha`` of the way from a demo start (``anchor_q``, full
    qpos) to a real fall (``fall_q``): orientation slerped (headings removed),
    joints interpolated, dropped onto the floor, 1 s holding those joints
    (PD), then 1 s damping only like the controller, kept when at rest and
    still on the same face. Returns a library row or None."""
    d = mujoco.MjData(m)
    kp, kd, eff = _gains(m)
    lo, hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
    qa, qf = _yaw_free(anchor_q[3:7]), _yaw_free(fall_q[3:7])
    quat = _slerp(qa, qf, alpha)
    target = np.clip((1 - alpha) * anchor_q[7:] + alpha * fall_q[7:], lo, hi)
    for s in (0, 5):
        target[s + 3] = min(target[s + 3], rc.knee_max(target[s + 2]) - 0.02)
    d.qpos[3:7] = quat
    d.qpos[7:] = target
    d.qpos[2] = 0.5
    mujoco.mj_forward(m, d)
    low = min(
        d.geom_xpos[g, 2] - m.geom_rbound[g]
        for g in range(m.ngeom)
        if m.geom_contype[g] and m.geom_bodyid[g] > 0
    )
    d.qpos[2] += 0.005 - low
    face = face_of(quat)
    t_rest = 0.0
    for i in range(int(2.0 / DT)):
        q, qd = d.qpos[7:], d.qvel[6:]
        tau = kp * (target - q) - kd * qd if i < int(1.0 / DT) else -KD_DAMP * qd
        d.qfrc_applied[6:] = np.clip(tau, -eff, eff)
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return None
        if i > int(1.0 / DT) and np.abs(d.qvel).max() < REST_QVEL:
            t_rest += DT
        else:
            t_rest = 0.0
    if t_rest < 0.2 and np.abs(d.qvel).max() > SLOW_QVEL:
        return None
    if face_of(d.qpos[3:7]) != face:
        return None
    q = d.qpos[7:].copy()
    if not _knee_ok(q, 0.1):
        return None
    return float(d.qpos[2]), d.qpos[3:7].copy(), q, _tilt_deg(d), np.zeros(6), np.zeros(nj()), float(alpha)


def sample_bridge_states(demo_path: Path, falls: list, n: int, seed: int, m: mujoco.MjModel):
    """n "bridge_prone" / "bridge_supine" rows (n each at most): states between
    the demo starts (prone get-up, supine roll) and the real falls of that
    face, alpha ~ U(0, 1) (0 = the demo start, 1 = the real fall)."""
    rng = np.random.default_rng(seed)
    dd = np.load(demo_path)
    Q, demo = dd["qpos"].astype(np.float64), dd["demo"]
    # demo 0 starts prone, demo 1 supine (the arms demo has demo 1 only)
    anchors = {face: Q[np.flatnonzero(demo == k)[0]] for k, face in ((0, "prone"), (1, "supine")) if (demo == k).any()}
    by_face = {"prone": [], "supine": []}
    for z, quat, q in falls:
        by_face[face_of(quat)].append(np.r_[0.0, 0.0, z, quat, q])
    rows = []
    for face, pool in by_face.items():
        if face not in anchors:
            continue
        got, tries = 0, 0
        while got < n and tries < 3 * n and pool:
            tries += 1
            f = pool[rng.integers(len(pool))]
            r = bridge_state(anchors[face], f, float(rng.uniform(0, 1)), m)
            if r is not None:
                rows.append((f"bridge_{face}", r))
                got += 1
    return rows


def _work(args):
    seed, kind = args
    m = build_model()
    return kind, simulate(seed, kind, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", action="store_true", help="the model with arms (default out / demo: the _arms files)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-path", type=int, default=1500, help="states along the demo get-ups")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--n-bridge", type=int, default=1500, help="bridge states per face (prone / supine)")
    ap.add_argument(
        "--reuse-sim", default="", help="take the simulated (non-path) rows from this library instead of simulating"
    )
    a = ap.parse_args()
    global ARMS
    ARMS = a.arms
    a.out = a.out or str(rc.QDD_GETUP_ARMS_STATES_NPZ if ARMS else rc.QDD_GETUP_STATES_NPZ)
    a.demo = a.demo or str(rc.QDD_GETUP_ARMS_DEMO_NPZ if ARMS else rc.QDD_GETUP_DEMO_NPZ)
    if a.reuse_sim:
        old = np.load(a.reuse_sim)
        old_kinds = [str(k) for k in old["kinds"]]
        rows = []
        for j in range(len(old["root_z"])):
            k = old_kinds[old["kind"][j]]
            if k == "path" or k.startswith("bridge_"):
                continue
            rows.append((k, (float(old["root_z"][j]), old["root_quat"][j], old["joint_pos"][j],
                             float(old["tilt_deg"][j]), old["root_vel"][j], old["joint_vel"][j])))
        res = rows
    else:
        rng = np.random.default_rng(a.seed)
        jobs = [(a.seed * 100000 + i, SIM_KINDS[rng.choice(len(SIM_KINDS), p=KIND_P)]) for i in range(int(a.n * 1.3))]
        with mp.Pool(a.workers or None) as pool:
            res = pool.map(_work, jobs, chunksize=8)
        rows = [(k, r) for k, r in res if r is not None][: a.n]
    if a.n_path > 0:
        rows += [("path", r) for r in sample_path_states(Path(a.demo), a.n_path, a.seed + 7, build_model())]
    if a.n_bridge > 0:
        falls = [(r[0], r[1], r[2]) for k, r in rows if k in FALL_KINDS]
        rows += sample_bridge_states(Path(a.demo), falls, a.n_bridge, a.seed + 11, build_model())
    z = np.array([r[0] for _, r in rows], np.float32)
    quat = np.array([r[1] for _, r in rows], np.float32)
    q = np.array([r[2] for _, r in rows], np.float32)
    tilt = np.array([r[3] for _, r in rows], np.float32)
    kind = np.array([KINDS.index(k) for k, _ in rows], np.int8)
    root_vel = np.array([r[4] for _, r in rows], np.float32)
    joint_vel = np.array([r[5] for _, r in rows], np.float32)
    remain_s = np.array([r[6] if len(r) > 6 and k == "path" else -1.0 for k, r in rows], np.float32)
    alpha = np.array([r[6] if len(r) > 6 and k.startswith("bridge_") else -1.0 for k, r in rows], np.float32)
    demo_frame = np.array([r[7] if len(r) > 7 and k == "path" else -1 for k, r in rows], np.int32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, root_z=z, root_quat=quat, joint_pos=q, tilt_deg=tilt, kind=kind,
        root_vel=root_vel, joint_vel=joint_vel, remain_s=remain_s, alpha=alpha, demo_frame=demo_frame,
        kinds=np.array(KINDS), joint_names=np.array(joint_names()),
    )
    print(f"{len(rows)} states ({len(res) - sum(r is None for _, r in res)} usable of {len(res)}) -> {a.out}")
    for i, k in enumerate(KINDS):
        sel = kind == i
        if sel.any():
            print(f"  {k:6s} {sel.sum():5d}  tilt {np.percentile(tilt[sel], [10, 50, 90]).round(0)} deg  "
                  f"z {np.percentile(z[sel], [10, 50, 90]).round(3)} m")
    fall = np.isin(kind, [KINDS.index(k) for k in FALL_KINDS])
    print(f"  fall kinds still upright-ish (tilt < 30 deg): {(tilt[fall] < 30).mean():.1%}")


if __name__ == "__main__":
    main()
