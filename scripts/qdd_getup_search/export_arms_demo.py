"""Export a found arms get-up (gpu_cem.py result) as the get-up task's demo with arms.

Replays the motion in plain MuJoCo (search model, torque-speed limit on) from
supine falls of the library and keeps the first that stands; writes qpos / qvel
every 20 ms in the format of getup_demo_traj.npz, plus joint_names (legs + arms).
demo = 1 (supine; the arms demo has no prone get-up, demo 0).

  uv run python scripts/qdd_getup_search/export_arms_demo.py \
      logs/getup_arms_v5b/supine_cad_from20_sd010.npz \
      --out src/mjlab_microduck/robot/qdd_biped/getup_demo_traj_arms.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arms_replay as R
import gpu_cem as G
import mujoco
import numpy as np
import torch

from mjlab_microduck.robot import qdd_biped_constants as rc

FRAME_DT = 0.02


def record(S, x, q_start):
    keys, durs = S.keys(torch.tensor(x[None], dtype=torch.float32))
    keys, durs = keys[0].numpy(), durs[0].numpy()
    m, a = S.m, S.a
    d = mujoco.MjData(m)
    d.qpos[:] = q_start
    mujoco.mj_forward(m, d)
    kp, kd, eff = S.kp.numpy(), S.kd.numpy(), S.eff.numpy()
    vmax = np.array(G.vmax_list(a.elbow, getattr(a, "cad", False)))
    allk = np.r_[keys, S.q0.numpy()[None]]
    segs = [(d.qpos[7:].copy(), allk[0], a.prep)] + [(allk[j], allk[j + 1], durs[j]) for j in range(a.nk)]
    every = round(FRAME_DT / G.DT)
    Q, V, step = [d.qpos.copy()], [d.qvel.copy()], 0
    for p, q, du in segs + [(allk[-1], allk[-1], a.hold)]:
        n = round(du / G.DT)
        for i in range(n):
            tgt = p + min(1, (i + 1) / n) * (q - p)
            tau = kp * (tgt - d.qpos[7:]) - kd * d.qvel[6:]
            lo, hi = G.torque_limits(d.qvel[6:], eff, vmax)
            d.qfrc_applied[6:] = np.clip(tau, lo, hi)
            mujoco.mj_step(m, d)
            step += 1
            if step % every == 0:
                Q.append(d.qpos.copy())
                V.append(d.qvel.copy())
    ok = d.xmat[1][8] > 0.9 and abs(d.qpos[2] - rc.STAND_BASE_Z) < 0.03
    return ok, np.array(Q), np.array(V), allk, np.asarray(durs, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--which", default="best", choices=("best", "mu"))
    ap.add_argument("--out", type=Path, default=rc.QDD_GETUP_ARMS_DEMO_NPZ)
    ap.add_argument("--arm-len", type=float, default=None, help="replay on another arm length (default: the search's)")
    ap.add_argument("--starts", type=int, default=32)
    r = ap.parse_args()
    f = np.load(r.result)
    a = argparse.Namespace(**json.loads(f["args"].item()))
    if r.arm_len is not None:
        a.arm_len = r.arm_len
    assert a.face == "supine" and a.elbow, "the arms demo is the supine get-up with elbows"
    S = R.cpu_search(a)
    na = S.nj - 10
    st = G.fall_starts(a.face, r.starts, 1234)
    st = np.c_[st, np.zeros((len(st), na))]
    for k, s in enumerate(st):
        ok, Q, V, allk, durs = record(S, f[r.which], s)
        if ok:
            break
    else:
        raise SystemExit("no start stood")
    # trim the standing hold to 0.5 s after the motion
    keep = len(Q) - round((a.hold - 0.5) / FRAME_DT)
    Q, V = Q[:keep], V[:keep]
    names = [S.m.joint(i).name for i in range(1, S.m.njnt)]
    jn = [{"l_shoulder": "l_shoulder_pitch", "r_shoulder": "r_shoulder_pitch"}.get(n, n) for n in names]
    assert tuple(jn) == rc.QDD_ARMS_JOINT_NAMES, jn
    np.savez_compressed(
        r.out,
        qpos=Q.astype(np.float32),
        qvel=V.astype(np.float32),
        demo=np.ones(len(Q), np.int8),
        names=np.array(["", f"arms_supine:{r.result.name}"]),
        joint_names=np.array(jn),
        # the open-loop joint targets (Mjlab-Getup-Supine-QddBipedArms plays them as the
        # feed-forward of a residual policy): from the start pose to ref_keys[0] in
        # ref_prep_s, then linearly key to key in ref_durs_s, then ref_keys[-1] (standing)
        ref_keys=allk.astype(np.float32),
        ref_durs_s=durs.astype(np.float32),
        ref_prep_s=np.float32(a.prep),
    )
    print(f"start {k}: {len(Q)} frames ({len(Q) * FRAME_DT:.1f} s) -> {r.out}")


if __name__ == "__main__":
    main()
