"""Replay a gpu_cem.py result in plain MuJoCo (CPU): success count over falls, optional GIF.

uv run python scripts/qdd_getup_search/arms_replay.py logs/getup_arms/prone_L20.npz [--gif out.gif]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gpu_cem as G
import mujoco
import numpy as np
import torch


def cpu_search(a) -> "G.Search":
    """A CPU Search object (model, gains, key builder) for a saved result's args."""
    S = G.Search.__new__(G.Search)
    S.a, S.dev = a, "cpu"
    S.arm = G.arm_cfg(a)
    S.m = G.build_model(S.arm)
    S.nj = S.m.nq - 7
    S.npk = G.n_per_key(a)
    kp, kd, eff = G.leg_gains()
    na = S.nj - 10
    S.kp = torch.tensor(kp + [S.arm.kp] * na)
    S.kd = torch.tensor(kd + [S.arm.kd] * na)
    S.eff = torch.tensor(eff + [S.arm.effort] * na)
    S.lo = torch.tensor(S.m.jnt_range[1:, 0], dtype=torch.float32)
    S.hi = torch.tensor(S.m.jnt_range[1:, 1], dtype=torch.float32)
    S.q0 = torch.tensor(np.r_[G.default_legs(), np.zeros(na)], dtype=torch.float32)
    return S


def rollout(m, S, x, q0_state, rec=False, speed_limit=True):
    keys, durs = S.keys(torch.tensor(x[None], dtype=torch.float32))
    keys, durs = keys[0].numpy(), durs[0].numpy()
    d = mujoco.MjData(m)
    d.qpos[:] = q0_state
    mujoco.mj_forward(m, d)
    kp, kd, eff = S.kp.numpy(), S.kd.numpy(), S.eff.numpy()
    allk = np.r_[keys, S.q0.numpy()[None]]
    cur = d.qpos[7:].copy()
    a = S.a
    segs = [(cur, allk[0], a.prep)] + [
        (allk[j], allk[j + 1], durs[j]) for j in range(a.nk)
    ]
    frames = []
    vmax = np.array(G.vmax_list(a.elbow, getattr(a, "cad", False)))
    peak = np.zeros_like(vmax)
    spin = 0.0
    for p, q, du in segs + [(allk[-1], allk[-1], a.hold)]:
        n = round(du / G.DT)
        for i in range(n):
            tgt = p + min(1, (i + 1) / n) * (q - p)
            tau = kp * (tgt - d.qpos[7:]) - kd * d.qvel[6:]
            if speed_limit:
                lo, hi = G.torque_limits(d.qvel[6:], eff, vmax)
            else:
                lo, hi = -eff, eff
            d.qfrc_applied[6:] = np.clip(tau, lo, hi)
            mujoco.mj_step(m, d)
            peak = np.maximum(peak, np.abs(d.qvel[6:]) / vmax)
            spin = max(spin, float(np.linalg.norm(d.qvel[3:6])))
            if rec and i % 10 == 0:
                frames.append(d.qpos.copy())
    up, z = d.xmat[1][8], d.qpos[2]
    ok = up > 0.9 and abs(z - G.rc.STAND_BASE_Z) < 0.03
    return ok, (frames if rec else np.r_[peak, spin])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--which", default="best", choices=("best", "mu"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--speed-limit", choices=("on", "off"), default="on")
    r = ap.parse_args()
    f = np.load(r.result)
    args = json.loads(f["args"].item())
    a = argparse.Namespace(**args)
    a.pop, a.ns = 1, 1
    S = cpu_search(a)
    na = S.nj - 10
    x = f[r.which]
    st = G.fall_starts(a.face, r.n, 1234)
    st = np.c_[st, np.zeros((len(st), na))]
    sl = r.speed_limit == "on"
    res = [rollout(S.m, S, x, s, speed_limit=sl) for s in st]
    oks = [o for o, _ in res]
    peak = np.max([p for _, p in res], 0)
    print(
        "peak |qd| / no-load speed per joint:",
        np.round(peak[:-1], 2),
        "torso spin max",
        round(peak[-1], 1),
        "rad/s",
    )
    print(
        f"{r.which}: stood {sum(oks)} / {len(oks)} ({a.face}, arm {a.arm_len} m, elbow {a.elbow})"
    )
    if r.gif:
        from PIL import Image, ImageDraw

        idx = [i for i, o in enumerate(oks) if o][:2] + [
            i for i, o in enumerate(oks) if not o
        ][:1]
        idx = (idx + [0, 1, 2])[:3]
        F = [rollout(S.m, S, x, st[i], rec=True, speed_limit=sl)[1] for i in idx]
        T = min(len(q) for q in F)
        m = S.m
        d = mujoco.MjData(m)
        w, h = 320, 260
        m.vis.headlight.ambient[:] = [0.6] * 3
        rr = mujoco.Renderer(m, h, w)
        cam = mujoco.MjvCamera()
        cam.distance = 1.2
        cam.elevation = -10
        cam.azimuth = 90
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
        frames = []
        for t in range(0, T, 2):
            canvas = np.zeros((h, w * len(idx), 3), np.uint8)
            for j in range(len(idx)):
                d.qpos[:] = F[j][t]
                mujoco.mj_forward(m, d)
                cam.lookat[:] = [F[j][t][0], F[j][t][1], 0.18]
                rr.update_scene(d, cam, opt)
                canvas[:, j * w : (j + 1) * w] = rr.render()
            im = Image.fromarray(canvas)
            dr = ImageDraw.Draw(im)
            dr.text((6, 6), f"{t * 10 * G.DT:.1f} s", fill=(255, 255, 0))
            for j, i in enumerate(idx):
                dr.text(
                    (j * w + 6, 20), "stood" if oks[i] else "fell", fill=(255, 255, 0)
                )
            frames.append(im)
        frames[0].save(
            r.gif, save_all=True, append_images=frames[1:], duration=40, loop=0
        )
        print("gif", r.gif, len(frames))


if __name__ == "__main__":
    main()
