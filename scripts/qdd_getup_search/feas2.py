"""CEM keyframe search for a get-up (asymmetric, 4 keyframes) from prone / supine.
--relax: no knee_max coupling, wider hip pitch (what-if hardware)."""
import sys, math, argparse, numpy as np, mujoco, multiprocessing as mp
sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas
from feas import OUT
g = feas.g; rc = feas.rc
NK = 4
RELAX = False
import os
GOAL = os.environ.get('GOAL', 'stand')
def lims(m):
    lo, hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
    if RELAX:
        for s in (0, 5): lo[s+2] = -2.6; hi[s+2] = 1.2; hi[s+3] = 2.6; lo[s+4] = -1.2; hi[s+4] = 1.2
    return lo, hi
def build():
    spec_m = g.build_model()
    if RELAX:
        lo, hi = lims(spec_m)
        spec_m.jnt_range[1:, 0] = lo; spec_m.jnt_range[1:, 1] = hi
    return spec_m
_M = None
def model():
    global _M
    if _M is None: _M = build()
    return _M
def clampq(q, m):
    lo, hi = lims(m); q = np.clip(q, lo, hi)
    if not RELAX:
        for s in (0, 5): q[s+3] = min(q[s+3], rc.knee_max(q[s+2]))
    return q
def rollout(args):
    x, qstart, record = args
    m = model(); d = mujoco.MjData(m); d.qpos[:] = qstart
    kp, kd, eff = g._gains(m); q0 = g._default_q()
    keys = [clampq(x[10*i:10*i+10].copy(), m) for i in range(NK)] + [q0]
    durs = np.clip(x[10*NK:10*NK+NK+1], 0.15, 1.5)
    T = durs.sum() + 1.5; n = int(T / g.DT)
    bounds = np.cumsum(durs); prev = d.qpos[7:].copy(); k = 0; t0 = 0.0
    best_phi = -9; frames = []
    for i in range(n):
        t = i * g.DT
        while k <= NK and t >= bounds[k]:
            prev = keys[k]; t0 = bounds[k]; k += 1
        tgt = q0 if k > NK else prev + min(1, (t - t0) / durs[k]) * (keys[k] - prev)
        d.qfrc_applied[6:] = np.clip(kp * (tgt - d.qpos[7:]) - kd * d.qvel[6:], -eff, eff)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): return -10.0, None
        if i % 10 == 0:
            up = d.xmat[1][8]; z = d.qpos[2]
            best_phi = max(best_phi, up + min(z / rc.STAND_BASE_Z, 1.0))
            if record and i % 20 == 0: frames.append(d.qpos.copy())
    up = d.xmat[1][8]; z = d.qpos[2]
    if GOAL == "prone":  # roll over onto the front: torso x axis pointing down, at rest
        fz = d.xmat[1][6]
        return -fz - 0.25 * np.linalg.norm(d.qvel[:6]), (frames if record else (round(fz, 3), round(z, 3)))
    final = up + 1 - abs(z - rc.STAND_BASE_Z) / 0.1 - 0.25 * np.linalg.norm(d.qvel[:6])
    return 0.3 * best_phi + final, (frames if record else (round(up, 3), round(z, 3)))
def worker_init(relax):
    global RELAX; RELAX = relax
def cem(kind, iters, pop, seed, relax):
    global RELAX; RELAX = relax
    rng = np.random.default_rng(seed); qs = feas.start_state(kind)
    q0 = g._default_q()
    mu = np.r_[np.tile(q0, NK), [0.5] * (NK + 1)]
    sd = np.r_[np.tile([0.15, 0.15, 0.8, 0.8, 0.5], 2 * NK), [0.3] * (NK + 1)]
    best = (-1e9, None, None)
    with mp.Pool(4, initializer=worker_init, initargs=(relax,)) as pool:
        for it in range(iters):
            X = mu + sd * rng.normal(size=(pop, mu.size))
            res = pool.map(rollout, [(x, qs, False) for x in X])
            S = np.array([r[0] for r in res]); el = np.argsort(S)[-pop // 8:]
            mu = 0.3 * mu + 0.7 * X[el].mean(0); sd = 0.3 * sd + 0.7 * X[el].std(0) + 0.01
            if S.max() > best[0]: best = (S.max(), X[S.argmax()], res[S.argmax()][1])
            if it % 5 == 0 or it == iters - 1:
                print(kind, "relax" if relax else "hw", it, round(S.max(), 3), "best up,z", best[2], flush=True)
    return best, qs
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("--relax", action="store_true")
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--pop", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (s, x, info), qs = cem(a.kind, a.iters, a.pop, a.seed, a.relax)
    tag = f"{a.kind}_{'relax' if a.relax else 'hw'}_{a.seed}"
    np.savez(OUT + f"/best2_{tag}.npz", x=x, qs=qs, score=s, relax=a.relax)
    print("BEST", tag, s, info)

def refine(path, iters, pop, seed, sd_scale=0.15):
    f = np.load(path); relax = bool(f["relax"])
    global RELAX; RELAX = relax
    rng = np.random.default_rng(seed); qs = f["qs"]; mu = f["x"].copy()
    sd = np.r_[np.full(10 * NK, sd_scale), [0.1] * (NK + 1)]
    best = (-1e9, None, None)
    with mp.Pool(4, initializer=worker_init, initargs=(relax,)) as pool:
        for it in range(iters):
            X = mu + sd * rng.normal(size=(pop, mu.size)); X[0] = mu
            res = pool.map(rollout, [(x, qs, False) for x in X])
            S = np.array([r[0] for r in res]); el = np.argsort(S)[-pop // 8:]
            mu = 0.3 * mu + 0.7 * X[el].mean(0); sd = 0.3 * sd + 0.7 * X[el].std(0) + 0.005
            if S.max() > best[0]: best = (S.max(), X[S.argmax()], res[S.argmax()][1])
            if it % 5 == 0 or it == iters - 1:
                print("refine", it, round(S.max(), 3), "best up,z", best[2], flush=True)
    out = path.replace(".npz", "_r.npz")
    np.savez(out, x=best[1], qs=qs, score=best[0], relax=relax)
    print("BEST", out, best[0], best[2])
