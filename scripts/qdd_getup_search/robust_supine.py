"""Robust CEM for supine falls: prep + 4 roll keys, then the fixed robust prone get-up."""
import os
os.environ["FACE"] = "supine"
import sys, numpy as np, mujoco, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robust
from robust import g, feas2, rc
PX = np.load(robust.OUT + "/robust_prone.npz")["x"]
SETTLE = 0.6
def seed():
    R = np.load(robust.OUT + "/roll_supine.npz")
    return np.r_[R["qs"][7:], R["x"][:40], R["x"][40:45]]
def roll(args):
    x, qstart, dr, rec = args
    m = feas2.model(); d = mujoco.MjData(m); d.qpos[:] = qstart
    fr0 = m.geom_friction[:, 0].copy(); m.geom_friction[:, 0] = fr0 * dr[0]
    kp, kd, eff = g._gains(m); kp = kp * dr[1]; DT = g.DT; q0 = g._default_q()
    rk = [feas2.clampq(x[10*i:10*i+10].copy(), m) for i in range(5)]
    rd = np.clip(x[50:55], 0.1, 1.5)
    pk, pd = robust.unpack(PX, m)
    frames = []
    def run(tgt_fn, n):
        for i in range(n):
            tgt = tgt_fn(i)
            d.qfrc_applied[6:] = np.clip(kp*(tgt-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all(): raise FloatingPointError
            if rec and i % 10 == 0: frames.append(d.qpos.copy())
    def seq(start, keys, durs, hold_last):
        b = np.cumsum(durs)
        def f(i):
            t = i * DT; k = int(np.searchsorted(b, t, side="right"))
            if k >= len(durs): return keys[-1]
            p = start if k == 0 else keys[k - 1]; t0 = 0 if k == 0 else b[k - 1]
            return p + min(1, (t - t0) / durs[k]) * (keys[k] - p)
        return f, int((b[-1] + hold_last) / DT)
    try:
        cur = d.qpos[7:].copy()
        f, n = seq(cur, [rk[0], rk[0]], [1.0, 0.5], 0); run(f, n)          # prep
        f, n = seq(rk[0], rk[1:] + [rk[4]], list(rd), SETTLE); run(f, n)    # roll (5th dur: hold)
        cur = d.qpos[7:].copy()
        f, n = seq(cur, [pk[0], pk[0]], [1.0, 0.5], 0); run(f, n)           # prone prep
        f, n = seq(pk[0], pk[1:], list(pd), 1.5); run(f, n)                 # prone get-up
    except FloatingPointError:
        return -10, None
    finally:
        m.geom_friction[:, 0] = fr0
    up = d.xmat[1][8]; z = d.qpos[2]
    s = up + 1 - min(abs(z - rc.STAND_BASE_Z) / 0.05, 3) - 0.5*np.linalg.norm(d.qvel[:6]) - 0.3*np.abs(d.qpos[7:] - q0).sum()/10
    ok = up > 0.9 and abs(z - rc.STAND_BASE_Z) < 0.03
    return s, (frames if rec else ok)
if __name__ == "__main__":
    NS = 8; IT = int(os.environ.get("IT", "100")); POP = 96
    rng = np.random.default_rng(2); P = robust.starts(64)
    mu = seed(); sd = np.r_[[0.15]*50, [0.05]*5]; best = (-1, None)
    with mp.Pool(4) as pool:
        r = pool.map(roll, [(mu, s, (1.0, 1.0), False) for s in P]); print("seed success", np.mean([q[1] for q in r]), flush=True)
        for it in range(IT):
            X = mu + sd * rng.normal(size=(POP, mu.size)); X[0] = mu
            ids = rng.choice(len(P), NS, replace=False); drs = [(rng.uniform(0.6, 1.2), rng.uniform(0.9, 1.1)) for _ in ids]
            res = pool.map(roll, [(x, P[j], drs[a], False) for x in X for a, j in enumerate(ids)], chunksize=8)
            sc = np.array([q[0] for q in res]).reshape(POP, NS); okr = np.array([bool(q[1]) for q in res]).reshape(POP, NS)
            S = sc.mean(1) + okr.mean(1); el = np.argsort(S)[-12:]
            mu = 0.3*mu + 0.7*X[el].mean(0); sd = 0.5*sd + 0.5*X[el].std(0) + 0.005
            if it % 10 == 9 or it == IT - 1:
                r = pool.map(roll, [(mu, s, (rng.uniform(0.6, 1.2), rng.uniform(0.9, 1.1)), False) for s in P])
                sr = np.mean([q[1] for q in r]); print(it, "elite", round(S[el[-1]], 3), "mu success(64)", sr, flush=True)
                if sr >= best[0]: best = (sr, mu.copy()); np.savez(robust.OUT + "/robust_supine.npz", x=mu, success=sr)
            else: print(it, "elite", round(S[el[-1]], 3), "ok", okr[el].mean().round(2), flush=True)
    print("BEST", best[0])
