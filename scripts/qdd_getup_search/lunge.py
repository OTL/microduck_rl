"""From the upright split/lunge reached by the prone search, find a keyframe motion to standing."""
import sys, numpy as np, mujoco, multiprocessing as mp
sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas2
from feas import OUT
g = feas2.g; rc = feas2.rc
def capture(path, t_cap):
    f = np.load(path); x = f["x"]; m = feas2.model(); d = mujoco.MjData(m); d.qpos[:] = f["qs"]
    kp, kd, eff = g._gains(m); NK = feas2.NK
    keys = [feas2.clampq(x[10*i:10*i+10].copy(), m) for i in range(NK)] + [g._default_q()]
    durs = np.clip(x[10*NK:10*NK+NK+1], 0.15, 1.5); bounds = np.cumsum(durs)
    prev = d.qpos[7:].copy(); k = 0; t0 = 0.0
    for i in range(int(t_cap / g.DT)):
        t = i * g.DT
        while k <= NK and t >= bounds[k]: prev = keys[k]; t0 = bounds[k]; k += 1
        tgt = prev + min(1, (t - t0) / durs[k]) * (keys[k] - prev)
        d.qfrc_applied[6:] = np.clip(kp*(tgt-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy(), tgt.copy()
START = None
def roll(args):
    x, rec = args
    qpos, qvel, tgt0 = START; m = feas2.model(); d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel
    kp, kd, eff = g._gains(m); q0 = g._default_q()
    keys = [feas2.clampq(x[10*i:10*i+10].copy(), m) for i in range(3)] + [q0]
    durs = np.clip(x[30:34], 0.1, 1.5); bounds = np.cumsum(durs); prev = tgt0; k = 0; t0 = 0.0
    n = int((durs.sum() + 2.0) / g.DT); frames = []; fell = False
    for i in range(n):
        t = i * g.DT
        while k <= 3 and t >= bounds[k]: prev = keys[k]; t0 = bounds[k]; k += 1
        tgt = q0 if k > 3 else prev + min(1, (t - t0) / durs[k]) * (keys[k] - prev)
        d.qfrc_applied[6:] = np.clip(kp*(tgt-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): return -10, None
        if rec and i % 20 == 0: frames.append(d.qpos.copy())
    up = d.xmat[1][8]; z = d.qpos[2]
    s = up + 1 - abs(z - rc.STAND_BASE_Z) / 0.05 - 0.5 * np.linalg.norm(d.qvel[:6]) - 0.3 * np.abs(d.qpos[7:] - q0).sum() / 10
    return s, (frames if rec else (round(up, 3), round(z, 3)))
def init(st):
    global START; START = st
if __name__ == "__main__":
    path, t_cap = sys.argv[1], float(sys.argv[2])
    st = capture(path, t_cap); init(st)
    rng = np.random.default_rng(0)
    mu = np.r_[np.tile(st[2], 3), [0.4] * 4]; sd = np.r_[[0.4] * 30, [0.2] * 4]
    best = (-1e9, None, None)
    with mp.Pool(4, initializer=init, initargs=(st,)) as pool:
        for it in range(60):
            X = mu + sd * rng.normal(size=(128, mu.size))
            res = pool.map(roll, [(x, False) for x in X]); S = np.array([r[0] for r in res]); el = np.argsort(S)[-16:]
            mu = 0.3*mu + 0.7*X[el].mean(0); sd = 0.3*sd + 0.7*X[el].std(0) + 0.01
            if S.max() > best[0]: best = (S.max(), X[S.argmax()], res[S.argmax()][1])
            if it % 5 == 0: print("lunge", it, round(S.max(), 3), best[2], flush=True)
    np.savez(OUT + "/lunge_best.npz", x=best[1], qpos=st[0], qvel=st[1], tgt0=st[2], score=best[0])
    print("BEST", best[0], best[2])
