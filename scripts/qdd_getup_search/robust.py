"""Robust CEM: one keyframe get-up that stands from many real prone falls.

The motion: ramp the joints from wherever the fall left them to a prep pose K0
(1.0 s), hold it (0.5 s), then K1..K7 -> default pose. The score is averaged
over NS library falls per candidate with friction (x0.6-1.2) and kp (x0.9-1.1)
randomised, plus a bonus for the fraction that ends standing. Seeded from the
found prone get-up (best2_prone_hw_0 + lunge_best), which stood only from its
own start pose. Result: OUT/robust_prone.npz. Supine: robust_supine.py."""
import sys, os, numpy as np, mujoco, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mjlab_microduck  # noqa
import feas2
from feas import OUT
from mjlab_microduck.robot import qdd_biped_constants as rc
g = feas2.g
NKEY = 8; PREP, HOLD = 1.0, 0.5
FACE = os.environ.get("FACE", "prone")
def seed():
    f = np.load(OUT + "/best2_prone_hw_0.npz"); L = np.load(OUT + "/lunge_best.npz")
    x, x2 = f["x"], L["x"]; q0 = g._default_q()
    keys = [f["qs"][7:]] + [x[10*i:10*i+10] for i in range(3)] + [L["tgt0"]] + [x2[10*i:10*i+10] for i in range(3)] + [q0]
    durs = [0.152, 0.15, 0.975, 0.023, 0.205, 0.158, 0.462, 0.29]
    return np.r_[np.concatenate(keys[:8]), durs]   # K0 prep target, K1..K7, durs for K1..K7,q0
def unpack(x, m):
    keys = [feas2.clampq(x[10*i:10*i+10].copy(), m) for i in range(NKEY)] + [g._default_q()]
    durs = np.clip(x[10*NKEY:10*NKEY+8], 0.02, 1.5)
    return keys, durs
def starts(n, seed_=0):
    lib = np.load(str(rc.QDD_GETUP_STATES_NPZ)); kinds = list(lib["kinds"])
    w, x, y, z = lib["root_quat"].T; prone = 2*(x*z - w*y) < 0
    sel = np.isin(lib["kind"], [kinds.index(k) for k in ("push", "stride", "drop", "bridge_prone" if FACE=="prone" else "bridge_supine")])
    sel &= prone if FACE == "prone" else ~prone
    if "alpha" in lib.files: sel &= ~((lib["kind"] == kinds.index("bridge_" + FACE)) & (lib["alpha"] < 0.5))
    ids = np.random.default_rng(seed_).choice(np.flatnonzero(sel), n, replace=False)
    out = []
    for i in ids:
        q = np.zeros(17); q[2] = lib["root_z"][i] + 0.003; q[3:7] = lib["root_quat"][i]; q[7:] = lib["joint_pos"][i]
        out.append(q)
    return out
def roll(args):
    x, qstart, dr, rec = args
    m = feas2.model(); d = mujoco.MjData(m); d.qpos[:] = qstart
    fr0 = m.geom_friction[:, 0].copy(); m.geom_friction[:, 0] = fr0 * dr[0]
    kp, kd, eff = g._gains(m); kp = kp * dr[1]
    keys, durs = unpack(x, m); q0 = g._default_q()
    cur = d.qpos[7:].copy(); n0 = int(PREP / g.DT); nh = int(HOLD / g.DT)
    bounds = np.cumsum(durs); n = n0 + nh + int((durs.sum() + 1.5) / g.DT)
    prev = keys[0]; k = 0; t0 = 0.0; frames = []; best = -9
    try:
        for i in range(n):
            if i < n0: tgt = cur + (i + 1) / n0 * (keys[0] - cur)
            elif i < n0 + nh: tgt = keys[0]
            else:
                t = (i - n0 - nh) * g.DT
                while k < 8 and t >= bounds[k]: prev = keys[k + 1]; t0 = bounds[k]; k += 1
                tgt = q0 if k >= 8 else prev + min(1, (t - t0) / durs[k]) * (keys[k + 1] - prev)
            d.qfrc_applied[6:] = np.clip(kp*(tgt-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all(): return -10, None
            if rec and i % 10 == 0: frames.append(d.qpos.copy())
    finally:
        m.geom_friction[:, 0] = fr0
    up = d.xmat[1][8]; z = d.qpos[2]
    s = up + 1 - min(abs(z - rc.STAND_BASE_Z) / 0.05, 3) - 0.5*np.linalg.norm(d.qvel[:6]) - 0.3*np.abs(d.qpos[7:] - q0).sum()/10
    ok = up > 0.9 and abs(z - rc.STAND_BASE_Z) < 0.03
    return s, (frames if rec else ok)
if __name__ == "__main__":
    NS = int(os.environ.get("NS", "8")); IT = int(os.environ.get("IT", "80")); POP = 96
    rng = np.random.default_rng(1); pool_starts = starts(64)
    mu = seed(); sd = np.r_[[0.15]*80, [0.05]*8]
    best = (-1e9, None)
    with mp.Pool(4) as pool:
        # check the seed
        X0 = [(mu, s, (1.0, 1.0), False) for s in pool_starts]
        r = pool.map(roll, X0); print("seed success", np.mean([q[1] for q in r]), "mean", np.mean([q[0] for q in r]), flush=True)
        for it in range(IT):
            X = mu + sd * rng.normal(size=(POP, mu.size)); X[0] = mu
            S_ids = rng.choice(len(pool_starts), NS, replace=False)
            drs = [(rng.uniform(0.6, 1.2), rng.uniform(0.9, 1.1)) for _ in range(NS)]
            jobs = [(x, pool_starts[j], drs[a], False) for x in X for a, j in enumerate(S_ids)]
            res = pool.map(roll, jobs, chunksize=8)
            sc = np.array([q[0] for q in res]).reshape(POP, NS); okr = np.array([bool(q[1]) for q in res]).reshape(POP, NS)
            S = sc.mean(1) + 1.0 * okr.mean(1); el = np.argsort(S)[-12:]
            mu = 0.3*mu + 0.7*X[el].mean(0); sd = 0.5*sd + 0.5*X[el].std(0) + 0.005
            if it % 10 == 9 or it == IT - 1:
                r = pool.map(roll, [(mu, s, (rng.uniform(0.6, 1.2), rng.uniform(0.9, 1.1)), False) for s in pool_starts])
                sr = np.mean([q[1] for q in r]); print(it, "elite", round(S[el[-1]], 3), "mu success(64)", sr, flush=True)
                if sr >= best[0]: best = (sr, mu.copy()); np.savez(f"{OUT}/robust_{FACE}.npz", x=mu, success=sr)
            else:
                print(it, "elite", round(S[el[-1]], 3), "ok", okr[el].mean().round(2), flush=True)
    print("BEST", best[0])
