"""Feasibility: can a symmetric keyframe PD motion get mochi up from prone / supine?
CEM over 3 keyframes (5 joint targets each, mirrored legs) + timings."""
import os
OUT = os.environ.get("GETUP_SEARCH_OUT", "/tmp/qdd_getup_search")
os.makedirs(OUT, exist_ok=True)
import sys, math, numpy as np, mujoco, importlib.util, multiprocessing as mp
spec = importlib.util.spec_from_file_location("g", __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "qdd_getup_states.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
from mjlab_microduck.robot import qdd_biped_constants as rc

M = None
def model():
    global M
    if M is None: M = g.build_model()
    return M

def start_state(kind):
    m = model(); d = mujoco.MjData(m)
    q0 = g._default_q()
    if kind == "prone":   # torso x axis down: rotate +90 deg about y
        quat = g._quat_from_axis_angle([0, 1, 0], math.pi / 2)
    else:                 # supine: torso x axis up
        quat = g._quat_from_axis_angle([0, 1, 0], -math.pi / 2)
    d.qpos[:3] = [0, 0, 0.12]; d.qpos[3:7] = quat; d.qpos[7:] = q0
    kp, kd, eff = g._gains(m)
    for i in range(1500):  # settle in damping
        d.qfrc_applied[6:] = np.clip(-0.5 * d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
    return d.qpos.copy()

LO = np.array([-0.1, -0.1, -1.74, 0.0, -0.87]); HI = np.array([0.1, 0.1, 0.69, 2.09, 0.87])

def legs(p5):
    p5 = np.clip(p5, LO, HI).copy()
    p5[3] = min(p5[3], rc.knee_max(p5[2]))
    l = p5.copy(); r = p5.copy(); r[0] *= -1; r[1] *= -1
    return np.r_[l, r]

def rollout(args):
    x, qstart, record = args
    m = model(); d = mujoco.MjData(m); d.qpos[:] = qstart
    kp, kd, eff = g._gains(m); q0 = g._default_q()
    keys = [legs(x[5*i:5*i+5]) for i in range(3)] + [q0]
    durs = np.clip(x[15:19], 0.2, 1.5)
    t_end = durs.sum() + 1.5
    frames = []; T = 0.0; k = 0; tk = 0.0; prev = d.qpos[7:].copy()
    steps = int(t_end / g.DT); score_acc = 0.0
    for i in range(steps):
        t = i * g.DT
        while k < 4 and t - tk > durs[k] if k < 4 else False:
            tk += durs[k]; prev = keys[k]; k += 1
        if k < 4:
            a = min(1.0, (t - tk) / durs[k]); tgt = prev + a * (keys[k] - prev)
        else:
            tgt = q0
        d.qfrc_applied[6:] = np.clip(kp * (tgt - d.qpos[7:]) - kd * d.qvel[6:], -eff, eff)
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)): return -10.0, None
        if record and i % 20 == 0: frames.append(d.qpos.copy())
    R = d.xmat[1].reshape(3, 3); up = R[2, 2]; z = d.qpos[2]
    score = up + 1.0 - abs(z - rc.STAND_BASE_Z) / 0.1 - 0.5 * np.linalg.norm(d.qvel[:6]) / 2
    # knee_max violations along the way are not checked here (targets respect it)
    return score, (frames if record else (up, z))

def cem(kind, iters=25, pop=64, seed=0):
    rng = np.random.default_rng(seed); qs = start_state(kind)
    mu = np.r_[np.tile([0, 0, -1.0, 1.8, -0.6], 3), [0.6, 0.6, 0.6, 0.6]]
    sd = np.r_[np.tile([0.05, 0.05, 0.6, 0.5, 0.4], 3), [0.3] * 4]
    best = (-1e9, None)
    with mp.Pool(4) as pool:
        for it in range(iters):
            X = mu + sd * rng.normal(size=(pop, mu.size))
            res = pool.map(rollout, [(x, qs, False) for x in X])
            S = np.array([r[0] for r in res]); el = np.argsort(S)[-8:]
            mu = X[el].mean(0); sd = X[el].std(0) + 0.02
            if S.max() > best[0]: best = (S.max(), X[S.argmax()], res[S.argmax()][1])
            print(kind, it, round(S.max(), 3), "up,z", np.round(best[2], 3), flush=True)
    return best, qs

if __name__ == "__main__":
    kind = sys.argv[1]
    (s, x, info), qs = cem(kind)
    np.savez(OUT + f"/best_{kind}.npz", x=x, qs=qs, score=s)
    print("BEST", kind, s, info)
