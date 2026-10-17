"""Save the robust get-ups (robust.py / robust_supine.py) as the demo trajectories
(qpos/qvel every 20 ms) that the get-up task's path states and demo tracking use.

demo 0 (prone): a real prone fall after the 1.0 s prep ramp, from the 0.5 s
prep hold through K1..K7 to the default pose (+1.0 s). demo 1 (supine): a real
supine fall from its prep hold through the roll and the prone prep, ending lying
prone in the prep pose K0, i.e. at the start of demo 0.
    python export_robust.py src/mjlab_microduck/robot/qdd_biped/getup_demo_traj.npz"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FACE"] = "prone"
import robust
from robust import g, feas2, OUT
m = feas2.model(); q0 = g._default_q(); kp, kd, eff = g._gains(m); DT = g.DT
import json
# the search results: OUT/robust_*.npz if present, else the copy in robust_keyframes.json
_kf = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robust_keyframes.json")))["raw"]
_px = np.load(OUT + "/robust_prone.npz")["x"] if os.path.exists(OUT + "/robust_prone.npz") else np.array(_kf["prone_x"])
xs = np.load(OUT + "/robust_supine.npz")["x"] if os.path.exists(OUT + "/robust_supine.npz") else np.array(_kf["supine_x"])
pk, pd = robust.unpack(_px, m)
rk = [feas2.clampq(xs[10*i:10*i+10].copy(), m) for i in range(5)]; rd = np.clip(xs[50:55], 0.1, 1.5)
def seq(start, keys, durs, hold):
    b = np.cumsum(durs)
    def f(i):
        t = i * DT; k = int(np.searchsorted(b, t, side="right"))
        if k >= len(durs): return keys[-1]
        p = start if k == 0 else keys[k - 1]; t0 = 0 if k == 0 else b[k - 1]
        return p + min(1, (t - t0) / durs[k]) * (keys[k] - p)
    return f, int(round((b[-1] + hold) / DT))
def run(d, segs, rec_from):
    Q, V, step = [], [], 0
    for si, (start, keys, durs, hold) in enumerate(segs):
        f, n = seq(d.qpos[7:].copy() if start is None else start, keys, durs, hold)
        for i in range(n):
            d.qfrc_applied[6:] = np.clip(kp*(f(i)-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
            if si >= rec_from:
                step += 1
                if step % int(round(0.02 / DT)) == 0: Q.append(d.qpos.copy()); V.append(d.qvel.copy())
    return Q, V
PREP = lambda K: (None, [K, K], [robust.PREP, 1e-3], 0)     # ramp only
HOLD = lambda K: (K, [K], [robust.HOLD], 0)
def start(face, seed):
    os.environ["FACE"] = face; robust.FACE = face
    for s in robust.starts(64, seed_=seed):
        d = mujoco.MjData(m); d.qpos[:] = s; yield d
def ok(d):
    return d.xmat[1][8] > 0.9 and abs(d.qpos[2] - robust.rc.STAND_BASE_Z) < 0.03
PRONE = [HOLD(pk[0]), (pk[0], pk[1:], list(pd), 1.0)]
for d in start("prone", 11):
    Q0, V0 = run(d, [PREP(pk[0])] + PRONE, 1)
    if ok(d): break
SUP = [HOLD(rk[0]), (rk[0], rk[1:] + [rk[4]], list(rd), 0.6), PREP(pk[0]), HOLD(pk[0])]
for d in start("supine", 12):
    Q1, V1 = run(d, [PREP(rk[0])] + SUP, 1)
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos; d2.qvel[:] = d.qvel; run(d2, PRONE, 9)
    if ok(d2): break
Q = np.array(Q0 + Q1); V = np.array(V0 + V1)
demo = np.r_[np.zeros(len(Q0), np.int8), np.ones(len(Q1), np.int8)]
print("prone", len(Q0), "frames, end z %.3f" % Q0[-1][2], "| supine", len(Q1), "frames, end z %.3f" % Q1[-1][2])
np.savez(sys.argv[1], qpos=Q.astype(np.float32), qvel=V.astype(np.float32), demo=demo,
         names=np.array(["robust_prone_getup", "robust_supine_roll"]))
