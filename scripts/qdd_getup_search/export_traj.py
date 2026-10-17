"""Replay the found get-up (prone: flip 1.3 s + lunge->stand) and save qpos/qvel every 20 ms."""
import sys, numpy as np, mujoco
sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas2
from feas import OUT
g = feas2.g; m = feas2.model(); q0 = g._default_q()
def sim(qpos, qvel, keys, durs, prev, T):
    d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel; kp, kd, eff = g._gains(m)
    bounds = np.cumsum(durs); k = 0; t0 = 0; Q = []; V = []
    for i in range(int(round(T / g.DT))):
        t = i * g.DT
        while k < len(keys) and t >= bounds[k]: prev = keys[k]; t0 = bounds[k]; k += 1
        tgt = q0 if k >= len(keys) else prev + min(1, (t - t0) / durs[k]) * (keys[k] - prev)
        d.qfrc_applied[6:] = np.clip(kp*(tgt-d.qpos[7:]) - kd*d.qvel[6:], -eff, eff); mujoco.mj_step(m, d)
        if i % 20 == 19: Q.append(d.qpos.copy()); V.append(d.qvel.copy())
    return Q, V
f = np.load(OUT + "/best2_prone_hw_0.npz"); L = np.load(OUT + "/lunge_best.npz")
x = f["x"]; keys = [feas2.clampq(x[10*i:10*i+10].copy(), m) for i in range(4)] + [q0]
Q1, V1 = sim(f["qs"], np.zeros(m.nv), keys, np.clip(x[40:45], .15, 1.5), f["qs"][7:].copy(), 1.3)
x2 = L["x"]; keys2 = [feas2.clampq(x2[10*i:10*i+10].copy(), m) for i in range(3)] + [q0]
d2 = np.clip(x2[30:34], .1, 1.5)
Q2, V2 = sim(L["qpos"], L["qvel"], keys2, d2, L["tgt0"], d2.sum() + 1.0)
demos = [(Q1 + Q2, V1 + V2, "prone_flip_lunge")]
# back -> front roll (feas2.py supine with GOAL=prone), as optimised (ends holding the default pose)
import os
if os.path.exists(OUT + "/roll_supine.npz"):
    Rr = np.load(OUT + "/roll_supine.npz"); xr = Rr["x"]
    keys_r = [feas2.clampq(xr[10*i:10*i+10].copy(), m) for i in range(4)] + [q0]
    dr = np.clip(xr[40:45], .15, 1.5)
    Qr, Vr = sim(Rr["qs"], np.zeros(m.nv), keys_r, dr, Rr["qs"][7:].copy(), dr.sum() + 1.5)
    demos.append((Qr, Vr, "supine_roll_to_front"))
# rolled-over front -> standing (ref.py on the rolled end state), if found
if os.path.exists(OUT + "/rolled_start_r.npz"):
    Rs = np.load(OUT + "/rolled_start_r.npz")
    if Rs["score"] > 1.0:
        xs = Rs["x"]; keys_s = [feas2.clampq(xs[10*i:10*i+10].copy(), m) for i in range(4)] + [q0]
        ds = np.clip(xs[40:45], .15, 1.5)
        Qs, Vs = sim(Rs["qs"], np.zeros(m.nv), keys_s, ds, Rs["qs"][7:].copy(), ds.sum() + 1.0)
        demos.append((Qs, Vs, "rolled_front_to_stand"))
Q = np.concatenate([np.array(q) for q, _, _ in demos]); V = np.concatenate([np.array(v) for _, v, _ in demos])
demo = np.concatenate([np.full(len(q), i, np.int8) for i, (q, _, _) in enumerate(demos)])
for i, (q, _, n) in enumerate(demos):
    print(i, n, len(q), "end z %.3f" % q[-1][2])
np.savez(sys.argv[1], qpos=Q.astype(np.float32), qvel=V.astype(np.float32), demo=demo,
         names=np.array([n for _, _, n in demos]))
