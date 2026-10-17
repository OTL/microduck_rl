import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas2, render
f = np.load(sys.argv[1]); feas2.RELAX = bool(f["relax"])
s, frames = feas2.rollout((f["x"], f["qs"], True))
m = feas2.model(); m.vis.headlight.ambient[:] = [0.6]*3
render.feas.M = m
idx = np.linspace(0, len(frames) - 1, 12).astype(int)
render.render_states([frames[i] for i in idx], sys.argv[2])
print("score", s, "n", len(frames))
