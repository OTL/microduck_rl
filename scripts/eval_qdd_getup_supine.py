"""Stood fraction of the supine get-up with arms (Mjlab-Getup-Supine-QddBipedArms)
from each kind of start, on CPU.

  uv run python scripts/eval_qdd_getup_supine.py zero          # the found motion alone
  uv run python scripts/eval_qdd_getup_supine.py policy.onnx   # feed-forward + residual

"stood" = at the end of the episode the torso is within 15 deg of upright and its
height within 3 cm of standing.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.robot.qdd_biped_constants import STAND_BASE_Z
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.qdd_biped_arms import make_qdd_biped_getup_supine_arms_env_cfg

KINDS = ("push", "stride", "drop", "crouch", "bridge_supine", "path")


def run(onnx_path: str, kind: str, n_envs: int, play: bool, seed: int) -> float:
    torch.manual_seed(seed)
    cfg = make_qdd_biped_getup_supine_arms_env_cfg(play=play)
    cfg.scene.num_envs = n_envs
    cfg.curriculum = {}
    r = cfg.events["reset_getup"].params
    r["kind_probs"] = {kind: 1.0}
    r["standing_prob"] = 0.0
    r["fixed_order"] = False  # play cfgs replay library states in order
    r["frontiers"] = {}  # whole range of each kind (no reverse curriculum)
    if onnx_path == "zero":
        cfg.actions["joint_pos"].zero_residual = True
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    sess = None
    if onnx_path != "zero":
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
    obs, _ = env.reset()
    steps = int(round(cfg.episode_length_s / env.step_dt)) - 2
    robot = env.scene["robot"]
    for _ in range(steps):
        if sess is None:
            act = torch.zeros(n_envs, env.action_manager.total_action_dim)
        else:
            x = obs["actor"].numpy().astype(np.float32)
            act = torch.tensor(sess.run(None, {sess.get_inputs()[0].name: x})[0])
        obs, *_ = env.step(act)
    up = microduck_mdp.qdd_torso_up(env)
    z = microduck_mdp.qdd_root_height(env)
    ok = (up > math.cos(math.radians(15))) & ((z - STAND_BASE_Z).abs() < 0.03)
    env.close()
    return float(ok.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="policy ONNX, or 'zero' for the feed-forward alone")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--play", action="store_true", help="no domain randomisation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in a.kinds:
        print(f"{k:14s} stood {run(a.onnx, k, a.n, a.play, a.seed):.2f}", flush=True)


if __name__ == "__main__":
    main()
