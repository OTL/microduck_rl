"""Export a found prone -> supine roll (gpu_cem.py --goal supine --roll side) as the
first stage of the chained get-up's reference (mdp.GetupReferenceJointPositionAction
roll_path): ref_keys (the keys, then the search's lying pose), ref_durs_s, ref_prep_s
and joint_names (legs + arms, entity names).

  uv run python scripts/qdd_getup_search/export_roll_ref.py \
      logs/getup_arms_v7/roll_side_s0.npz \
      --out src/mjlab_microduck/robot/qdd_biped/getup_roll_ref_arms.npz
"""

from __future__ import annotations

import argparse
import json

import arms_replay as R
import numpy as np
import torch

RENAME = {"l_shoulder": "l_shoulder_pitch", "r_shoulder": "r_shoulder_pitch"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="gpu_cem.py result npz (best, args)")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    f = np.load(o.result)
    a = argparse.Namespace(**json.loads(f["args"].item()))
    S = R.cpu_search(a)
    keys, durs = S.keys(torch.tensor(f["best"][None], dtype=torch.float32))
    allk = np.r_[keys[0].numpy(), S.q0.numpy()[None]]
    names = [RENAME.get(S.m.joint(i).name, S.m.joint(i).name) for i in range(1, S.m.njnt)]
    np.savez(
        o.out,
        joint_names=np.array(names),
        ref_keys=allk.astype(np.float32),
        ref_durs_s=durs[0].numpy().astype(np.float32),
        ref_prep_s=np.float32(a.prep),
        source=np.array(o.result),
    )
    print(f"{o.out}: {len(allk)} keys, prep {a.prep} s, total {a.prep + float(durs[0].sum()):.2f} s")


if __name__ == "__main__":
    main()
