#!/usr/bin/env bash
# Arms design study on the GPU: both-feet get-up search for each arm variant.
# Results: logs/getup_arms/<face>_L<len>[_elbow].npz + .log ; then on any machine:
#   uv run python scripts/qdd_getup_search/arms_replay.py logs/getup_arms/<run>.npz --gif <run>.gif
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=logs/getup_arms; mkdir -p "$OUT"
ITERS=${ITERS:-120}
for face in prone supine; do
  for len in 0.20 0.30; do
    for elbow in "" "--elbow"; do
      tag="${face}_L${len/0./}${elbow:+_elbow}"
      echo "== $tag"
      uv run python scripts/qdd_getup_search/gpu_cem.py --face "$face" --arm-len "$len" $elbow \
        --iters "$ITERS" --out "$OUT/$tag.npz" 2>&1 | grep -v -E "^Module |took .* ms" | tee "$OUT/$tag.log" | tail -n 3
    done
  done
done
for f in "$OUT"/*.log; do echo "$f: $(grep -E '^[0-9]+ best' "$f" | tail -n 1)"; done
