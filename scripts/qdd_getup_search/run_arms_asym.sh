#!/usr/bin/env bash
# Asymmetric follow-up (OTL 2026-10-18): elbow arm 20 cm, left and right keys free + hip roll.
# Results: $OUT/<tag>.npz + .log ; replay with arms_replay.py as for run_arms_batch.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-logs/getup_arms_v3}; mkdir -p "$OUT"
ITERS=${ITERS:-200}
run() {
  tag=$1; shift
  echo "== $tag"
  uv run python scripts/qdd_getup_search/gpu_cem.py --arm-len 0.20 --elbow --asym \
    --iters "$ITERS" --out "$OUT/$tag.npz" "$@" 2>&1 | grep -v -E "^Module |took .* ms" | tee "$OUT/$tag.log" | tail -n 3
}
# refine the symmetric supine candidate (25/32) without the mirror constraint
run supine_asym_from_v2 --face supine --init logs/getup_arms_v2/supine_L20_elbow.npz --sd0 0.2
# fresh asymmetric searches
run supine_asym --face supine
run prone_asym --face prone
run prone_asym_s1 --face prone --seed 1
for f in "$OUT"/*.log; do echo "$f: $(grep -E '^[0-9]+ best' "$f" | tail -n 1)"; done
