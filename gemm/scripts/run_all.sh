#!/usr/bin/env bash
set -euo pipefail
M=${1:-1024}; N=${2:-1024}; K=${3:-1024}; ITERS=${4:-20}
for b in bin/v{0_naive_bad_mapping,1_coalesced,2_smem_tiled,3_1d_threadtile,4_2d_threadtile,5_2d_xor_swizzle,6_2d_vectorized,7_warptiling}; do
  echo "================ $b ================"
  "$b" "$M" "$N" "$K" "$ITERS" 5 --no-verify
  echo
done
