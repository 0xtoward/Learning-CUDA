#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-bin_nvtx/v7_warptiling}
M=${2:-2048}; N=${3:-2048}; K=${4:-2048}
mkdir -p profile/nsys
NAME=$(basename "$BIN")
# WSL/non-root CUDA injection needs an explicit same-user shared-memory opt-in.
export CUDA_INJECTION_SHM_ALLOWED=TRUE
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "profile/nsys/${NAME}_${M}x${N}x${K}" \
  "$BIN" "$M" "$N" "$K" 20 5 --no-verify
REP="profile/nsys/${NAME}_${M}x${N}x${K}.nsys-rep"
echo "=== CUDA kernel summary ==="
nsys stats --force-export=true --report cuda_gpu_kern_sum "$REP" || \
  nsys stats --force-export=true --report gpukernsum "$REP"
echo "Tip: nsys stats --help-reports"
