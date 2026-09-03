#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-bin/v7_warptiling}
M=${2:-2048}; N=${3:-2048}; K=${4:-2048}
mkdir -p reports
NAME=$(basename "$BIN")
# Full set is slow. These sections cover the course's main questions:
# SOL, memory/coalescing, stalls, occupancy, source counters/bank conflicts.
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section LaunchStats \
  --section Occupancy \
  --section SourceCounters \
  --launch-skip 5 --launch-count 1 \
  --force-overwrite \
  -o "reports/${NAME}_${M}x${N}x${K}" \
  "$BIN" "$M" "$N" "$K" 8 5 --no-verify

echo "Open reports/${NAME}_${M}x${N}x${K}.ncu-rep in Nsight Compute GUI."
echo "If a section name differs in your version: ncu --list-sections"
