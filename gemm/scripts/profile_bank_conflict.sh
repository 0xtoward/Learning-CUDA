#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports
for mode in conflict padded swizzle; do
  echo "=== $mode ==="
  ncu --section MemoryWorkloadAnalysis --section SourceCounters \
      --launch-count 1 --force-overwrite \
      -o "reports/bank_${mode}" \
      ./bin/bank_conflict_demo "$mode"
done
echo "Compare L1 Wavefronts Shared Excessive / shared bank-conflict counters in the three reports."
