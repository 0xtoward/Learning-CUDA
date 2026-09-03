#!/usr/bin/env bash
set -euo pipefail

n="${1:-16777216}"
iterations="${2:-100}"
warmup="${3:-20}"

for op in sum max min; do
  for version in v0 v1 v2 v3 cub; do
    ./bin/reduce_benchmark "$version" "$op" "$n" "$iterations" "$warmup"
  done
done
