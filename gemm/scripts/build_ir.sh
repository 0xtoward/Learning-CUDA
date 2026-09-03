#!/usr/bin/env bash
set -euo pipefail
SRC=${1:-kernels/v7_warptiling.cu}
ARCH=${2:-sm_89}
BASE=$(basename "$SRC" .cu)
OUT="build/ir_${BASE}_${ARCH}"
mkdir -p "$OUT/keep"
COMPUTE=${ARCH/sm_/compute_}

# 1) Keep NVCC intermediates (version-dependent exact filenames).
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Iinclude -keep -keep-dir "$OUT/keep" "$SRC" -o "$OUT/${BASE}"

# 2) PTX: CUDA's virtual ISA / most useful device-side IR-like representation.
nvcc -O3 -std=c++17 -lineinfo -arch="$COMPUTE" -code="$COMPUTE" -Iinclude -ptx "$SRC" -o "$OUT/${BASE}.ptx"

# 3) CUBIN and final SASS.
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Iinclude -cubin "$SRC" -o "$OUT/${BASE}.cubin"
cuobjdump --dump-sass "$OUT/${BASE}.cubin" > "$OUT/${BASE}.sass"
if command -v nvdisasm >/dev/null 2>&1; then
  nvdisasm -g "$OUT/${BASE}.cubin" > "$OUT/${BASE}.nvdisasm.txt" || true
fi

echo "Generated:"
find "$OUT" -maxdepth 2 -type f -printf '  %p\n' | sort
