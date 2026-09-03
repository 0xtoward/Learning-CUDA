# Nsys / NCU / PTX / SASS / Codex workflow

## Build
```bash
make ARCH=sm_89 -j
# 4070 = Ada SM89
```
`-lineinfo` is already enabled so Nsight Compute can map metrics back to source lines.
`-Xptxas=-v` prints register/shared-memory usage during compilation.

## Nsight Systems: first ask "where did time go?"
```bash
./scripts/profile_nsys.sh bin/v7_warptiling 2048 2048 2048
```
Look at:
- kernel duration and launch frequency
- CPU launch gaps
- memcpy / synchronization
- overlap between kernels/copies if you later build pipelines

## Nsight Compute: then ask "why is this kernel slow?"
```bash
./scripts/profile_ncu.sh bin/v7_warptiling 2048 2048 2048
```
Course-aligned checklist:
1. Speed Of Light: Compute vs Memory, and L1/L2/DRAM submetrics.
2. Source / L2 Theoretical Sectors Global Excessive: global coalescing.
3. Warp State: `Lg Throttle`, `MIO Throttle`, long/short scoreboard, barrier.
4. Memory Workload Analysis: cache hit rates, shared-memory transactions.
5. Source Counters: `L1 Wavefronts Shared Excessive` / bank-conflict evidence.
6. Launch/Occupancy: registers/thread, smem/block, active warps.

For the bank-conflict micro-demo:
```bash
./scripts/profile_bank_conflict.sh
```
Compare `conflict`, `padded`, and `swizzle` reports.

If profiling counters are denied (`ERR_NVGPUCTRPERM`), run with admin privileges or enable profiling access in the NVIDIA driver policy.

## PTX / SASS
```bash
./scripts/build_ir.sh kernels/v7_warptiling.cu sm_89
```
Outputs:
- `.ptx`: CUDA virtual ISA; the most useful "IR-ish" layer for CUDA device code.
- `.cubin`: machine-code container for the target SM.
- `.sass`: final native instructions dumped from cubin.
- `keep/`: NVCC intermediate files; exact names vary by toolkit.

Things to grep:
```bash
grep -E "ld\.global|st\.global|ld\.shared|st\.shared|fma|mad" build/ir_*/**/*.ptx 2>/dev/null || true
grep -E "LDG|STG|LDS|STS|FFMA" build/ir_*/*.sass
```
For vectorization, look for wider global operations such as 128-bit loads/stores in SASS. Exact mnemonics vary by architecture/toolkit.

## Optional LLVM IR with clang
NVCC is not an LLVM frontend. If your installed clang supports CUDA, extract a kernel-only source and try:
```bash
clang++ --cuda-device-only --cuda-gpu-arch=sm_89 --cuda-path=/usr/local/cuda \
  -O3 -S -emit-llvm kernel_only.cu -o kernel.ll
```
For CUDA optimization work, PTX + SASS are usually more directly useful than LLVM IR.

## Codex prompt template
After generating PTX/SASS and NCU text/CSV exports, point Codex at those files and ask:

> Compare `v4_2d_threadtile` and `v7_warptiling` source, optimized PTX and SASS. Trace GMEM->SMEM->register movement, identify wide loads/stores, shared-memory operations, FMA issue pattern, register pressure, barriers, and likely dependency chains. Cross-check each claim against the NCU report; do not infer a bank conflict from source alone when counters disagree. Suggest one change at a time and predict which NCU metric should move if the hypothesis is correct.

That last sentence is important: use Codex as a profiler-guided hypothesis generator, not as an oracle.
