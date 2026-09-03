# NCU report: handwritten Ada Tensor Core GEMM

## Contract

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (SM89)
- Kernel: `hgemm_mma_ada_pipeline`
- FP16 A/B, FP32 accumulation/output, row-major public interface
- Shape: `M=512, N=6144, K=4096`
- Target: `./bin/tensorcore_fp16_lab ada 512 6144 4096 12 5 --no-verify`
- NCU: skip 5 launches, profile 1 launch, all analysis stages plus GUI report

The timed GEMM consumes an already prepacked RHS. Persistent RHS packing is
outside the repeated-kernel timing, matching an inference weight-reuse case.

## Key measurements

| Metric | Value |
|---|---:|
| NCU duration | 1.081216 ms |
| SM throughput | 34.86% |
| Memory / DRAM throughput | 84.95% / 84.95% |
| Measured DRAM bandwidth | 188.72 GB/s |
| Tensor/HMMA pipe active | 38.26% |
| Registers / thread | 139 |
| Static shared memory / CTA | 50,176 B |
| Achieved / theoretical occupancy | 16.18% / 16.67% |
| Long-scoreboard per-issue-active ratio | 1.24 |
| Math-pipe-throttle per-issue-active ratio | 16.88 |

## Diagnosis

`cp.async -> permuted SMEM -> ldmatrix -> mma.sync` plus three pipeline stages
fixes the naive WMMA kernel's primary problem: Tensor Cores are no longer
waiting directly on global-memory fragment loads. Relative to the naive report,
Tensor-pipe activity rises from about 5.26% to 38.26% and the long-scoreboard
per-issue-active ratio falls from about 47.94 to 1.24.

The remaining cost is resource balance. A `128x128x32` CTA uses 256 threads,
139 registers/thread, and about 50 KiB shared memory, leaving only one CTA per
SM and about 16% achieved occupancy. DRAM throughput is still about 85%, while
math-pipe throttle shows that the now-fed MMA stream also creates dependency
and issue pressure. CUTLASS's selected `64x128x32` three-stage kernel improves
the measured duration to about 0.707 ms in NCU by using a better-balanced tile,
iterator, schedule, and epilogue.

## Open and reproduce

- GUI report: `tensor_fp16_ada_detailed.ncu-rep`
- Exact generated commands: `commands.sh`
- Run metadata: `run_manifest.yaml`
- Machine-readable metrics: local `details/metrics_summary.json`

In the NCU GUI, compare this report against the naive WMMA and CUTLASS reports.
Keep `M/N/K`, dtypes, accumulation type, layout, and RHS-prepack policy fixed;
different grid/block sizes are expected because each implementation assigns a
different output tile to a CTA.
