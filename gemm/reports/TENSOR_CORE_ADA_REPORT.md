# Ada FP16 Tensor Core GEMM report

## Scope

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, Ada SM 8.9
- Workload: SGLang-style prefill QKV projection
- Shape: `[512,4096] x [4096,6144] -> [512,6144]`
- Arithmetic: FP16 A/B, FP32 accumulation and C
- FLOPs per launch: `2*M*N*K = 25,769,803,776`
- Correctness: every version passed against FP32 PyTorch reference with FP16
  inputs (`atol=0.025`, `rtol=0.003`)

The custom implementation does not copy cuBLAS. cuBLAS is a binary SDK library,
and the CUDA SDK license prohibits reverse engineering/decompiling/disassembling
the SDK. The implementation instead uses documented WMMA/PTX behavior,
open-source CUTLASS, and the MIT-licensed `spatters/mma-matmul` Ada example. Its
license is preserved in `licenses/mma-matmul-MIT.txt`.

## Implementations

### Naive WMMA

Each warp owns one `16x16` output tile. For every `K=16` step it directly loads
A and B fragments from global memory, executes `wmma::mma_sync`, and eventually
stores the accumulator. It proves the Tensor Core arithmetic, but has almost no
inter-warp/block operand reuse and no software pipeline.

### Handwritten Ada pipeline

The optimized teaching kernel exposes the machinery hidden by WMMA:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` performs warp-wide MMA;
- 128-bit `cp.async` loads populate a three-stage circular shared-memory buffer;
- XOR-permuted shared layouts avoid conflicts for warp matrix loads;
- `ldmatrix.sync.aligned.x4/x2` loads lane-distributed A/B registers;
- CTA and warp tiling reuse one loaded tile for many output fragments;
- accumulators remain in registers until the final coalesced epilogue;
- B is transposed/prepacked on the GPU once and cached, modeling a persistent
  inference weight layout. Prepack time is excluded from steady-state GEMM time.

### CUTLASS tuned variant

The best tested CUTLASS configuration for this shape uses threadblock tile
`64x128x32`, warp tile `32x64x32`, instruction tile `16x8x16`, and three stages.
CUTLASS supplies production-quality predication, iterators, shared-memory
layouts, multistage scheduling, and epilogue handling while remaining readable
open-source reference code.

## Controlled benchmark

CUDA events time only the preallocated GEMM call. Trials thrash L2 between
launches, use seed 42, discard the first three samples, and run independently.

| Kernel | Samples | Median (ms) | TFLOP/s | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive WMMA | 50 | 8.8417 | 2.91 | 1.00x |
| Handwritten Ada pipeline | 50 | 1.0199 | 25.27 | 8.67x |
| CUTLASS 64x128x32, 3-stage | 100 | 0.6769 | 38.07 | 13.06x |
| cuBLAS | 100 | 0.6994 | 36.85 | 12.64x |

On this one shape/protocol, CUTLASS is about 3.3% faster than the selected
cuBLAS path. This is not a general library ranking. In the warm-cache standalone
driver, cuBLAS measured 0.601 ms and was faster than all custom variants. Laptop
display activity, clocks, thermals, cache state, and cuBLAS algorithm selection
all matter.

## NCU diagnosis: naive versus CUTLASS

Both detailed captures profile one launch after five skipped launches.

| Metric | Naive WMMA | CUTLASS | Meaning |
|---|---:|---:|---|
| NCU duration | 5.171 ms | 0.7068 ms | 7.32x faster in this profiling run |
| Tensor/HMMA pipe active | 5.26% | 49.44% | Tensor Cores are fed instead of mostly idle |
| L1/TEX throughput | 97.21% | 37.09% | naive pounds the load path |
| Long Scoreboard stall / issue | 47.94 | 0.19 | global-memory dependency nearly removed |
| MIO Throttle stall / issue | 10.89 | 0.04 | matrix/load issue pressure is relieved |
| Math Pipe Throttle stall / issue | 0.76 | 13.39 | tuned kernel now waits on useful math capacity |
| Achieved occupancy | 48.88% | 15.90% | lower occupancy is an intentional resource trade |
| Registers/thread | 40 | 140 | more fragments and pipelined state live in registers |
| Shared memory/CTA | 1 KiB driver reserve | 37 KiB | staged, reused operand tiles |
| Spill instructions read/write | 0 / 0 | 0 / 0 | high register use did not spill |

This is the important profiler story: T0 is not slow because it failed to use
Tensor Cores. It is slow because it cannot feed them. The tuned kernel spends
more registers/shared memory to remove redundant global loads and hide latency;
the bottleneck shifts from long-scoreboard memory waits toward Tensor math pipe
availability.

The source/SASS capture confirms `HMMA.16816.F32` in the naive kernel. The tuned
kernel additionally contains 128-bit `LDGSTS` asynchronous global-to-shared
copies and `LDSM` warp matrix loads around its `HMMA` instructions.

## Artifacts

- `bench_skill/results/tensor_fp16_naive/benchmark.md`
- `bench_skill/results/tensor_fp16_ada_hand/benchmark.md`
- `bench_skill/results/tensor_fp16_cutlass_s3_round2/benchmark.md`
- `bench_skill/results/tensor_fp16_cublas_round2/benchmark.md`
- `profile/tensor_fp16_naive_gui/tensor_fp16_naive_detailed.ncu-rep`
- `profile/tensor_fp16_cutlass_gui/tensor_fp16_cutlass_detailed.ncu-rep`
- `profile/tensor_fp16_cutlass_gui/comparison/naive_vs_cutlass.md`
