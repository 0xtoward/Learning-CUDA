# CUDA GEMM Optimization Lab

Educational CUDA GEMM ladder corresponding to the course sequence:

`naive -> coalescing -> SMEM tiling -> 1D thread tiling -> 2D thread tiling -> swizzle/BCF -> vectorized access -> warp tiling`

## Build on RTX 4070
```bash
make ARCH=sm_89 -j
make ARCH=sm_89 nvtx -j
```

## Run
```bash
./bin/v0_naive_bad_mapping 1024 1024 1024 20 5
./bin/v7_warptiling         2048 2048 2048 20 5
./bin/v3_cp_async_double_buffered 2048 2048 2048 20 5
./bin/wmma_tf32_gemm        2048 2048 2048 20 5
./bin/wmma_tf32_block_tiled 2048 2048 2048 20 5
./bin/cublas_baseline       2048 2048 2048 20 5
./bin/cublas_tf32_baseline  2048 2048 2048 20 5
./scripts/run_all.sh 1024 1024 1024 20
```
CLI is `M N K iters warmup [--no-verify]`.

## Profile
```bash
./scripts/profile_nsys.sh bin/v7_warptiling 2048 2048 2048
./scripts/profile_ncu.sh  bin/v7_warptiling 2048 2048 2048
./scripts/profile_bank_conflict.sh
./scripts/build_ir.sh kernels/v7_warptiling.cu sm_89
```

Read:
- `notes/GEMM_OPT_TABLE.md`
- `notes/CODE_TEMPLATES.md`
- `notes/PROFILING_AND_IR.md`
- `notes/PERFORMANCE_ANALYSIS_TUTORIAL.md` — CUDA Event / NVTX / NCU / PTX / SASS / Roofline from first principles
- `notes/LOW_PRECISION_PROJECT_PLAN.md`
- `notes/TENSOR_CORE_AND_PIPELINE.md` — handwritten WMMA and `cp.async` experiments
- `reports/transformer_inference_shapes.md`
- `reports/SGLANG_GEMM_PROFILE_REPORT.md` — SGLang-grounded prefill/decode case study
- `reports/sglang_gemm_benchmark_summary.csv` — timing distributions and correctness
- `reports/sglang_gemm_ncu_summary.csv` — normalized profiler metrics

The reproducible Nsight Compute captures are under `profile/sglang_*`. Each
directory contains the exact command, run manifest, raw CSV, normalized JSON,
source hotspots, a compact report, and a rendered summary image. The two tested
Llama 3.1 8B / TP=1 shapes are:

- prefill QKV: `M=512, N=6144, K=4096`
- decode output projection: `M=16, N=4096, K=4096`

The benchmark-skill wrappers and generated reports are under `bench_skill/`.
The four teaching stages are `v1_coalesced` (V0), `v2_smem_tiled` (V1),
`v4_2d_threadtile` (V2), and `v7_warptiling` (V3).

`cublas_baseline` keeps the lab's default FP32 SGEMM behavior.
`cublas_tf32_baseline` explicitly enables the TF32 Tensor Core path so the
arithmetic and accuracy tradeoff is visible rather than implicit.

Three experimental teaching kernels live under `extras/`: one adds an
Ampere-or-newer `cp.async` two-stage pipeline to V3, and two use handwritten
WMMA TF32 Tensor Core code (one warp tile, then a reused block tile).  They are
intentionally kept
outside the four-stage CUDA-core learning ladder.

## Important
These are compact teaching kernels, deliberately easy to read. They are not expected to match cuBLAS/CUTLASS across shapes. Exact speedups depend on GPU, shape, clocks, toolkit, and compiler. The point is to correlate each code change with a profiler signal.
