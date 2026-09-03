# SGLang-like GEMM profiling report

## Scope and shape validation

The reference model is Llama 3.1 8B at tensor parallel size 1: hidden size 4096, 32 query heads, 8 KV heads, head dimension 128, and intermediate size 14336. Meta's model registry records the 4096/32/8 architecture and GQA; SGLang's Llama implementation constructs a fused `QKVParallelLinear`, fused `gate_up_proj`, and row-parallel `o_proj`/`down_proj`.

SGLang does not require a rectangular user-level batch. It stores `batch_size`, a flat `input_ids` tensor, and `extend_num_tokens`; `EXTEND` is prefill and `DECODE` processes one token per active request. Therefore the linear-layer GEMM row count is the number of scheduled tokens on this rank:

- Prefill/extend: `M = sum(request extend lengths)`; the experiment uses 4 requests × 128 new tokens = 512.
- Decode: `M = active request count` before graph/DP padding; the experiment uses 16.
- With TP=1, fused QKV is `(M,4096) x (4096,6144)` because `(32 + 2*8) * 128 = 6144`.
- The decode control uses the attention output projection `(16,4096) x (4096,4096)`.
- MLP is larger in total FLOPs: gate+up has `N=28672`, down has `K=14336`. The exhaustive NCU matrix uses QKV so all six implementations remain practical on the 8 GB laptop GPU.

Sources: [SGLang Llama projections](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py), [SGLang forward-batch modes and token fields](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py), [SGLang TP linear semantics](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py), [Meta Llama 3.1 registry](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py).

## Four teaching versions

The compact V0-V3 names map to existing files as follows:

| Teaching name | Existing binary | Main idea |
|---|---|---|
| V0 | `bin/v1_coalesced` | one output/thread, coalesced global access |
| V1 | `bin/v2_smem_tiled` | shared-memory tiling |
| V2 | `bin/v4_2d_threadtile` | 2-D register tile per thread |
| V3 | `bin/v7_warptiling` | block/warp/thread hierarchy |

The original `v0_naive_bad_mapping` remains a negative control rather than one of the four teaching stages.

## Timing result

These are CUDA-event medians from 15 measured cold-L2 trials after 5 warmups and one discarded sample. Full distribution data is in `sglang_gemm_benchmark_summary.csv`.

| Scenario | V0 | V1 | V2 | V3 | cuBLAS FP32 | cuBLAS TF32 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill QKV, ms | 28.1856 | 17.3670 | 4.5138 | 3.4324 | 2.3593 | 1.4141 |
| Prefill QKV, TFLOP/s | 0.914 | 1.484 | 5.709 | 7.508 | 10.923 | 18.223 |
| Decode output, ms | 0.6205 | 0.8950 | 0.8612 | 0.5417 | 0.4403 | 0.3533 |
| Decode output, TFLOP/s | 0.865 | 0.600 | 0.623 | 0.991 | 1.219 | 1.520 |

V3 reaches 68.7% of FP32 cuBLAS on prefill and 81.3% on decode. V1/V2 regress on decode because their tiles were designed for large `M`: V1 launches 32 row threads for only 16 rows, while V2's 128-row tile underfills the grid and carries 128 registers/thread. This is the useful lesson: a monotonic square-GEMM optimization ladder is not a monotonic inference-shape ladder.

TF32 is a different numerical contract. It required `atol=0.1, rtol=0.02` for normally distributed `K=4096` data; the FP32 paths passed `0.005/0.005`. Do not report the 1.67× TF32 speedup as an equal-precision comparison.

## NCU evidence

The full prefill profiles collected `basic`, Speed of Light, memory, compute, occupancy/launch, roofline, and SASS source pages. Decode profiles collected `basic`, Speed of Light, and occupancy because the purpose was shape utilization.

| Variant | NCU ms | SM % | memory % | DRAM GB/s | occupancy % | regs/thread | FMA % | tensor/HMMA % | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 | 29.11 | 82.0 | 92.7 | 228.6 | 99.5 | 40 | 6.2 | 0 | repeated global loads; bandwidth/scoreboard bound |
| V1 | 21.58 | 75.8 | 75.8 | 80.7 | 66.7 | 38 | 7.9 | 0 | DRAM reduced, but 1024-thread CTA and barriers dominate |
| V2 | 4.83 | 56.7 | 60.7 | 96.3 | 30.8 | 128 | 36.3 | 0 | register reuse raises FMA activity; register-limited occupancy |
| V3 | 3.44 | 54.2 | 77.1 | 164.5 | 46.3 | 80 | 45.4 | 0 | better FMA feed and warp mapping; mixed memory/FMA limit |
| cuBLAS FP32 | 2.45 | 73.0 | 57.6 | 69.0 | 32.9 | 122 | 63.3 | 0 | tuned SIMT SGEMM using scalar FFMA |
| cuBLAS TF32 | 1.57 | 45.1 | 47.0 | 127.9 | 16.5 | 220 | 3.2 | 45.5 | Tensor Core HMMA; high register/shared-memory footprint |

NCU's section replay changes runtime and may save/restore memory. Its `gpu__time_duration` is supporting evidence, not the headline benchmark. CUDA-event distributions are the latency result.

## IR/SASS evidence

The custom kernels compile to scalar FP32 instructions. Static SASS opcode counts are:

| Variant | FFMA | HMMA | LDG | LDS | STS | BAR | Registers |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 29 | 0 | 58 | 0 | 0 | 0 | 40 |
| V1 | 32 | 0 | 2 | 40 | 2 | 2 | 38 |
| V2 | 512 | 0 | 14 | 32 | 14 | 2 | 128 |
| V3 | 256 | 0 | 10 | 25 | 12 | 2 | 80 |

These are static instruction instances after compiler unrolling, not dynamic executed counts. They show the code-shape transition: V0 performs global loads in the inner product; V1 moves reuse to `LDS/STS`; V2/V3 expose many independent accumulators to `FFMA`. Default cuBLAS also has no HMMA in this FP32 mode. Explicit TF32 cuBLAS contains `HMMA.1688.F32.TF32`, confirmed both by SASS and the 45.5% HMMA-pipe metric.

PTX and SASS are under `build/ir_*_sm_89/`; cuBLAS SASS is in each profile's `details/07_source_raw.csv`.

## Artifact map and limitations

- `profile/sglang_prefill_*`: six complete NCU evidence directories with raw CSV, normalized JSON, command manifest, hotspot table, and PNG.
- `profile/sglang_decode_*`: six shape-utilization NCU directories.
- `bench_skill/results/*/benchmark.md`: correctness and timing distributions generated by the benchmark skill.
- `profile/nsys/*.nsys-rep`: NVTX host ranges were captured, but this WSL Nsight Systems 2024.5 installation emitted no CUDA kernel track. These reports are not used for GPU-kernel timing. The current NVIDIA documentation describes WSL CUPTI support as preview and non-root same-user injection requirements; upgrading the Linux Nsight Systems package is the next retry.
- Laptop clocks and display contention were not locked. Differences under about 2% should be treated as noise; the major gaps here are much larger.

The step-by-step method is in `notes/PERFORMANCE_ANALYSIS_TUTORIAL.md`.
