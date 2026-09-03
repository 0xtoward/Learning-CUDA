# Kernel Profile: WMMA TF32 block-tiled GEMM

- Target: `./bin/wmma_tf32_block_tiled 512 6144 4096 8 5 --no-verify`
- Selected kernel: `gemm_wmma_tf32_block_tiled_aligned`
- Filter: `regex:.*gemm_wmma_tf32_block_tiled_aligned.*`
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, SM89
- NCU: 2025.3.1; privilege: none
- Launch window: skip 5 matching launches, profile 1
- Collected CSV stages: basic, memory, compute/instruction, occupancy/scheduler/warp-state, source SASS
- GUI report: `wmma_tf32_block_detailed.ncu-rep` (`detailed` set, source embedded)
- Exact commands: `commands.sh`

## Result

The kernel genuinely executes TF32 Tensor Core instructions, but the Tensor
Core pipe is underfed. The dominant evidence is global-memory dependency and
block-barrier waiting rather than register spilling.

| Evidence | Result | Interpretation |
|---|---:|---|
| NCU kernel duration | 6.4445 ms | Close to the separate cold-L2 benchmark |
| SM throughput | 57.36% | Moderate total SM utilization |
| Memory throughput | 64.11% | Memory side is more active than compute |
| DRAM throughput | 47.88%, 116.93 GB/s | Significant global-memory feed cost |
| L1/TEX throughput | 65.88% | Load/shared feed path is busy |
| Tensor/HMMA pipe | 11.89% | Tensor Core is present but poorly saturated |
| FP32 FMA pipe | 14.51% | Address/conversion/fallback arithmetic also consumes compute |
| Registers per thread | 40 | Not register-heavy |
| Achieved/theoretical occupancy | 93.97% / 100% | Occupancy is healthy |
| Allocated/static shared memory | 4096 / 3072 bytes | Includes 1024-byte driver-reserved block allocation |
| Long-scoreboard stall per issue | 11.58 | Waiting on global/L2 data is the largest stall |
| Barrier stall per issue | 4.22 | K=8 staging synchronizes too frequently |

## Runtime spill check

The compile-time `ptxas` output reported zero spill loads and stores. This NCU
run independently confirms the measured launch did not execute register spill
traffic:

| Runtime metric | Value |
|---|---:|
| `derived__local_spilling_requests` | 0 |
| `sass__inst_executed_register_spilling` | 0 |
| `sass__inst_executed_register_spilling_op_read` | 0 |
| `sass__inst_executed_register_spilling_op_write` | 0 |
| `memory_l2_theoretical_sectors_local` | 0 |

`launch__stack_size=1024` is a configured per-thread stack limit and is not
evidence that the kernel performed local-memory spills.

## ISA/source evidence

- CUDA source declares WMMA A/B/accumulator fragments and calls
  `wmma::load_matrix_sync`, `wmma::mma_sync`, and `wmma::store_matrix_sync`.
- PTX contains `wmma.load.*.m16n16k8.tf32` and
  `wmma.mma.sync.*.m16n16k8.f32.tf32.tf32.f32`.
- SASS contains four `HMMA.1684.F32.TF32` instructions in the loop body.
- This TF32 build does not contain `LDSM`; its WMMA loads lower to ordinary
  load instructions. `wmma::load_matrix_sync` therefore must not be assumed to
  map to `ldmatrix` for every type and architecture.

## Bottleneck and next experiment

Classification: **memory-feed and synchronization limited Tensor Core kernel**.
Confidence is high because the runtime throughput/stall counters, SASS, and
benchmark timing agree.

The next useful implementation experiment is not to increase occupancy. It is
to reduce K-loop barriers and improve the Global-to-Shared pipeline while
controlling register/shared-memory growth: vectorized or asynchronous staging,
multi-stage buffering, and a larger block/MMA schedule selected by measured
shape. The previous naive `BLOCK_K=32` attempt regressed because it raised
register and shared-memory pressure without a sufficiently good pipeline.

## Limitations

NCU replays a kernel to collect mutually incompatible hardware counters, so the
profiled application wall time is not a benchmark. Use the NCU kernel duration
and counters for diagnosis, and the separate CUDA-event benchmark for final
latency comparisons. Only one stabilized launch of this shape was profiled.
