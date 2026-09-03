# NCU report: CUDA Core 2-D register-tiled GEMM

## Contract

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (SM89)
- Kernel: `gemm_v4_2d`
- FP32 row-major GEMM, `M=512, N=6144, K=4096`
- Target: `./bin/v4_2d_threadtile 512 6144 4096 12 5 --no-verify`
- NCU: skip 5 launches, profile 1 launch, all analysis stages plus GUI report

## Key measurements

| Metric | Value |
|---|---:|
| NCU duration | 4.564432 ms |
| SM throughput | 57.78% |
| Memory throughput | 61.87% |
| DRAM throughput / measured bandwidth | 38.78% / 100.96 GB/s |
| FP32 FMA pipe active | 36.36% |
| Tensor pipe active | 0% |
| Registers / thread | 128 |
| Static shared memory / CTA | 5,120 B |
| Achieved / theoretical occupancy | 30.69% / 33.33% |
| Long-scoreboard per-issue-active ratio | 1.29 |

## Diagnosis

This is the useful middle state between SMEM tiling and explicit warp tiling.
The `8x8` per-thread accumulator increases A/B register reuse and largely
removes waiting on uncached global data: long-scoreboard pressure is low and
the FP32 FMA pipe is now doing meaningful work. The cost is 128 registers per
thread, which limits occupancy to about one third.

The next change is therefore not “add more cache”. Split the `64x64` block tile
into explicit warp tiles and reduce the per-thread micro-tile so that register
reuse, instruction-level parallelism, and resident warps are balanced. In this
lab the later warp-tiled version uses 80 registers/thread and reaches 7.508
TFLOP/s under the CUDA-event benchmark contract.

## Open and reproduce

- GUI report: `ccore_v2_register_detailed.ncu-rep`
- Exact generated commands: `commands.sh`
- Run metadata: `run_manifest.yaml`
- Machine-readable metrics: local `details/metrics_summary.json`

In the NCU GUI, inspect `Summary -> Speed Of Light`, then `Details -> Compute
Workload Analysis`, `Occupancy`, `Warp State Statistics`, and finally `Source`
to associate the hot SASS instructions with the register-tiled inner loop.
