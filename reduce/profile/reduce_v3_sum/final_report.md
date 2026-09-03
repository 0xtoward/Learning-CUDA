# NCU report: `reduce_v3_vectorized`

The profiled first pass is bandwidth-bound and already close to the available
DRAM limit. This supports treating V3 and CUB as effectively tied in the
independent CUDA-event benchmark and shifting future effort toward fusion.

| Metric | Value | Interpretation |
|---|---:|---|
| Duration under NCU | 0.280688 ms | profiler-replay timing; do not substitute for the independent benchmark |
| DRAM throughput | 260.59 GB/s / 96.3% | the main bottleneck is reading the input tensor |
| SM throughput | 1.56% | very little arithmetic per loaded byte, as expected for reduction |
| L2 throughput | 26.6% | not the limiting unit in this capture |
| Achieved occupancy | 91.25% | enough resident warps to hide the simple reduction latency |
| Registers / thread | 34 | no spill stores or spill loads were generated |
| Shared memory / CTA | 1,152 B | 128 B static warp partials plus profiler-reported driver overhead |

The Source/SASS capture contains `LDG.E.128`, confirming that the `float4`
source load compiled to a 128-bit global load. The exact capture recipe is in
`commands.sh`; the raw normalized metrics remain under `details/` locally.

- Backend: NVIDIA Nsight Compute
- Kernel filter: `.*reduce_v3_vectorized.*`
- Launch window: `--launch-skip 5 --launch-count 1`
- Target: `./bin/reduce_benchmark v3 sum 16777216 8 5 --no-verify`
