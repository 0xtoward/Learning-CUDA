# CUDA Reduce optimization ladder

`reduce(x, op)` is not limited to addition. The parallel tree requires an
associative operation and an identity value. This lab instantiates:

- sum: `op(a,b)=a+b`, identity `0`;
- max: `op(a,b)=max(a,b)`, identity `-FLT_MAX`;
- min: `op(a,b)=min(a,b)`, identity `+FLT_MAX`.

Argmax is the same pattern over `{value,index}` pairs with a tie-breaking
operator. Floating-point sum is only approximately associative, so different
tree orders can differ in the low bits.

Measurements use FP32 sum, `N=16,777,216`, an RTX 4070 Laptop GPU, 50 CUDA-event
trials after 10 warmups, preallocated buffers, and an L2-thrash between trials.
Effective bandwidth counts the 64 MiB input once; the tiny partial/output traffic
does not materially change the result.

| Stage | Problem being solved | Data level | Core shape | Median / effective BW | Newer architecture continuation |
|---|---|---|---|---:|---|
| V0 Interleaved | make a complete multi-block reduction correct | GMEM -> SMEM | one input/thread; modulo-based interleaved tree; repeated partial kernels | 0.7311 ms / 91.79 GB/s | no architecture-specific feature needed |
| V1 Sequential | remove divergence and halve first-pass blocks | GMEM -> SMEM | two inputs/thread; contiguous active lanes | 0.3553 ms / 188.88 GB/s | use wider vector loads when aligned |
| V2 Warp Shuffle | stop using SMEM/barriers inside each warp | registers + one shared value/warp | `__shfl_down_sync`, hierarchical warp/block tree | 0.3543 ms / 189.42 GB/s | cooperative groups collectives can improve portability/readability |
| V3 Vectorized Grid-Stride | reduce load instructions and intermediate launches | GMEM -> registers -> warp -> block | `float4`, grid-stride accumulation, grid capped near resident work, usually two launches | 0.3523 ms / 190.49 GB/s | Hopper+: persistent/cooperative reduction, thread-block clusters/Distributed Shared Memory for cross-CTA aggregation when worthwhile |
| CUB DeviceReduce | production-quality tuned library baseline | full hierarchy | architecture-specialized dispatch | 0.3533 ms / 189.95 GB/s | newer CUB/CCCL automatically selects newer policies |

V3 and CUB differ by only 0.3% in the medians, below what should be treated as
a stable win on a laptop GPU. NCU reports that V3's first pass already reaches
96.3% DRAM throughput, so the meaningful next optimization is usually to fuse
the reduction into its consumer (softmax, RMSNorm, argmax, and so on).

Do not compare only kernel arithmetic. A production benchmark must also state
whether it measures one reduction kernel, the complete multi-pass scalar
result, temporary allocation, framework dispatch, cache policy, input size,
dtype, operator, and determinism contract.
