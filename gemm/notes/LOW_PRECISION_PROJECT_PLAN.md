# MXFP8 / NVFP4 project extension plan

This is the recommended bridge from the course GEMM work to the low-precision project.

## Phase A: reference numerics first
1. CPU bit-exact-ish software encoders/decoders for E4M3, E8M0 and E2M1.
2. Golden tests around 0, subnormals/underflow policy, max finite, saturation, ties, negatives, NaN/Inf input policy.
3. MXFP8: 32 consecutive E4M3 values + one E8M0 block scale.
4. NVFP4: 16 E2M1 values + one E4M3 local scale + one FP32 global tensor scale; pack two FP4 nibbles per byte.

## Phase B: CUDA functional path
- One warp handles one 32-element MXFP8 block: each lane loads one element, warp-reduces `amax`, derives E8M0 scale, quantizes from registers, writes 32 bytes + 1 scale byte.
- One warp handles two 16-element NVFP4 blocks: use two 16-lane subgroups, subgroup reduce `amax`, quantize E2M1, pair lanes for nibble packing.
- Dequantize to FP16/BF16/FP32 with packed loads and vectorized output stores.

## Phase C: profiler-driven optimization
1. Fuse load + amax + quantization so each source value is read once and kept in registers.
2. Vectorize input/output (`float4`, `half2`, packed 32/64/128-bit words where aligned).
3. Make scale traffic coalesced and contiguous.
4. Compare one-warp-per-block vs multiple blocks/warp mapping.
5. Add stochastic rounding after nearest-rounding correctness is locked.

## Phase D: Marlin-lite stretch goal on Ada
The base assignment is quantize/dequantize, not GEMM. To learn Tensor Cores, add an optional fused W4A16-style path:

`packed NVFP4-like weights -> unpack/dequant in registers -> FP16 fragments -> mma.sync/WMMA -> FP32 accum`

This is "Marlin-like" in spirit (packed 4-bit weight layout, scale-aware register dequantization, hierarchical tiling, software pipeline), while remaining a software FP4 emulation path on Ada. It will not be native NVFP4 Tensor Core compute.

## Phase E: native hardware comparison
- Ada SM89 (RTX 4070): native FP8 Tensor Core support is available, but MXFP8/NVFP4 block-scaled Tensor Core paths are not native.
- Hopper SM90 (H100/H200): excellent FP8 reference; not the native target for MXFP8/NVFP4.
- Blackwell SM100/SM120+: native block-scaled narrow-precision paths. Use B200/B300 for the safest datacenter reference, or RTX 5090/5080 for a cheaper GeForce Blackwell experiment. CUTLASS has SM120 NVFP4 examples.
