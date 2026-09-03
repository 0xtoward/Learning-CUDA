# Kernel Profile: decode cuBLAS TF32 Tensor Core

- Target: `./bin/cublas_tf32_baseline 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `gemm`, `regex:.*gemm.*`; selected `cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Skinny Tensor-Core kernel limited by memory and available parallelism. NCU duration 0.295 ms, SM 20.26%, memory/DRAM 95.15%, occupancy 8.20/8.33%, 136 registers/thread, 74752 B shared memory/block. The demangled kernel name proves the s1688 TensorOp family; prefill compute/SASS evidence confirms HMMA for this explicit TF32 path.

Compute/source stages were intentionally omitted. Cold-L2 median is 0.3533 ms; correctness required atol=0.1, rtol=0.02. Low occupancy is expected for the resource-heavy Tensor-Core CTA and is not by itself a defect. Confidence: high for performance/dispatch, medium for numerical suitability.
