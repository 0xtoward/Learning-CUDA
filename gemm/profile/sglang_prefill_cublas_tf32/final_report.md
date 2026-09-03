# Kernel Profile: prefill cuBLAS TF32 Tensor Core

- Target: `./bin/cublas_tf32_baseline 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `s1688gemm`, `regex:.*s1688gemm.*`; selected `cutlass_80_tensorop_s1688gemm...`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

Tensor-Core path with high per-CTA resource use. NCU duration 1.57 ms, SM 45.05%, memory/DRAM 46.95%/127.87 GB/s, occupancy 16.52/16.67%, 220 registers/thread, 74752 B shared memory/block, tensor and HMMA pipe 45.53%. SASS contains `HMMA.1688.F32.TF32`; this is not the same numerical contract as FP32 FFMA.

Vendor source is unavailable; use `details/07_source_raw.csv` for the selected SASS. Cold-L2 median is 1.4141 ms, but normally distributed `K=4096` inputs required atol=0.1 and rtol=0.02. Report speed and error together.

Confidence: high for ISA/performance classification, medium for application-level suitability because model-quality impact was not evaluated.
