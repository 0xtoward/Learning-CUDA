# Kernel Profile: prefill cuBLAS FP32

- Target: `./bin/cublas_baseline 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `sgemm`, `regex:.*sgemm.*`; selected kernel is a tuned SIMT SGEMM
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

Scalar-FP32 compute-side baseline. NCU duration 2.45 ms, SM 73.03%, memory 57.64%, DRAM 25.94%/68.97 GB/s, occupancy 32.85/33.33%, 122 registers/thread, FMA pipe 63.32%, tensor/HMMA pipe 0%. The lower DRAM pressure and higher FMA utilization explain the remaining gap to V3.

Vendor source is unavailable; `details/07_source_raw.csv` contains SASS and confirms `FFMA` with no `HMMA`. Next action for a custom equal-precision kernel is pipeline/vectorization and shape specialization, not assuming cuBLAS used Tensor Cores.

Confidence: high. Cold-L2 median is 2.3593 ms and correctness passed at atol/rtol 0.005.
