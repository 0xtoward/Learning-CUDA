# Kernel Profile: decode cuBLAS FP32

- Target: `./bin/cublas_baseline 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `sgemm`, `regex:.*sgemm.*`; selected `cutlass_80_simt_sgemm_128x32_8x5_nn_align1`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Skinny SIMT SGEMM dominated by memory traffic. NCU duration 0.292 ms, SM 28.80%, memory/DRAM 93.82%, occupancy 24.03/25%, 80 registers/thread, 26624 B shared memory/block. cuBLAS selected a different kernel family from prefill, demonstrating shape dispatch.

Compute/source stages were intentionally omitted. Cold-L2 median is 0.4403 ms; correctness passed at atol/rtol 0.005. Next custom-kernel action is a skinny-M tile or split-K strategy. Confidence: high.
