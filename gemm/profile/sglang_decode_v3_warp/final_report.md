# Kernel Profile: decode V3 warp tile

- Target: `./bin/v7_warptiling 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v7_warptiling`, `regex:.*gemm_v7_warptiling.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Shape-underfilled but better balanced than V2. NCU duration 0.584 ms, SM 29.45%, memory 52.42%, occupancy 14.75/50%, 80 registers/thread, 5376 B shared memory/block. The 128-row block tile still underfills `M=16`, but lower register pressure and warp mapping make it the fastest custom version.

Source/roofline stages were intentionally omitted; ISA is in `build/ir_v7_warptiling_sm_89/`. Cold-L2 median is 0.5417 ms, 81.3% of equal-precision cuBLAS FP32 throughput. Next action: decode-specific `BM=16/32` and possibly split-K. Confidence: high.
