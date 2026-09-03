# Kernel Profile: decode V0 coalesced

- Target: `./bin/v1_coalesced 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v1_coalesced`, `regex:.*gemm_v1_coalesced.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Skinny-M memory/latency case. NCU duration 0.573 ms, SM/memory 78.48%, DRAM 69.96%, occupancy 73.78% versus a 100% theoretical ceiling, 40 registers/thread. Long-scoreboard ratio is 11.80. Its small output tile wastes less row work than V1/V2, so it is unexpectedly competitive on `M=16`.

Source/roofline stages were intentionally not collected for the decode sweep; use the prefill V0 report and `build/ir_v1_coalesced_sm_89/` for ISA. Cold-L2 median is 0.6205 ms. Confidence: high for the shape comparison.
