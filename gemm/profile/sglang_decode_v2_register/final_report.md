# Kernel Profile: decode V2 register tile

- Target: `./bin/v4_2d_threadtile 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v4_2d`, `regex:.*gemm_v4_2d.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Grid-underfilled and register-limited skinny-M case. NCU duration 1.055 ms, SM 23.92%, memory 32.23%, achieved/theoretical occupancy 7.44/33.33%, and 128 registers/thread. The 128-row tile is a poor match for `M=16`; most potential output rows are inactive and too few CTAs are available to fill all SMs.

Source/roofline stages were intentionally omitted; ISA is in `build/ir_v4_2d_threadtile_sm_89/`. Cold-L2 median is 0.8612 ms. Next action: specialize `BM` for decode and reduce accumulator footprint. Confidence: high.
