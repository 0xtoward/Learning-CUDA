# Kernel Profile: decode V1 shared-memory tile

- Target: `./bin/v2_smem_tiled 16 4096 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v2_smem_tiled`, `regex:.*gemm_v2_smem_tiled.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, occupancy/launch/scheduler/warp-state; exact commands in `commands.sh`

## Classification and evidence

Shape-mismatched CTA and synchronization bound. NCU duration 1.045 ms, SM/memory 67.01%, DRAM 30.18%, occupancy 66.73/66.67%, 38 registers/thread, 9216 B shared memory/block. A 32-row tile launches half of its row threads out of bounds for `M=16`, while all valid threads still cross barriers. Barrier ratio is 2.35.

Source/roofline stages were intentionally omitted; ISA is in `build/ir_v2_smem_tiled_sm_89/`. Cold-L2 median is 0.8950 ms, slower than V0. Next action: a decode-specific row tile, not more shared memory. Confidence: high.
