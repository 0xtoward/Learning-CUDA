# Kernel Profile: prefill V2 register tile

- Target: `./bin/v4_2d_threadtile 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v4_2d`, `regex:.*gemm_v4_2d.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

Register-limited FP32 FMA kernel. NCU duration 4.83 ms, SM 56.67%, memory 60.68%, DRAM 39.22%/96.31 GB/s, occupancy 30.75/33.33%, 128 registers/thread, 5120 B shared memory/block, FMA pipe 36.28%. DRAM is no longer the sole roof; the register footprint restricts resident warps.

SASS/PTX: `build/ir_v4_2d_threadtile_sm_89/`; 512 static `FFMA`, 14 `LDG`, 32 `LDS`, 14 `STS`, 2 barriers, no `HMMA`. Next action: keep 2-D reuse while reducing per-thread accumulator/register pressure and improving warp-level mapping (V3).

Confidence: high. Cold-L2 median is 4.5138 ms.
