# Kernel Profile: prefill V1 shared-memory tile

- Target: `./bin/v2_smem_tiled 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v2_smem_tiled`, `regex:.*gemm_v2_smem_tiled.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

Shared-memory reuse reduces DRAM traffic, but the 1024-thread CTA and barriers limit issue efficiency. NCU duration 21.58 ms, SM/memory 75.82%, DRAM 33.04%, DRAM 80.72 GB/s, occupancy 66.65/66.67%, 38 registers/thread, 9216 B shared memory/block. Barrier ratio is 2.51 and long-scoreboard ratio 5.41.

SASS/PTX: `build/ir_v2_smem_tiled_sm_89/`; static body has 2 `LDG`, 40 `LDS`, 2 `STS`, 2 barriers, 32 `FFMA`, no `HMMA`. Next action: assign multiple outputs per thread to amortize synchronization and shared loads (V2).

Confidence: high. Cold-L2 median is 17.3670 ms; p20-p80 is 17.1641-20.0782 ms, so this version also shows visible run-to-run clock/contention sensitivity.
