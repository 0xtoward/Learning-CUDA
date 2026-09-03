# Kernel Profile: prefill V0 coalesced

- Target: `./bin/v1_coalesced 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v1_coalesced`, `regex:.*gemm_v1_coalesced.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

DRAM/long-scoreboard bound with high nominal occupancy. NCU duration 29.11 ms, SM 82.03%, memory/DRAM 92.70%, DRAM 228.59 GB/s, achieved/theoretical occupancy 99.46/100%, 40 registers/thread, and 1024 B shared memory/block. The long-scoreboard ratio is 21.50 per issue-active cycle. High occupancy cannot hide the repeated global loads indefinitely.

SASS/PTX: `build/ir_v1_coalesced_sm_89/`; static body has 58 `LDG`, 29 `FFMA`, no `LDS/STS/BAR`, and no `HMMA`. The strongest next action is shared-memory reuse, represented by V1.

Confidence: high. Limitation: NCU replay duration is not the headline benchmark; cold-L2 median is 28.1856 ms in `bench_skill/results/prefill_v0_coalesced/benchmark.md`.
