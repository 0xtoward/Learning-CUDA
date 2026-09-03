# Kernel Profile: prefill V3 warp tile

- Target: `./bin/v7_warptiling 512 6144 4096 20 5 --no-verify`
- Kernel/filter: `gemm_v7_warptiling`, `regex:.*gemm_v7_warptiling.*`
- NCU: 2025.3.1; privilege: none; launch window: skip 5, count 3
- Sections: basic, SpeedOfLight, memory, compute, occupancy/launch/scheduler/warp-state, roofline, source SASS
- Exact collector commands: `commands.sh`; environment: `details/00_environment.txt`

## Classification and evidence

Mixed memory-feed and scalar-FMA limit. NCU duration 3.44 ms, SM 54.18%, memory 77.06%, DRAM 60.98%/164.47 GB/s, L1/TEX 79.65%, occupancy 46.26/50%, 80 registers/thread, FMA pipe 45.40%. Compared with V2, fewer registers and better warp mapping raise occupancy and FMA feed, although memory traffic remains important.

SASS/PTX: `build/ir_v7_warptiling_sm_89/`; 256 static `FFMA`, 10 `LDG`, 25 `LDS`, 12 `STS`, 2 barriers, no `HMMA`. Next actions: vectorized/asynchronous global-to-shared movement, double buffering, and shape-specialized tiles; Tensor Core use would be a separate numerical/ISA path.

Confidence: high. Cold-L2 median 3.4324 ms (68.7% of equal-precision cuBLAS FP32 throughput).
