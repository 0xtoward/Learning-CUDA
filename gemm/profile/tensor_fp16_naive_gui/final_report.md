# Kernel profile: naive FP16 WMMA

- Target: `./bin/tensorcore_fp16_lab naive 512 6144 4096 8 5 --no-verify`
- Filter: `regex:.*hgemm_wmma_naive.*`
- Launch window: skip 5, capture 1
- NCU duration: 5.171216 ms
- Tensor/HMMA pipe active: 5.26%
- Memory / L1TEX throughput: 96.75% / 97.21%
- Long Scoreboard / MIO stalls per issue: 47.94 / 10.89
- Achieved / theoretical occupancy: 48.875% / 50.0%
- Registers/thread: 40
- Spill instructions read/write: 0 / 0

Diagnosis: the kernel does use Tensor Cores, but direct global fragment loads
leave them starved. L1/TEX is saturated and long-latency data dependencies are
the dominant reason eligible warps cannot issue.
