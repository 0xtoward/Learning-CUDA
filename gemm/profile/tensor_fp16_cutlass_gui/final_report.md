# Kernel profile: tuned CUTLASS FP16 TensorOp

- Target: `./bin/tensorcore_fp16_cutlass_lab 64x128 512 6144 4096 8 5 --no-verify`
- Filter: `regex:.*cutlass::Kernel.*`
- Launch window: skip 5, capture 1
- NCU duration: 0.706808 ms
- Tensor/HMMA pipe active: 49.44%
- Memory / L1TEX throughput: 47.70% / 37.09%
- Long Scoreboard / MIO stalls per issue: 0.19 / 0.04
- Math Pipe Throttle per issue: 13.39
- Achieved / theoretical occupancy: 15.895% / 16.667%
- Registers/thread: 140; shared memory/CTA: 37,888 bytes
- Spill instructions read/write: 0 / 0

Diagnosis: block/warp reuse and multistage copies remove the naive kernel's
global-load dependency. The kernel accepts lower occupancy to hold more live
fragments and staged shared tiles, but it keeps Tensor Cores busy and shifts the
bottleneck toward useful math-pipe pressure.
