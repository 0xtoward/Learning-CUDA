# Naive WMMA versus tuned CUTLASS

This is a cross-implementation optimization comparison, not a same-kernel
regression test. Lower occupancy or bandwidth utilization is therefore not
automatically a regression; elapsed time and the causal stall shift decide.

| Metric | Naive | CUTLASS | Change |
|---|---:|---:|---:|
| Duration (ms) | 5.171216 | 0.706808 | -86.33%, 7.32x speedup |
| Tensor/HMMA pipe active | 5.26% | 49.44% | +44.18 points |
| L1/TEX throughput | 97.21% | 37.09% | -60.12 points |
| Long Scoreboard stall / issue | 47.94 | 0.19 | -99.60% |
| MIO Throttle stall / issue | 10.89 | 0.04 | -99.63% |
| Math Pipe Throttle stall / issue | 0.76 | 13.39 | math becomes the useful pressure point |
| Achieved occupancy | 48.875% | 15.895% | deliberate resource trade |
| Registers/thread | 40 | 140 | more register-resident fragments/state |
| Shared memory/CTA | 1,024 B | 37,888 B | staged reusable tiles |
| Spill read/write instructions | 0 / 0 | 0 / 0 | no runtime spilling |

Conclusion: the optimized implementation is faster because it changes where
work waits. The naive warp repeatedly waits on global data; CUTLASS spends more
on-chip resources to reuse and pipeline operands, raising Tensor Core activity
by about 9.4x while eliminating nearly all long-scoreboard stalls.
