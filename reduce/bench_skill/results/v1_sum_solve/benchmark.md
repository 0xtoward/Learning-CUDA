# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v1_sum_solve.cu` |
| **Reference** | `reduce_ref.py` |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU |
| **Arch** | sm_89 |
| **Dims** | {'n': 16777216} |
| **Correctness** | PASS |
| **Timing Method** | cuda_event |
| **Prewarm Calls** | 1 |
| **Cache Mode** | torch_l2_thrash |
| **Timing Scope** | Preallocated/static tensors; solution and selected baselines exclude per-call input cloning. |

## Timing

| Metric | Solution |
|--------|----------:|
| Mean Time (ms) | 0.3900 |
| Median Time (ms) | 0.3553 |
| P20 Time (ms) | 0.3551 |
| P80 Time (ms) | 0.3564 |
| Min Time (ms) | 0.3543 |
| Max Time (ms) | 0.9994 |
| Std dev (ms) | 0.1347 |
| Samples | 50 |
