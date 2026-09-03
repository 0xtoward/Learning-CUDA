# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0_sum_solve.cu` |
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
| Mean Time (ms) | 0.8264 |
| Median Time (ms) | 0.7311 |
| P20 Time (ms) | 0.7311 |
| P80 Time (ms) | 0.7322 |
| Min Time (ms) | 0.7301 |
| Max Time (ms) | 1.6179 |
| Std dev (ms) | 0.2388 |
| Samples | 50 |
