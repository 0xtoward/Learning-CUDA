# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3_warp_solve.cu` |
| **Reference** | `gemm_ref.py` |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU |
| **Arch** | sm_89 |
| **Dims** | {'M': 512, 'N': 6144, 'K': 4096} |
| **Correctness** | PASS |
| **Timing Method** | cuda_event |
| **Prewarm Calls** | 1 |
| **Cache Mode** | torch_l2_thrash |
| **Timing Scope** | Preallocated/static tensors; solution and selected baselines exclude per-call input cloning. |

## Timing

| Metric | Solution |
|--------|----------:|
| Mean Time (ms) | 3.1554 |
| Median Time (ms) | 2.9788 |
| P20 Time (ms) | 2.9764 |
| P80 Time (ms) | 3.4675 |
| Min Time (ms) | 2.9716 |
| Max Time (ms) | 3.7673 |
| Std dev (ms) | 0.2966 |
| Samples | 10 |
