# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `wmma_tf32_solve.cu` |
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
| Mean Time (ms) | 10.1251 |
| Median Time (ms) | 9.9671 |
| P20 Time (ms) | 9.9514 |
| P80 Time (ms) | 10.4251 |
| Min Time (ms) | 9.9267 |
| Max Time (ms) | 10.6199 |
| Std dev (ms) | 0.2641 |
| Samples | 10 |
