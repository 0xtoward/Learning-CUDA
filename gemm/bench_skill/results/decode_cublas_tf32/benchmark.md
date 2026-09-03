# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `cublas_tf32_solve.cu` |
| **Reference** | `gemm_ref.py` |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU |
| **Arch** | sm_89 |
| **Dims** | {'M': 16, 'N': 4096, 'K': 4096} |
| **Correctness** | PASS |
| **Timing Method** | cuda_event |
| **Prewarm Calls** | 1 |
| **Cache Mode** | torch_l2_thrash |
| **Timing Scope** | Preallocated/static tensors; solution and selected baselines exclude per-call input cloning. |

## Timing

| Metric | Solution |
|--------|----------:|
| Mean Time (ms) | 0.3529 |
| Median Time (ms) | 0.3533 |
| P20 Time (ms) | 0.3523 |
| P80 Time (ms) | 0.3533 |
| Min Time (ms) | 0.3512 |
| Max Time (ms) | 0.3543 |
| Std dev (ms) | 0.0008 |
| Samples | 15 |
