# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `cublas_fp32_solve.cu` |
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
| Mean Time (ms) | 2.1783 |
| Median Time (ms) | 2.0787 |
| P20 Time (ms) | 2.0777 |
| P80 Time (ms) | 2.1789 |
| Min Time (ms) | 2.0777 |
| Max Time (ms) | 2.5866 |
| Std dev (ms) | 0.2102 |
| Samples | 10 |
