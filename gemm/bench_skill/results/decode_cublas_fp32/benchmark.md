# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `cublas_fp32_solve.cu` |
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
| Mean Time (ms) | 0.4734 |
| Median Time (ms) | 0.4403 |
| P20 Time (ms) | 0.4393 |
| P80 Time (ms) | 0.4434 |
| Min Time (ms) | 0.4362 |
| Max Time (ms) | 0.9175 |
| Std dev (ms) | 0.1230 |
| Samples | 15 |
