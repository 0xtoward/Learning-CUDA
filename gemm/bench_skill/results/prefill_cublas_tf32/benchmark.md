# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `cublas_tf32_solve.cu` |
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
| Mean Time (ms) | 1.4137 |
| Median Time (ms) | 1.4136 |
| P20 Time (ms) | 1.4121 |
| P80 Time (ms) | 1.4154 |
| Min Time (ms) | 1.4121 |
| Max Time (ms) | 1.4162 |
| Std dev (ms) | 0.0017 |
| Samples | 10 |
