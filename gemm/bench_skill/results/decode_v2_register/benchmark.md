# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v2_register_solve.cu` |
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
| Mean Time (ms) | 0.9528 |
| Median Time (ms) | 0.8612 |
| P20 Time (ms) | 0.8602 |
| P80 Time (ms) | 0.9535 |
| Min Time (ms) | 0.8591 |
| Max Time (ms) | 1.3230 |
| Std dev (ms) | 0.1900 |
| Samples | 15 |
