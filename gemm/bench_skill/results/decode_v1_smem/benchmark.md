# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v1_smem_solve.cu` |
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
| Mean Time (ms) | 0.9111 |
| Median Time (ms) | 0.8950 |
| P20 Time (ms) | 0.8929 |
| P80 Time (ms) | 0.8964 |
| Min Time (ms) | 0.8909 |
| Max Time (ms) | 1.1438 |
| Std dev (ms) | 0.0644 |
| Samples | 15 |
