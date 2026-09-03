# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v2_register_solve.cu` |
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
| Mean Time (ms) | 4.4548 |
| Median Time (ms) | 4.5138 |
| P20 Time (ms) | 4.4135 |
| P80 Time (ms) | 4.5922 |
| Min Time (ms) | 4.0090 |
| Max Time (ms) | 4.6428 |
| Std dev (ms) | 0.2025 |
| Samples | 15 |
