# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `wmma_tf32_block_solve.cu` |
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
| Mean Time (ms) | 6.4628 |
| Median Time (ms) | 6.5664 |
| P20 Time (ms) | 6.3111 |
| P80 Time (ms) | 6.6650 |
| Min Time (ms) | 5.7774 |
| Max Time (ms) | 6.8393 |
| Std dev (ms) | 0.3066 |
| Samples | 10 |
