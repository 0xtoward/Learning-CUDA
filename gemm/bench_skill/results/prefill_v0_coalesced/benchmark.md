# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0_coalesced_solve.cu` |
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
| Mean Time (ms) | 28.2240 |
| Median Time (ms) | 28.1856 |
| P20 Time (ms) | 27.9220 |
| P80 Time (ms) | 28.5528 |
| Min Time (ms) | 27.3940 |
| Max Time (ms) | 28.9783 |
| Std dev (ms) | 0.4383 |
| Samples | 15 |
