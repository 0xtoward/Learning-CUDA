# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3_pipeline_solve.cu` |
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
| Mean Time (ms) | 4.1212 |
| Median Time (ms) | 3.9875 |
| P20 Time (ms) | 3.7540 |
| P80 Time (ms) | 4.4286 |
| Min Time (ms) | 3.7530 |
| Max Time (ms) | 4.9183 |
| Std dev (ms) | 0.4336 |
| Samples | 10 |
