# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v1_smem_solve.cu` |
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
| Mean Time (ms) | 18.1505 |
| Median Time (ms) | 17.3670 |
| P20 Time (ms) | 17.1641 |
| P80 Time (ms) | 20.0782 |
| Min Time (ms) | 17.0609 |
| Max Time (ms) | 21.0268 |
| Std dev (ms) | 1.5225 |
| Samples | 15 |
