# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `tensorcore_fp16_naive_solve.cu` |
| **Reference** | `hgemm_ref.py` |
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
| Mean Time (ms) | 8.8219 |
| Median Time (ms) | 8.8417 |
| P20 Time (ms) | 2.9757 |
| P80 Time (ms) | 13.8324 |
| Min Time (ms) | 2.9307 |
| Max Time (ms) | 15.6334 |
| Std dev (ms) | 4.5848 |
| Samples | 50 |
