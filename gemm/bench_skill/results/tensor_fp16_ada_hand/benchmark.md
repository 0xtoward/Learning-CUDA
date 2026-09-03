# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `tensorcore_fp16_ada_solve.cu` |
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
| Mean Time (ms) | 1.1361 |
| Median Time (ms) | 1.0199 |
| P20 Time (ms) | 1.0189 |
| P80 Time (ms) | 1.2634 |
| Min Time (ms) | 0.9892 |
| Max Time (ms) | 1.8616 |
| Std dev (ms) | 0.2375 |
| Samples | 50 |
