# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `tensorcore_fp16_cutlass_s4_solve.cu` |
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
| Mean Time (ms) | 0.7182 |
| Median Time (ms) | 0.6830 |
| P20 Time (ms) | 0.6799 |
| P80 Time (ms) | 0.6840 |
| Min Time (ms) | 0.6779 |
| Max Time (ms) | 1.3128 |
| Std dev (ms) | 0.1434 |
| Samples | 50 |
