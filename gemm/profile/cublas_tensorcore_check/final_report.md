# Kernel Profile Report: default cuBLAS SGEMM

## Target

| Field | Value |
|---|---|
| Runtime | Native CUDA / cuBLAS |
| Executable | `./bin/cublas_baseline` |
| Arguments | `2048 2048 2048 8 5 --no-verify` |
| Resolved kernel | `ampere_sgemm_128x64_nn` |
| Filter | `regex:.*gemm.*` |
| GPU | NVIDIA GeForce RTX 4070 Laptop, SM 8.9 |
| Profiler | Nsight Compute 2025.3.1 |
| Privilege | none |

Exact collection commands are recorded in `commands.sh`.

## Evidence

| Metric / instruction | Value | Interpretation |
|---|---:|---|
| Profiled duration | 1.460064 ms | NCU-instrumented kernel duration |
| Registers/thread | 122 | Vendor-kernel launch metadata |
| Static shared memory | 12544 bytes | Vendor-kernel launch metadata |
| FP32 FMA pipe | 62.34% elapsed peak | Main arithmetic path |
| HMMA instruction pipe | 0% | Tensor Core instructions were not executed |
| Tensor pipe active | 0% | Tensor Core pipeline was idle |
| SASS opcode | `FFMA` | Scalar FP32 fused multiply-add |

## Conclusion

High confidence: the unmodified lab's `cublasSgemm` call did not use Tensor Cores for this shape and environment. cuBLAS selected `ampere_sgemm_128x64_nn`, which executed scalar FP32 `FFMA`.

This conclusion applies to this call configuration, not to all cuBLAS GEMMs. Explicit TF32, FP16, BF16, FP8, or other compute-type configurations can select Tensor-Core kernels.

## Artifacts

- Basic metrics: `details/01_basic_raw.csv`
- Compute and instruction metrics: `details/04_compute_raw.csv`
- SASS source view: `details/07_source_raw.csv`
- Extracted hotspots: `details/source_hotspots.csv`

## Limitations and next action

- NCU changes execution conditions, so use ordinary CUDA-event runs for final timing.
- Only the Tensor-Core question was profiled; no broad bottleneck claim is made.
- Compare against the explicit TF32 experiment before attributing the speed gap to Tensor Cores.
