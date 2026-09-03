# Kernel Profile Report: explicit TF32 cuBLAS SGEMM

## Target

| Field | Value |
|---|---|
| Runtime | Native CUDA / cuBLAS |
| Executable | `./bin/cublas_tf32_baseline` |
| Arguments | `2048 2048 2048 8 5 --no-verify` |
| Resolved kernel | CUTLASS-style `s1688gemm` TensorOp kernel |
| Filter | `regex:.*gemm.*` |
| GPU | NVIDIA GeForce RTX 4070 Laptop, SM 8.9 |
| Profiler | Nsight Compute 2025.3.1 |
| Privilege | none |

Exact collection commands are recorded in `commands.sh`.

## Evidence

| Metric / instruction | Value | Interpretation |
|---|---:|---|
| Profiled duration | 1.069280 ms | NCU-instrumented kernel duration |
| Registers/thread | 226 | Vendor-kernel launch metadata |
| FP32 FMA pipe | 3.09% elapsed peak | Minor scalar work |
| HMMA instruction pipe | 21.06% elapsed peak | Tensor Core instructions executed |
| Tensor pipe active | 42.12% elapsed peak | Tensor Core pipeline active |
| SASS opcode | `HMMA.1688.F32.TF32` | TF32 Tensor Core matrix multiply |
| Ordinary event benchmark | 1.110630 ms, 15.469 TFLOP/s | 2048-cube run |
| FP32-tolerance check | Failed at odd shape 257x259x263 | 0.003154 absolute error exceeded 0.002 tolerance |

## Conclusion

High confidence: explicitly setting `CUBLAS_TF32_TENSOR_OP_MATH` selected a Tensor-Core path. The SASS contains `HMMA.1688.F32.TF32`, in direct contrast with the scalar `FFMA` path used by the unmodified baseline.

## Artifacts

- Basic metrics: `details/01_basic_raw.csv`
- Compute and instruction metrics: `details/04_compute_raw.csv`
- SASS source view: `details/07_source_raw.csv`
- Extracted hotspots: `details/source_hotspots.csv`

## Limitations and next action

- TF32 reduces input mantissa precision relative to full FP32. The lab's odd-shape
  FP32 tolerance check failed, so TF32 must use explicitly justified tolerances
  and error statistics rather than being treated as numerically equivalent.
- NCU timing is not the final benchmark timing.
- Add a paired strict-FP32/TF32 option to the lab rather than silently changing the baseline.
