# Kernel Profile Report: V3 scalar FP32 GEMM

## Target

| Field | Value |
|---|---|
| Runtime | Native CUDA |
| Executable | `./bin/v7_warptiling` |
| Arguments | `2048 2048 2048 8 5 --no-verify` |
| Kernel | `gemm_v7_warptiling` |
| Filter | `regex:.*gemm_v7_warptiling.*` |
| GPU | NVIDIA GeForce RTX 4070 Laptop, SM 8.9 |
| Profiler | Nsight Compute 2025.3.1 |
| Privilege | none |

Exact collection commands are recorded in `commands.sh`.

## Evidence

| Metric / instruction | Value | Interpretation |
|---|---:|---|
| Profiled duration | 1.982560 ms | NCU-instrumented kernel duration |
| Registers/thread | 80 | Compiler launch metadata |
| Static shared memory | 4352 bytes | Compiler launch metadata |
| FP32 FMA pipe | 46.17% elapsed peak | Main arithmetic path |
| HMMA instruction pipe | 0% | Tensor Core instructions were not executed |
| Tensor pipe active | 0% | Tensor Core pipeline was idle |
| SASS opcode | `FFMA` | Scalar FP32 fused multiply-add |

## Conclusion

High confidence: this kernel does not use Tensor Cores. The inner `fmaf` loop compiles to scalar `FFMA` instructions. No `HMMA` or `MMA` instruction was observed.

## Artifacts

- Basic metrics: `details/01_basic_raw.csv`
- Compute and instruction metrics: `details/04_compute_raw.csv`
- SASS source view: `details/07_source_raw.csv`
- Extracted hotspots: `details/source_hotspots.csv`

## Limitations and next action

- NCU changes execution conditions, so use ordinary CUDA-event runs for final timing.
- Only the Tensor-Core question was profiled; no broad bottleneck claim is made.
- A Tensor-Core version requires WMMA/MMA, CUTLASS, or an appropriate cuBLAS compute mode and datatype.
