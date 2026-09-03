# Transformer-like GEMM benchmark

Date: 2026-09-03  
GPU: NVIDIA GeForce RTX 4070 Laptop, SM 8.9  
CUDA toolkit: 12.6  
Precision: FP32 inputs and accumulation  
Custom kernel: `v7_warptiling` (the proposed four-step lab's V3)

## Model-shaped workload

The workload is an 8B-class GQA-like decoder configuration:

- hidden size `H = 4096`
- intermediate size `I = 14336`
- query heads `32`
- KV heads `8`
- head dimension `128`
- fused QKV width `H + 2 * KV_heads * head_dim = 6144`

For linear layers, batch and sequence dimensions flatten exactly into GEMM
`M = B * S`. The prefill case uses `B=4, S=128, M=512`; the decode case
uses `B=16, S_new=1, M=16`.

Each result below is the mean of two independent processes after in-process
warmup. Every run passed the lab's sampled correctness check. Timing uses CUDA
events around repeated kernel launches.

## Results

| Phase | Operation | M | N | K | V3 ms | cuBLAS ms | V3 / cuBLAS throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| Prefill | Attention QKV projection | 512 | 6144 | 4096 | 3.0973 | 2.3367 | 75.4% |
| Prefill | Attention output projection | 512 | 4096 | 4096 | 2.2131 | 1.5253 | 68.9% |
| Prefill | MLP fused gate + up | 512 | 28672 | 4096 | 19.6600 | 10.6743 | 54.3% |
| Prefill | MLP down | 512 | 4096 | 14336 | 7.2119 | 5.1343 | 71.2% |
| Decode | Attention QKV projection | 16 | 6144 | 4096 | 0.5782 | 0.4993 | 86.4% |
| Decode | Attention output projection | 16 | 4096 | 4096 | 0.5223 | 0.3482 | 66.7% |
| Decode | MLP fused gate + up | 16 | 28672 | 4096 | 2.4368 | 1.8403 | 75.5% |
| Decode | MLP down | 16 | 4096 | 14336 | 1.6897 | 1.1171 | 66.1% |

Linear-GEMM totals per layer:

| Phase | V3 attention projections | V3 MLP | cuBLAS attention projections | cuBLAS MLP |
|---|---:|---:|---:|---:|
| Prefill | 5.3104 ms | 26.8719 ms | 3.8620 ms | 15.8086 ms |
| Decode | 1.1005 ms | 4.1265 ms | 0.8475 ms | 2.9574 ms |

For this GQA-like configuration, MLP linear layers contain about 4.2x the
FLOPs of the attention projection linear layers. The measured MLP/attention
projection time ratio is 5.1x for V3 and 4.1x for cuBLAS during prefill.

## Square-size scaling

| Shape | V3 median TFLOP/s | cuBLAS median TFLOP/s | V3 / cuBLAS |
|---|---:|---:|---:|
| 4096 cubed | 9.003 | 12.300 | 73.2% |
| 6144 cubed | 8.884 | 11.368 | 78.1% |
| 8192 cubed | 8.589 | 14.255 | 60.3% |

Larger square matrices do not force convergence. V3 plateaus around 8.6-9.0
TFLOP/s while cuBLAS changes kernel scheduling and reaches a higher plateau.

## Tensor Core check

- V3: scalar `FFMA`; Tensor pipeline 0%.
- Unmodified `cublasSgemm`: `ampere_sgemm_128x64_nn`, scalar `FFMA`;
  Tensor pipeline 0%.
- Explicit `CUBLAS_TF32_TENSOR_OP_MATH`: `HMMA.1688.F32.TF32`; Tensor
  pipeline active 42.12%; 2048-cubed ordinary timing 15.469 TFLOP/s.

See the three reports under `profile/*tensorcore_check/final_report.md`.

## Scope and limitations

- Attention QK-transpose, softmax, and P-times-V are not included. They are
  strided-batched/head-batched operations and the current kernel ABI is only
  one 2-D GEMM.
- A realistic attention benchmark should add batch/head strides or compare
  against fused attention. Repeatedly launching one GEMM per head would
  overstate launch overhead and materialize the score matrix.
- Decode has a very small M dimension and is closer to GEMV/skinny-GEMM than
  the square matrices used in the original lab.
- The current harness repeats the same buffers and reports an aggregate mean;
  a final report should add per-trial percentiles and explicit cache policy.
