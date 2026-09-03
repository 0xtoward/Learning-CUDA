# Three GEMM examples to read in order

These examples separate three ideas that are easy to mix together.

1. `cutlass_minimal.cu`: CUTLASS as a host-side C++ template library. The
   example is FP32 and deliberately selects the default SIMT/CUDA-Core path.
2. `wmma_minimal.cu`: the smallest readable Tensor Core kernel. One warp owns
   one `16x16` output tile and calls the warp-synchronous WMMA API.
3. `cuda_core_gemm_annotated.cu`: a readable optimized FP32 CUDA-Core GEMM with
   CTA/warp/thread tiling, SMEM reuse, register outer products, padding, and
   vectorized loads/stores.

## Build and run

From `gemm/`:

```bash
make teaching

./bin/cutlass_minimal 512 512 512 20 5
./bin/wmma_minimal 512 512 512 20 5
./bin/cuda_core_gemm_annotated 512 512 512 20 5
```

The argument order is `M N K iterations warmup [--no-verify]`.

Local smoke test on the RTX 4070 Laptop at `512x512x512`:

| Example | Correctness | Average time | SASS evidence |
|---|---|---:|---|
| CUTLASS minimal FP32 SIMT | passed | 0.0513 ms | `FFMA`, zero `HMMA` |
| WMMA minimal FP16 | passed | 0.0358 ms | `HMMA.16816.F32` |
| Annotated CUDA Core | passed | 0.0558 ms | `FFMA`, 80 registers, zero spills |
| Same kernel with inline PTX FMA | passed | 0.0557 ms | also `FFMA`, 80 registers, zero spills |

These small-shape numbers are a compile/correctness sanity check, not a claim
that one abstraction always wins. Use fixed larger shapes and distributions for
a performance conclusion.

The CUDA-Core example also has an educational inline-PTX build:

```bash
./bin/cuda_core_gemm_annotated_ptx 512 512 512 20 5
```

It replaces `fmaf()` with `fma.rn.f32`. Both should normally become an FFMA
instruction in SASS; the PTX version is not advertised as an optimization.
Confirm the compiler output instead of assuming:

```bash
cuobjdump --dump-sass ./bin/cuda_core_gemm_annotated | grep -m 5 FFMA
cuobjdump --dump-sass ./bin/cuda_core_gemm_annotated_ptx | grep -m 5 FFMA
```

For the actually useful inline-PTX case, see
`extras/tensorcore_fp16_kernels.cuh`: CUDA C++ does not provide the same direct
control over the exact `cp.async`, `ldmatrix`, and `mma.sync` instruction/data
layout sequence, so the optimized Ada teaching kernel spells those operations
out explicitly.

## What to say in an interview

- CUTLASS is an open-source, header-heavy CUDA C++ template toolkit. It exposes
  reusable device-, CTA-, warp-, and instruction-level GEMM components.
- CUTLASS is not synonymous with Tensor Cores: `OpClassSimt` targets ordinary
  CUDA-Core math, while `OpClassTensorOp` targets Tensor Core MMA.
- WMMA is a convenient warp-level Tensor Core API. It proves functionality but
  does not automatically provide CTA tiling, SMEM staging, pipelining, or a
  tuned epilogue.
- CUDA-Core GEMM normally does not need handwritten PTX. Prefer CUDA C++,
  intrinsics such as `__shfl_sync`, vector types such as `float4`, and then
  inspect generated PTX/SASS. Use inline PTX only when an exact instruction or
  operand layout is unavailable or insufficiently controllable from C++.
