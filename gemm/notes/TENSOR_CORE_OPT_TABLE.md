# Ada FP16 Tensor Core GEMM: four versions to remember

All rows compute row-major `C[M,N] = A[M,K] * B[K,N]` with FP16 inputs and
FP32 accumulation/output. Measurements below are from an RTX 4070 Laptop GPU
(Ada, SM 8.9), shape `M=512, N=6144, K=4096`, CUDA-event timing, L2-thrash
between trials, and correctness against a strict FP32 PyTorch reference.

| Stage | Problem being solved | Data path | Core code shape | Median / throughput | File | Hopper/Blackwell continuation |
|---|---|---|---|---:|---|---|
| T0 Naive WMMA | First make Tensor Core math correct | GMEM -> WMMA fragment -> GMEM | one warp computes one `16x16` C tile; operands repeatedly loaded from global memory | 8.8417 ms / 2.91 TFLOP/s | `extras/tensorcore_fp16_kernels.cuh` | keep as the portable teaching baseline; production code moves to newer async MMA APIs |
| T1 Handwritten Ada MMA | Reuse operands and overlap copies with math | GMEM -> `cp.async` -> permuted SMEM -> `ldmatrix` -> registers -> `mma.sync` | one CTA computes `128x128`; 8 warps; 3-stage circular pipeline; B is prepacked once | 1.0199 ms / 25.27 TFLOP/s, 8.67x T0 | `extras/tensorcore_fp16_kernels.cuh` | Hopper: TMA + `wgmma.mma_async` + producer/consumer warp specialization; Blackwell: add UMMA/TMEM through CUTLASS/CuTe first |
| T2 Shape-tuned CUTLASS | Finish iterator, edge, epilogue, and scheduling engineering | hierarchical TensorOp mainloop | CTA `64x128x32`; warp `32x64x32`; 3 stages | 0.6769 ms / 38.07 TFLOP/s, 13.06x T0 | `extras/tensorcore_fp16_cutlass.cuh` | use SM90/SM100 collective mainloops, TMA multicast, clusters, and persistent tile schedulers |
| cuBLAS | Vendor-tuned kernel family and runtime selection | proprietary library path | `cublasGemmEx`, FP16 inputs, FP32 compute/output | 0.6994 ms / 36.85 TFLOP/s | `extras/tensorcore_fp16_kernels.cuh` | recent cuBLAS versions select architecture-specific paths automatically |

The CUTLASS result is 3.3% faster than cuBLAS only for this exact fixed shape
and cold-L2 protocol. A separate warm-cache executable run measured cuBLAS at
0.601 ms and CUTLASS at 0.705--0.795 ms. This is why a serious claim must name
the shape, cache policy, clock/power state, data types, epilogue, and timing
scope.

## What changed from T0 to T1/T2

1. Block/warp tiling makes many output values reuse each A and B tile.
2. 128-bit cooperative copies reduce global-memory instructions.
3. `cp.async` moves the next K tile while Tensor Cores compute the current one.
4. A permuted shared-memory layout avoids conflicts for warp matrix loads.
5. `ldmatrix` collectively moves the exact register layout consumed by
   `mma.sync.m16n8k16`.
6. Several K stages stay in flight, while accumulator fragments remain in
   lane-private registers until the epilogue.

The central lesson is not "maximize occupancy". T2 uses 140 registers/thread
and about 37 KiB shared memory/CTA, so occupancy falls to about 16%; nevertheless
it is much faster because each resident warp does useful Tensor Core work instead
of waiting on global loads.

## Build and run

```bash
cd gemm
make ARCH=sm_89 CUTLASS_DIR=/home/l1q/WSL/CUTLASS -j

./bin/tensorcore_fp16_lab naive 512 6144 4096 50 10
./bin/tensorcore_fp16_lab ada 512 6144 4096 50 10
./bin/tensorcore_fp16_cutlass_lab 64x128 512 6144 4096 50 10
./bin/tensorcore_fp16_lab cublas 512 6144 4096 50 10
```

The direct executable is a convenient smoke test. Use the generated benchmark
reports under `bench_skill/results/` for the controlled cold-L2 comparison.

## Compare the two NCU reports

Open these in the same Nsight Compute instance:

- `profile/tensor_fp16_naive_gui/tensor_fp16_naive_detailed.ncu-rep`
- `profile/tensor_fp16_cutlass_gui/tensor_fp16_cutlass_detailed.ncu-rep`

Then:

1. Select the naive report, open **Details**, and choose **Compare -> Add
   Baseline**.
2. Select the CUTLASS report as **Current**.
3. Start with **GPU Speed of Light** and compare duration and throughput.
4. In **Compute Workload Analysis**, compare Tensor/HMMA pipe activity.
5. In **Warp State Statistics**, inspect Long Scoreboard, MIO Throttle, and
   Math Pipe Throttle.
6. In **Launch Statistics / Occupancy**, inspect registers, shared memory, and
   achieved occupancy.
7. In **Source**, switch to SASS and find `HMMA`, `LDGSTS` (`cp.async`), and
   `LDSM` (`ldmatrix`).

To regenerate both captures from the repository root:

```bash
bash profile/tensor_fp16_naive_gui/commands.sh
bash profile/tensor_fp16_cutlass_gui/commands.sh
```

Both captures use `--launch-skip 5 --launch-count 1`, the same shape, and exact
kernel filters. NCU replay affects elapsed time, so use NCU primarily to explain
why the independent CUDA-event benchmark moved.
