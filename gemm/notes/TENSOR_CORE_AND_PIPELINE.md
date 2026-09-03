# Tensor Core and Pipeline Experiments

This note keeps the experimental paths separate from the four-stage CUDA-core
teaching ladder.  All timings below use the SGLang-style prefill QKV projection
`[512,4096] x [4096,6144]`, CUDA events, five warmups, ten measured trials, and
an L2-thrashing cold-cache policy on an RTX 4070 Laptop GPU (SM89).

## `cp.async` double buffering

`extras/v3_cp_async_double_buffered.cu` keeps V3's `BM=64`, `BN=64`, `BK=8`,
four-warp hierarchy, and `4x8` per-thread accumulator.  It changes only time:

1. stage 0 asynchronously loads K tile 0;
2. while stage `s` is consumed, stage `s^1` receives the next K tile;
3. `cp.async.wait_group 0` and a block barrier protect the stage swap.

Each thread issues one 16-byte A copy and one 16-byte B copy.  A uses an XOR
half-row layout: `physical_k = k ^ ((row & 4) ? 4 : 0)`.  This keeps each
16-byte destination aligned while breaking the stride-8 shared-bank pattern,
so no padding column is needed.

The result was **3.9875 ms median**, versus **2.9788 ms** for V3: 0.75x, not a
speedup.  The kernel is correct and SASS contains `LDGSTS` (the Ada lowering of
the asynchronous global-to-shared copy), but overlap alone does not guarantee a
win.  This version adds stage-addressing/control work and keeps a relatively
small `BK=8`; the original kernel was not sufficiently latency-bound to repay
that overhead.

## Handwritten WMMA TF32

`extras/wmma_tf32_gemm.cu` is the smallest readable form: one warp computes one
16x16 output tile.  It explicitly constructs A/B/accumulator fragments, calls
`load_matrix_sync`, `mma_sync`, and `store_matrix_sync`, and converts FP32 input
to TF32 before MMA.  SASS contains `HMMA.1684.F32.TF32`, proving Tensor Core
execution.  It is intentionally inefficient because every warp reloads its own
A/B tiles and synchronizes every K=8 step.

`extras/wmma_tf32_block_tiled.cu` groups eight warps into a 64x32 block tile.
A is shared across two warp columns and B across four warp rows.  This improves
the one-warp version from **9.9671 ms** to **6.5664 ms** median (1.52x), but it
is still slower than V3 and cuBLAS.  An attempted `BLOCK_K=32` version reduced
barrier frequency but increased shared memory from 3 KB to 12 KB and registers
from 40 to 72; it regressed to **6.6688 ms**, so the source keeps `BK=8`.

## Same-shape comparison

| Kernel | Arithmetic | Median ms | Approx. TFLOP/s |
|---|---|---:|---:|
| cuBLAS explicit TF32 | Tensor Core | 1.4136 | 18.23 |
| cuBLAS SGEMM | default FP32 math policy | 2.0787 | 12.40 |
| V3 warp tiled | FP32 CUDA core | 2.9788 | 8.65 |
| V3 + `cp.async` double buffer | FP32 CUDA core | 3.9875 | 6.46 |
| WMMA block tiled | TF32 Tensor Core | 6.5664 | 3.92 |
| WMMA one warp/tile | TF32 Tensor Core | 9.9671 | 2.59 |

The lesson is not that Tensor Cores or asynchronous copies are slow.  It is
that a fast arithmetic instruction needs a matching data-movement pipeline,
large enough tiles, low-overhead layout conversion, sensible warp scheduling,
and shape-specific tuning.  cuBLAS/CUTLASS supply that surrounding machinery.
