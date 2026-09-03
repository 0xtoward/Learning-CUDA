# GEMM optimization ladder

| Stage | Core idea | Memory hierarchy target | What to look for in NCU | Source |
|---|---|---|---|---|
| V0 Naive bad mapping | One thread -> one C, intentionally bad lane mapping | GMEM | Uncoalesced Global Accesses, Lg Throttle | `kernels/v0_naive_bad_mapping.cu` |
| V1 Coalesced | Map lane-x to adjacent N columns | GMEM | fewer excessive L2 sectors, higher useful bandwidth | `kernels/v1_coalesced.cu` |
| V2 SMEM tiled | Stage A/B K-tiles in shared memory | GMEM -> SMEM | less repeated GMEM traffic; watch MIO/Barrier | `kernels/v2_smem_tiled.cu` |
| V3 1D thread tile | One thread computes multiple C values along M | SMEM -> registers | MIO throttle down, register reuse up | `kernels/v3_1d_threadtile.cu` |
| V4 2D thread tile | Per-thread C micro-tile, register outer product | registers | more ILP/reuse; register pressure starts to matter | `kernels/v4_2d_threadtile.cu` |
| V5 XOR swizzle | Remap shared-memory addresses | SMEM banks | L1 Wavefronts Shared Excessive / bank conflicts | `kernels/v5_2d_xor_swizzle.cu` |
| V6 Vectorized | `float4` GMEM loads/stores | GMEM transactions | fewer load/store instructions; useful on larger problems | `kernels/v6_2d_vectorized.cu` |
| V7 Warp tiling | block -> warp -> thread hierarchy; padded A tile | SMEM + registers + scheduling | SOL, occupancy, register count, remaining stalls | `kernels/v7_warptiling.cu` |

The exact performance ordering depends on GPU, compiler, shape, clocks, and tile constants. These files are educational templates, not replacements for cuBLAS/CUTLASS.
