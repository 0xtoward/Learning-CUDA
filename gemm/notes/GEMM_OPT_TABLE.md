# GEMM optimization ladder

| Stage | Core idea | Memory hierarchy target | What to look for in NCU | Source | Hopper/Blackwell continuation |
|---|---|---|---|---|---|
| V0 Naive bad mapping | One thread -> one C, intentionally bad lane mapping | GMEM | Uncoalesced Global Accesses, Lg Throttle | `kernels/v0_naive_bad_mapping.cu` | mapping/coalescing remains mandatory on every architecture |
| V1 Coalesced | Map lane-x to adjacent N columns | GMEM | fewer excessive L2 sectors, higher useful bandwidth | `kernels/v1_coalesced.cu` | vectorized/bulk transactions still need contiguous tiles |
| V2 SMEM tiled | Stage A/B K-tiles in shared memory | GMEM -> SMEM | less repeated GMEM traffic; watch MIO/Barrier | `kernels/v2_smem_tiled.cu` | TMA bulk/tensor copies can replace per-thread copies; clusters may multicast tiles |
| V3 1D thread tile | One thread computes multiple C values along M | SMEM -> registers | MIO throttle down, register reuse up | `kernels/v3_1d_threadtile.cu` | the reuse principle survives; Tensor Core paths distribute fragments across a warp/warpgroup |
| V4 2D thread tile | Per-thread C micro-tile, register outer product | registers | more ILP/reuse; register pressure starts to matter | `kernels/v4_2d_threadtile.cu` | WGMMA/UMMA changes the instruction, not the need to balance reuse and register pressure |
| V5 XOR swizzle | Remap shared-memory addresses | SMEM banks | L1 Wavefronts Shared Excessive / bank conflicts | `kernels/v5_2d_xor_swizzle.cu` | use layouts compatible with TMA descriptors and WGMMA/UMMA operand rules |
| V6 Vectorized | `float4` GMEM loads/stores | GMEM transactions | fewer load/store instructions; useful on larger problems | `kernels/v6_2d_vectorized.cu` | TMA can issue multidimensional asynchronous bulk copies instead of many lane loads |
| V7 Warp tiling | block -> warp -> thread hierarchy; padded A tile | SMEM + registers + scheduling | SOL, occupancy, register count, remaining stalls | `kernels/v7_warptiling.cu` | TMA + WGMMA/UMMA, warp specialization, persistent scheduling, and thread-block clusters |
| cuBLAS | shape- and architecture-selected production kernels | full hierarchy | compare duration, math pipe, stalls, and accuracy under the same protocol | `extras/cublas_baseline.cu` | newer releases select architecture-specific kernels automatically |

The exact performance ordering depends on GPU, compiler, shape, clocks, and tile constants. These files are educational templates, not replacements for cuBLAS/CUTLASS.
