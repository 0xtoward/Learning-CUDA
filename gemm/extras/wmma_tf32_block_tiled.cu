#include "gemm_common.cuh"
#include <mma.h>

using namespace nvcuda;

// A second WMMA version: eight warps cooperate on a 64x32 block tile.  The A
// tile is reused by two warp columns and the B tile by four warp rows.
constexpr int BM = 64, BN = 32, BK = 8;
constexpr int WM = 16, WN = 16;
constexpr int WARPS_M = BM / WM;
constexpr int WARPS_N = BN / WN;
constexpr int THREADS = WARPS_M * WARPS_N * 32;

__global__ void gemm_wmma_tf32_block_tiled_aligned(
    const float* A, const float* B, float* C, int M, int N, int K) {
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int warp_m = warp / WARPS_N;
  int warp_n = warp % WARPS_N;
  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;

  __shared__ __align__(32) float As[BM][BK];
  __shared__ __align__(32) float Bs[BK][BN];

  wmma::fragment<wmma::accumulator, WM, WN, BK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int idx = tid; idx < BM * BK; idx += THREADS) {
      int r = idx / BK;
      int k = idx % BK;
      As[r][k] = wmma::__float_to_tf32(
          A[(size_t)(block_row + r) * K + k0 + k]);
    }
    int bk = tid / BN;
    int c = tid % BN;
    Bs[bk][c] = wmma::__float_to_tf32(
        B[(size_t)(k0 + bk) * N + block_col + c]);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WM, WN, BK,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, BK,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::load_matrix_sync(a_frag, &As[warp_m * WM][0], BK);
    wmma::load_matrix_sync(b_frag, &Bs[0][warp_n * WN], BN);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  int out_row = block_row + warp_m * WM;
  int out_col = block_col + warp_n * WN;
  wmma::store_matrix_sync(&C[(size_t)out_row * N + out_col], c_frag, N,
                          wmma::mem_row_major);
}

__global__ void gemm_scalar_fallback(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= M || c >= N) return;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k)
    sum = fmaf(A[(size_t)r * K + k], B[(size_t)k * N + c], sum);
  C[(size_t)r * N + c] = sum;
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  if ((M % BM) == 0 && (N % BN) == 0 && (K % BK) == 0) {
    dim3 block(THREADS);
    dim3 grid(N / BN, M / BM);
    gemm_wmma_tf32_block_tiled_aligned<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K);
  } else {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_scalar_fallback<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  }
}

#define KERNEL_NAME "wmma_tf32_block_tiled"
#include "gemm_entrypoint.cuh"
