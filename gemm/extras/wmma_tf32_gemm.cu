#include "gemm_common.cuh"
#include <mma.h>

using namespace nvcuda;

// A deliberately small, readable Tensor Core baseline.  One warp computes one
// 16x16 output tile with TF32 inputs and FP32 accumulation.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

__global__ void gemm_wmma_tf32_aligned(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
  int tile_row = blockIdx.y * WMMA_M;
  int tile_col = blockIdx.x * WMMA_N;
  int lane = threadIdx.x;

  __shared__ __align__(32) float As[WMMA_M][WMMA_K];
  __shared__ __align__(32) float Bs[WMMA_K][WMMA_N];

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
#pragma unroll
    for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32) {
      int r = idx / WMMA_K;
      int k = idx % WMMA_K;
      As[r][k] = wmma::__float_to_tf32(A[(size_t)(tile_row + r) * K + k0 + k]);
    }
#pragma unroll
    for (int idx = lane; idx < WMMA_K * WMMA_N; idx += 32) {
      int k = idx / WMMA_N;
      int c = idx % WMMA_N;
      Bs[k][c] = wmma::__float_to_tf32(B[(size_t)(k0 + k) * N + tile_col + c]);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::load_matrix_sync(a_frag, &As[0][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &Bs[0][0], WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(&C[(size_t)tile_row * N + tile_col], c_frag, N,
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
  if ((M % WMMA_M) == 0 && (N % WMMA_N) == 0 && (K % WMMA_K) == 0) {
    dim3 block(32);
    dim3 grid(N / WMMA_N, M / WMMA_M);
    gemm_wmma_tf32_aligned<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  } else {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_scalar_fallback<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  }
}

#define KERNEL_NAME "wmma_tf32_gemm"
#include "gemm_entrypoint.cuh"
