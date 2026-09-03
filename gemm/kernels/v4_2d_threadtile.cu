#include "gemm_common.cuh"
constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN); // 64

__global__ void gemm_v4_2d(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  int tid = threadIdx.x;
  int tc = tid % (BN / TN);
  int tr = tid / (BN / TN);
  int lr0 = tr * TM, lc0 = tc * TN;
  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += THREADS) {
      int r = idx / BK, k = idx % BK;
      int gr = blockIdx.y * BM + r, gk = k0 + k;
      As[r][k] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += THREADS) {
      int k = idx / BN, c = idx % BN;
      int gk = k0 + k, gc = blockIdx.x * BN + c;
      Bs[k][c] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float ar[TM], br[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ar[i] = As[lr0 + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) br[j] = Bs[k][lc0 + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
    }
    __syncthreads();
  }

  int gr0 = blockIdx.y * BM + lr0;
  int gc0 = blockIdx.x * BN + lc0;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = gr0 + i, c = gc0 + j;
      if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
    }
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v4_2d<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v4_2d_threadtile"
#include "gemm_entrypoint.cuh"
