#include "gemm_common.cuh"
constexpr int BM = 64, BN = 32, BK = 8, TM = 8;
constexpr int THREADS = (BM / TM) * BN; // 256

__global__ void gemm_v3_1d(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  int tid = threadIdx.x;
  int local_col = tid % BN;
  int row_group = tid / BN;
  int row_base = blockIdx.y * BM + row_group * TM;
  int col = blockIdx.x * BN + local_col;
  float acc[TM] = {0.0f};

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
      float b = Bs[k][local_col];
#pragma unroll
      for (int i = 0; i < TM; ++i) acc[i] = fmaf(As[row_group * TM + i][k], b, acc[i]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = row_base + i;
    if (r < M && col < N) C[(size_t)r * N + col] = acc[i];
  }
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v3_1d<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v3_1d_threadtile"
#include "gemm_entrypoint.cuh"
