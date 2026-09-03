#include "gemm_common.cuh"
constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;
constexpr int THREADS = 64;

__global__ void gemm_v6_vec(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  int tid = threadIdx.x;
  int tc = tid % 8, tr = tid / 8;
  int lr0 = tr * TM, lc0 = tc * TN;
  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 128 float4s in each tile, 64 threads -> 2 vector loads per thread.
    for (int v = tid; v < BM * (BK / 4); v += THREADS) {
      int r = v / (BK / 4), vc = v % (BK / 4);
      int gr = blockIdx.y * BM + r, gk = k0 + vc * 4;
      if (gr < M && gk + 3 < K && (K % 4 == 0)) {
        *reinterpret_cast<float4*>(&As[r][vc * 4]) = *reinterpret_cast<const float4*>(&A[(size_t)gr * K + gk]);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) As[r][vc * 4 + q] = (gr < M && gk + q < K) ? A[(size_t)gr * K + gk + q] : 0.0f;
      }
    }
    for (int v = tid; v < BK * (BN / 4); v += THREADS) {
      int k = v / (BN / 4), vc = v % (BN / 4);
      int gk = k0 + k, gc = blockIdx.x * BN + vc * 4;
      if (gk < K && gc + 3 < N && (N % 4 == 0)) {
        *reinterpret_cast<float4*>(&Bs[k][vc * 4]) = *reinterpret_cast<const float4*>(&B[(size_t)gk * N + gc]);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) Bs[k][vc * 4 + q] = (gk < K && gc + q < N) ? B[(size_t)gk * N + gc + q] : 0.0f;
      }
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

  int gr0 = blockIdx.y * BM + lr0, gc0 = blockIdx.x * BN + lc0;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = gr0 + i;
    if (r >= M) continue;
    if (gc0 + TN <= N && (N % 4 == 0)) {
      float4 v0{acc[i][0],acc[i][1],acc[i][2],acc[i][3]};
      float4 v1{acc[i][4],acc[i][5],acc[i][6],acc[i][7]};
      *reinterpret_cast<float4*>(&C[(size_t)r * N + gc0]) = v0;
      *reinterpret_cast<float4*>(&C[(size_t)r * N + gc0 + 4]) = v1;
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j) if (gc0 + j < N) C[(size_t)r * N + gc0 + j] = acc[i][j];
    }
  }
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v6_vec<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v6_2d_vectorized"
#include "gemm_entrypoint.cuh"
