#include "gemm_common.cuh"
constexpr int TILE = 32;

__global__ void gemm_v2_smem_tiled(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    int ak = k0 + threadIdx.x;
    int bk = k0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[(size_t)row * K + ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[(size_t)bk * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
    __syncthreads();
  }
  if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(TILE, TILE); // 1024 threads: intentionally mirrors the teaching version.
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_v2_smem_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v2_smem_tiled"
#include "gemm_entrypoint.cuh"
