#include "gemm_common.cuh"

__global__ void gemm_v1_coalesced(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
  // x dimension maps to adjacent columns. Across a warp:
  // A[row,k] is broadcast-like; B[k,col] and C[row,col] are contiguous.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc = fmaf(A[(size_t)row * K + k], B[(size_t)k * N + col], acc);
  C[(size_t)row * N + col] = acc;
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  gemm_v1_coalesced<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v1_coalesced"
#include "gemm_entrypoint.cuh"
