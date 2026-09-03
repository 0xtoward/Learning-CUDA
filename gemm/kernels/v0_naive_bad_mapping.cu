#include "gemm_common.cuh"

__global__ void gemm_v0_bad_mapping(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
  // Intentionally bad for row-major matrices: x dimension maps to row.
  // A and C addresses become far apart across a warp -> uncoalesced accesses.
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc = fmaf(A[(size_t)row * K + k], B[(size_t)k * N + col], acc);
  C[(size_t)row * N + col] = acc;
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  gemm_v0_bad_mapping<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v0_naive_bad_mapping"
#include "gemm_entrypoint.cuh"
