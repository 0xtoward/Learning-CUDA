#include "../kernels/v1_coalesced.cu"

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  launch_gemm(A, B, C, M, N, K, 0);
}

