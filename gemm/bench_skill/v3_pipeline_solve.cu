#include "../extras/v3_cp_async_double_buffered.cu"

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
  launch_gemm(A, B, C, M, N, K, 0);
}
