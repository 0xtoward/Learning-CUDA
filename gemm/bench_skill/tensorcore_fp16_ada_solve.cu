#include "../extras/tensorcore_fp16_kernels.cuh"

extern "C" void solve(const half* A, const half* B, float* C, int M, int N,
                      int K) {
  tensorcore_fp16::launch_ada(A, B, C, M, N, K, 0);
}
