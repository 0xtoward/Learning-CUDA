#include "../extras/tensorcore_fp16_cutlass.cuh"

extern "C" void solve(const half* A, const half* B, float* C, int M, int N,
                      int K) {
  tensorcore_fp16_cutlass::launch_64x128(A, B, C, M, N, K, 0);
}
