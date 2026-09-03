#include <cub/device/device_reduce.cuh>

#include "reduce_bench_common.cuh"

extern "C" void solve(const float* x, float* out, int n) {
  static void* temporary = nullptr;
  static size_t temporary_bytes = 0;
  static int capacity = 0;
  if (n > capacity) {
    if (temporary) REDUCE_CUDA_CHECK(cudaFree(temporary));
    temporary_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temporary_bytes, x, out, n);
    REDUCE_CUDA_CHECK(cudaMalloc(&temporary, temporary_bytes));
    capacity = n;
  }
  cub::DeviceReduce::Sum(temporary, temporary_bytes, x, out, n);
}
