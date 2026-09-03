#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

struct HgemmBenchConfig {
  int M = 512;
  int N = 6144;
  int K = 4096;
  int iters = 30;
  int warmup = 10;
  bool verify = true;
};

inline HgemmBenchConfig parse_hgemm_args(int argc, char** argv, int offset) {
  HgemmBenchConfig cfg;
  if (argc > offset + 0) cfg.M = std::atoi(argv[offset + 0]);
  if (argc > offset + 1) cfg.N = std::atoi(argv[offset + 1]);
  if (argc > offset + 2) cfg.K = std::atoi(argv[offset + 2]);
  if (argc > offset + 3) cfg.iters = std::atoi(argv[offset + 3]);
  if (argc > offset + 4) cfg.warmup = std::atoi(argv[offset + 4]);
  if (argc > offset + 5 && std::string(argv[offset + 5]) == "--no-verify") {
    cfg.verify = false;
  }
  return cfg;
}

inline void fill_random_half(std::vector<half>& values, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (half& value : values) value = __float2half(distribution(generator));
}

inline bool verify_hgemm_samples(const std::vector<half>& A,
                                 const std::vector<half>& B,
                                 const std::vector<float>& C, int M, int N,
                                 int K) {
  constexpr int samples = 32;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    int row = (sample * 131 + 7) % M;
    int col = (sample * 197 + 11) % N;
    double reference = 0.0;
    for (int k = 0; k < K; ++k) {
      reference += static_cast<double>(__half2float(A[(size_t)row * K + k])) *
                   static_cast<double>(__half2float(B[(size_t)k * N + col]));
    }
    double got = C[(size_t)row * N + col];
    double abs_error = std::abs(got - reference);
    double rel_error = abs_error / std::max(1.0, std::abs(reference));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    double tolerance = 2.5e-2 + 3.0e-3 * std::abs(reference);
    if (!std::isfinite(got) || abs_error > tolerance) {
      std::fprintf(stderr,
                   "verify failed at (%d,%d): got=%g ref=%g abs=%g tol=%g\n",
                   row, col, got, reference, abs_error, tolerance);
      return false;
    }
  }
  std::printf("sample_max_abs_error: %.8g\n", max_abs_error);
  std::printf("sample_max_rel_error: %.8g\n", max_rel_error);
  return true;
}

using HgemmLaunchFn = void (*)(const half*, const half*, float*, int, int, int,
                               cudaStream_t);

inline int run_hgemm_benchmark(const char* kernel_name, HgemmLaunchFn launch,
                               const HgemmBenchConfig& cfg) {
  const size_t a_elements = (size_t)cfg.M * cfg.K;
  const size_t b_elements = (size_t)cfg.K * cfg.N;
  const size_t c_elements = (size_t)cfg.M * cfg.N;

  std::vector<half> host_a(a_elements), host_b(b_elements);
  std::vector<float> host_c(c_elements);
  fill_random_half(host_a, 1);
  fill_random_half(host_b, 2);

  half *device_a = nullptr, *device_b = nullptr;
  float* device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, a_elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_b, b_elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_c, c_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_elements * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_elements * sizeof(half),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  for (int i = 0; i < cfg.warmup; ++i) {
    launch(device_a, device_b, device_c, cfg.M, cfg.N, cfg.K, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < cfg.iters; ++i) {
    launch(device_a, device_b, device_c, cfg.M, cfg.N, cfg.K, stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float average_ms = total_ms / cfg.iters;
  double tflops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K /
                  (average_ms * 1.0e9);

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, c_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  bool passed = !cfg.verify ||
                verify_hgemm_samples(host_a, host_b, host_c, cfg.M, cfg.N, cfg.K);

  cudaDeviceProp properties{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("kernel: %s\n", kernel_name);
  std::printf("gpu: %s (sm_%d%d)\n", properties.name, properties.major,
              properties.minor);
  std::printf("dtype: fp16 x fp16 -> fp32 accumulate/output\n");
  std::printf("shape: M=%d N=%d K=%d\n", cfg.M, cfg.N, cfg.K);
  std::printf("avg_ms: %.6f\n", average_ms);
  std::printf("TFLOP/s: %.3f\n", tflops);
  std::printf("verification: %s\n", passed ? "PASSED" : "FAILED");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return passed ? 0 : 2;
}
