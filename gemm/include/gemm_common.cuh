#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};
#else
struct NvtxRange {
  explicit NvtxRange(const char*) {}
};
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

struct BenchConfig {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  int warmup = 5;
  int iters = 20;
  bool verify = true;
};

inline BenchConfig parse_args(int argc, char** argv) {
  BenchConfig c;
  if (argc > 1) c.M = std::atoi(argv[1]);
  if (argc > 2) c.N = std::atoi(argv[2]);
  if (argc > 3) c.K = std::atoi(argv[3]);
  if (argc > 4) c.iters = std::atoi(argv[4]);
  if (argc > 5) c.warmup = std::atoi(argv[5]);
  if (argc > 6 && std::string(argv[6]) == "--no-verify") c.verify = false;
  return c;
}

inline void fill_random(std::vector<float>& x, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(gen);
}

inline bool verify_samples(const std::vector<float>& A,
                           const std::vector<float>& B,
                           const std::vector<float>& C,
                           int M, int N, int K) {
  if (M == 0 || N == 0 || K == 0) return true;
  const int samples = 16;
  for (int s = 0; s < samples; ++s) {
    int r = (s * 131 + 7) % M;
    int c = (s * 197 + 11) % N;
    double ref = 0.0;
    for (int k = 0; k < K; ++k) {
      ref += static_cast<double>(A[(size_t)r * K + k]) *
             static_cast<double>(B[(size_t)k * N + c]);
    }
    double got = C[(size_t)r * N + c];
    double abs_err = std::abs(got - ref);
    double tol = 2e-3 * std::max(1.0, std::abs(ref));
    if (abs_err > tol) {
      std::fprintf(stderr, "verify failed at (%d,%d): got=%g ref=%g abs_err=%g tol=%g\n",
                   r, c, got, ref, abs_err, tol);
      return false;
    }
  }
  return true;
}

using LaunchFn = void(*)(const float*, const float*, float*, int, int, int, cudaStream_t);

inline int run_benchmark(const char* kernel_name, LaunchFn launch, int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);
  const size_t a_elems = (size_t)cfg.M * cfg.K;
  const size_t b_elems = (size_t)cfg.K * cfg.N;
  const size_t c_elems = (size_t)cfg.M * cfg.N;

  std::vector<float> hA(a_elems), hB(b_elems), hC(c_elems);
  fill_random(hA, 1);
  fill_random(hB, 2);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  {
    NvtxRange range("gemm.setup.alloc-copy");
    CUDA_CHECK(cudaMalloc(&dA, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, c_elems * sizeof(float)));
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  {
    NvtxRange range("gemm.warmup");
    for (int i = 0; i < cfg.warmup; ++i) launch(dA, dB, dC, cfg.M, cfg.N, cfg.K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  {
    NvtxRange range("gemm.measure");
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < cfg.iters; ++i) launch(dA, dB, dC, cfg.M, cfg.N, cfg.K, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
  }

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / cfg.iters;
  double flops = 2.0 * (double)cfg.M * cfg.N * cfg.K;
  double gflops = flops / (avg_ms * 1.0e6);

  {
    NvtxRange range("gemm.copyback-verify");
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
  }
  bool ok = !cfg.verify || verify_samples(hA, hB, hC, cfg.M, cfg.N, cfg.K);

  cudaDeviceProp prop{};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  std::printf("kernel: %s\n", kernel_name);
  std::printf("gpu: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("shape: M=%d N=%d K=%d\n", cfg.M, cfg.N, cfg.K);
  std::printf("avg_ms: %.6f\n", avg_ms);
  std::printf("GFLOP/s: %.3f\n", gflops);
  std::printf("verification: %s\n", ok ? "PASSED" : "FAILED");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return ok ? 0 : 2;
}
