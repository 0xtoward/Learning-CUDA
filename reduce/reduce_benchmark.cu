#include "include/reduce_kernels.cuh"

#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename Op>
float cpu_reference(const std::vector<float>& input) {
  Op op;
  float result = Op::identity();
  if constexpr (std::is_same_v<Op, reduce_lab::SumOp>) {
    double accurate = 0.0;
    for (float value : input) accurate += value;
    return static_cast<float>(accurate);
  } else {
    for (float value : input) result = op(result, value);
    return result;
  }
}

template <typename Op>
void launch_cub(void* temporary, size_t& temporary_bytes, const float* input,
                float* output, int n, cudaStream_t stream = nullptr) {
  cub::DeviceReduce::Reduce(temporary, temporary_bytes, input, output, n, Op{},
                            Op::identity(), stream);
}

template <typename Op>
int run(const std::string& version, int n, int iterations, int warmup,
        bool verify) {
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> host_input(n);
  for (float& value : host_input) value = distribution(generator);

  float* device_input = nullptr;
  float* device_output = nullptr;
  reduce_lab::BenchmarkWorkspace workspace;
  REDUCE_CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
  REDUCE_CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
  REDUCE_CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice));
  workspace.ensure(n);

  void* cub_temporary = nullptr;
  size_t cub_bytes = 0;
  if (version == "cub") {
    launch_cub<Op>(nullptr, cub_bytes, device_input, device_output, n);
    REDUCE_CUDA_CHECK(cudaMalloc(&cub_temporary, cub_bytes));
  }

  auto launch = [&] {
    if (version == "v0") {
      reduce_lab::launch_v0<Op>(device_input, device_output, workspace.buffer_a,
                                workspace.buffer_b, n);
    } else if (version == "v1") {
      reduce_lab::launch_v1<Op>(device_input, device_output, workspace.buffer_a,
                                workspace.buffer_b, n);
    } else if (version == "v2") {
      reduce_lab::launch_v2<Op>(device_input, device_output, workspace.buffer_a,
                                workspace.buffer_b, n);
    } else if (version == "v3") {
      reduce_lab::launch_v3<Op>(device_input, device_output, workspace.buffer_a,
                                workspace.buffer_b, n, workspace.max_blocks);
    } else if (version == "cub") {
      launch_cub<Op>(cub_temporary, cub_bytes, device_input, device_output, n);
    } else {
      std::cerr << "Unknown version: " << version << "\n";
      std::exit(EXIT_FAILURE);
    }
  };

  for (int i = 0; i < warmup; ++i) launch();
  REDUCE_CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  REDUCE_CUDA_CHECK(cudaEventCreate(&start));
  REDUCE_CUDA_CHECK(cudaEventCreate(&stop));
  REDUCE_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) launch();
  REDUCE_CUDA_CHECK(cudaEventRecord(stop));
  REDUCE_CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  REDUCE_CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float milliseconds = total_ms / iterations;

  float result = 0.0f;
  REDUCE_CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float),
                               cudaMemcpyDeviceToHost));
  const float reference = verify ? cpu_reference<Op>(host_input) : result;
  const float error = std::abs(result - reference);
  const float tolerance = std::is_same_v<Op, reduce_lab::SumOp>
                              ? 0.05f + 1.0e-4f * std::abs(reference)
                              : 1.0e-6f;
  const bool correct = !verify || error <= tolerance;
  const double effective_gbps =
      static_cast<double>(n) * sizeof(float) / (milliseconds * 1.0e6);

  std::cout << std::fixed << std::setprecision(6)
            << "version=" << version << " op=" << Op::name() << " N=" << n
            << " time_ms=" << milliseconds
            << " effective_GBps=" << effective_gbps << " result=" << result;
  if (verify) {
    std::cout << " reference=" << reference << " abs_error=" << error
              << " correctness=" << (correct ? "PASS" : "FAIL");
  }
  std::cout << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (cub_temporary) cudaFree(cub_temporary);
  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(workspace.buffer_a);
  cudaFree(workspace.buffer_b);
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string version = argc > 1 ? argv[1] : "v3";
  const std::string operation = argc > 2 ? argv[2] : "sum";
  const int n = argc > 3 ? std::stoi(argv[3]) : (1 << 24);
  const int iterations = argc > 4 ? std::stoi(argv[4]) : 100;
  const int warmup = argc > 5 ? std::stoi(argv[5]) : 20;
  const bool verify = argc <= 6 || std::strcmp(argv[6], "--no-verify") != 0;

  if (n <= 0 || iterations <= 0 || warmup < 0) {
    std::cerr << "N and iterations must be positive; warmup must be >= 0\n";
    return EXIT_FAILURE;
  }
  if (operation == "sum")
    return run<reduce_lab::SumOp>(version, n, iterations, warmup, verify);
  if (operation == "max")
    return run<reduce_lab::MaxOp>(version, n, iterations, warmup, verify);
  if (operation == "min")
    return run<reduce_lab::MinOp>(version, n, iterations, warmup, verify);
  std::cerr << "Unknown operation: " << operation << " (sum|max|min)\n";
  return EXIT_FAILURE;
}
