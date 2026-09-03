#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace reduce_lab {

constexpr int kBlockSize = 256;

#define REDUCE_CUDA_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t error__ = (call);                                               \
    if (error__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error__));                               \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

struct SumOp {
  __host__ __device__ float operator()(float a, float b) const { return a + b; }
  __host__ __device__ static constexpr float identity() { return 0.0f; }
  static constexpr const char* name() { return "sum"; }
};

struct MaxOp {
  __host__ __device__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
  __host__ __device__ static constexpr float identity() { return -FLT_MAX; }
  static constexpr const char* name() { return "max"; }
};

struct MinOp {
  __host__ __device__ float operator()(float a, float b) const {
    return a < b ? a : b;
  }
  __host__ __device__ static constexpr float identity() { return FLT_MAX; }
  static constexpr const char* name() { return "min"; }
};

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

// V0: one element per thread and interleaved modulo addressing. This is
// intentionally branch-heavy so the first optimization has a visible cause.
template <typename Op>
__global__ void reduce_v0_interleaved(const float* input, float* partial,
                                      int n) {
  extern __shared__ float values[];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;
  values[tid] = index < n ? input[index] : Op::identity();
  __syncthreads();

  Op op;
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    if ((tid % (stride << 1)) == 0 && tid + stride < blockDim.x) {
      values[tid] = op(values[tid], values[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = values[0];
}

// V1: coalesced two-elements-per-thread load and sequential shared-memory
// addressing. It halves the number of blocks and removes modulo divergence.
template <typename Op>
__global__ void reduce_v1_sequential(const float* input, float* partial,
                                     int n) {
  extern __shared__ float values[];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * (blockDim.x * 2) + tid;
  Op op;
  float value = index < n ? input[index] : Op::identity();
  if (index + blockDim.x < n) value = op(value, input[index + blockDim.x]);
  values[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) values[tid] = op(values[tid], values[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = values[0];
}

template <typename Op>
__forceinline__ __device__ float warp_reduce(float value) {
  Op op;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <typename Op>
__forceinline__ __device__ float block_reduce(float value) {
  __shared__ float warp_values[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce<Op>(value);
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();

  value = threadIdx.x < (blockDim.x + 31) / 32
              ? warp_values[lane]
              : Op::identity();
  if (warp == 0) value = warp_reduce<Op>(value);
  return value;
}

// V2: shared memory is only used for one value per warp; shuffle instructions
// perform the intra-warp tree without barriers.
template <typename Op>
__global__ void reduce_v2_warp_shuffle(const float* input, float* partial,
                                       int n) {
  const int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  Op op;
  float value = index < n ? input[index] : Op::identity();
  if (index + blockDim.x < n) value = op(value, input[index + blockDim.x]);
  value = block_reduce<Op>(value);
  if (threadIdx.x == 0) partial[blockIdx.x] = value;
}

// V3: each block grid-strides through many aligned float4 values. Capping the
// grid makes the first pass produce only a small number of partials, so the
// complete scalar reduction normally needs two kernel launches.
template <typename Op>
__global__ void reduce_v3_vectorized(const float* input, float* partial,
                                     int n) {
  const int vector_count = n / 4;
  const float4* vectors = reinterpret_cast<const float4*>(input);
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_threads = gridDim.x * blockDim.x;
  Op op;
  float value = Op::identity();

  for (int i = global_thread; i < vector_count; i += grid_threads) {
    const float4 v = vectors[i];
    value = op(value, v.x);
    value = op(value, v.y);
    value = op(value, v.z);
    value = op(value, v.w);
  }
  for (int i = vector_count * 4 + global_thread; i < n; i += grid_threads) {
    value = op(value, input[i]);
  }

  value = block_reduce<Op>(value);
  if (threadIdx.x == 0) partial[blockIdx.x] = value;
}

inline int max_partial_count(int n) { return std::max(1, ceil_div(n, kBlockSize)); }

template <typename Op>
inline void launch_v0(const float* input, float* output, float* buffer_a,
                      float* buffer_b, int n, cudaStream_t stream = nullptr) {
  if (n <= 1) {
    REDUCE_CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));
    return;
  }
  const float* source = input;
  int count = n;
  bool use_a = true;
  while (count > 1) {
    const int blocks = ceil_div(count, kBlockSize);
    float* destination = blocks == 1 ? output : (use_a ? buffer_a : buffer_b);
    reduce_v0_interleaved<Op><<<blocks, kBlockSize,
                                kBlockSize * sizeof(float), stream>>>(
        source, destination, count);
    source = destination;
    count = blocks;
    use_a = !use_a;
  }
}

template <typename Op>
inline void launch_v1(const float* input, float* output, float* buffer_a,
                      float* buffer_b, int n, cudaStream_t stream = nullptr) {
  if (n <= 1) {
    REDUCE_CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));
    return;
  }
  const float* source = input;
  int count = n;
  bool use_a = true;
  while (count > 1) {
    const int blocks = ceil_div(count, kBlockSize * 2);
    float* destination = blocks == 1 ? output : (use_a ? buffer_a : buffer_b);
    reduce_v1_sequential<Op><<<blocks, kBlockSize,
                               kBlockSize * sizeof(float), stream>>>(
        source, destination, count);
    source = destination;
    count = blocks;
    use_a = !use_a;
  }
}

template <typename Op>
inline void launch_v2(const float* input, float* output, float* buffer_a,
                      float* buffer_b, int n, cudaStream_t stream = nullptr) {
  if (n <= 1) {
    REDUCE_CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));
    return;
  }
  const float* source = input;
  int count = n;
  bool use_a = true;
  while (count > 1) {
    const int blocks = ceil_div(count, kBlockSize * 2);
    float* destination = blocks == 1 ? output : (use_a ? buffer_a : buffer_b);
    reduce_v2_warp_shuffle<Op><<<blocks, kBlockSize, 0, stream>>>(
        source, destination, count);
    source = destination;
    count = blocks;
    use_a = !use_a;
  }
}

template <typename Op>
inline void launch_v3(const float* input, float* output, float* buffer_a,
                      float* buffer_b, int n, int max_blocks,
                      cudaStream_t stream = nullptr) {
  if (n <= 1) {
    REDUCE_CUDA_CHECK(cudaMemcpyAsync(output, input, sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));
    return;
  }
  const int blocks = std::max(
      1, std::min(max_blocks, ceil_div(n, kBlockSize * 4)));
  float* first_output = blocks == 1 ? output : buffer_a;
  reduce_v3_vectorized<Op><<<blocks, kBlockSize, 0, stream>>>(
      input, first_output, n);
  if (blocks > 1) {
    // The large vectorized pass leaves only O(SM count) values. Reuse the
    // simple shuffle hierarchy to finish them.
    launch_v2<Op>(first_output, output, buffer_b, buffer_a, blocks, stream);
  }
}

struct BenchmarkWorkspace {
  float* buffer_a = nullptr;
  float* buffer_b = nullptr;
  int capacity = 0;
  int max_blocks = 0;

  void ensure(int n) {
    const int required = max_partial_count(n);
    if (required > capacity) {
      if (buffer_a) REDUCE_CUDA_CHECK(cudaFree(buffer_a));
      if (buffer_b) REDUCE_CUDA_CHECK(cudaFree(buffer_b));
      REDUCE_CUDA_CHECK(cudaMalloc(&buffer_a, required * sizeof(float)));
      REDUCE_CUDA_CHECK(cudaMalloc(&buffer_b, required * sizeof(float)));
      capacity = required;
    }
    if (max_blocks == 0) {
      int device = 0;
      cudaDeviceProp properties{};
      REDUCE_CUDA_CHECK(cudaGetDevice(&device));
      REDUCE_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      max_blocks = properties.multiProcessorCount * 8;
    }
  }
};

}  // namespace reduce_lab
