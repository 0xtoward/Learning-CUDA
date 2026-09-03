#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

// The Ada pipeline below is derived from Sam Patterson's MIT-licensed
// mma-matmul project, then adapted here for arbitrary aligned M/N/K, a
// row-major public B interface, GPU-side persistent RHS prepacking, a 1-D
// thread block, explicit error handling, and this lab's benchmark driver.
// Upstream: https://github.com/spatters/mma-matmul

namespace tensorcore_fp16 {

using namespace nvcuda;

constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

__global__ void hgemm_wmma_naive(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
  const int row = blockIdx.y * kMmaM;
  const int col = blockIdx.x * kMmaN;

  wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, half,
                 wmma::row_major>
      b_fragment;
  wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float>
      accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k = 0; k < K; k += kMmaK) {
    wmma::load_matrix_sync(a_fragment, A + (size_t)row * K + k, K);
    wmma::load_matrix_sync(b_fragment, B + (size_t)k * N + col, N);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }
  wmma::store_matrix_sync(C + (size_t)row * N + col, accumulator, N,
                          wmma::mem_row_major);
}

inline void launch_naive(const half* A, const half* B, float* C, int M, int N,
                         int K, cudaStream_t stream) {
  if ((M % kMmaM) || (N % kMmaN) || (K % kMmaK)) {
    std::fprintf(stderr, "naive WMMA requires M/N/K multiples of 16\n");
    std::exit(EXIT_FAILURE);
  }
  dim3 block(32);
  dim3 grid(N / kMmaN, M / kMmaM);
  hgemm_wmma_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

__forceinline__ __device__ void cp_async_16(uint4* destination,
                                             const uint4* source) {
  unsigned shared_address = __cvta_generic_to_shared(destination);
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
          "r"(shared_address), "l"(source));
}

__forceinline__ __device__ void ldmatrix_x4(unsigned* destination,
                                             const uint4* source) {
  unsigned shared_address = __cvta_generic_to_shared(source);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(destination[0]), "=r"(destination[1]),
        "=r"(destination[2]), "=r"(destination[3])
      : "r"(shared_address));
}

__forceinline__ __device__ void ldmatrix_x2(unsigned* destination,
                                             const uint4* source) {
  unsigned shared_address = __cvta_generic_to_shared(source);
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(destination[0]), "=r"(destination[1])
      : "r"(shared_address));
}

__forceinline__ __device__ void mma_m16n8k16(const unsigned* A,
                                              const unsigned* B, float* C) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
        "r"(B[1]));
}

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kStages = 3;
constexpr int kThreads = 256;

// Shape-specialized alternative for comparatively short M.  It computes a
// 64x64 block tile, which cuts accumulator/shared-memory pressure enough to
// keep two CTAs resident per SM on Ada.
__launch_bounds__(kThreads, 2) __global__ void hgemm_mma_ada_pipeline_64x64(
    const half* A, const half* B_transposed, float* C, int M, int N, int K) {
  __shared__ __align__(16) uint4 shared_a[kStages * 32][8];
  __shared__ __align__(16) uint4 shared_b[kStages * 32][8];

  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int block_row = blockIdx.y * 64;
  const int block_col = blockIdx.x * 64;
  const int warp_offset_a = 16 * (warp / 4);
  const int warp_offset_b = 8 * (warp % 4);

  unsigned a_registers[2][8];
  unsigned b_registers[2][4];
  float accumulators[2][2][4] = {};

  const int store_row = warp * 4 + lane / 8;
  const int store_col = (lane % 8) ^ (lane / 8);
  const int load_row_a = (lane % 16) / 2;
  const int load_col_a =
      (lane / 16 + 4 * (lane % 2)) ^ (load_row_a % 4);
  const int load_row_b = (lane % 8) / 2;
  const int load_col_b =
      (lane / 8 + 4 * (lane % 2)) ^ (load_row_b % 4);

  const int vectors_per_row = K / 8;
  const uint4* global_a = reinterpret_cast<const uint4*>(
      A + (size_t)block_row * K);
  const uint4* global_b = reinterpret_cast<const uint4*>(
      B_transposed + (size_t)block_col * K);
  const uint4* a_address =
      global_a + (warp * 8 + lane / 4) * vectors_per_row + lane % 4;
  const uint4* b_address =
      global_b + (warp * 8 + lane / 4) * vectors_per_row + lane % 4;

#pragma unroll
  for (int stage = 0; stage < kStages - 1; ++stage) {
    uint4(*a_store)[8] = shared_a + 32 * stage;
    uint4(*b_store)[8] = shared_b + 32 * stage;
    cp_async_16(a_store[store_row] + store_col, a_address + stage * 4);
    cp_async_16(b_store[store_row] + store_col, b_address + stage * 4);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  const int reduction_tiles = K / kBlockK;
  for (int tile = 0; tile < reduction_tiles; ++tile) {
    uint4(*a_load)[8] = shared_a + 32 * (tile % kStages);
    uint4(*b_load)[8] = shared_b + 32 * (tile % kStages);
    uint4(*a_store)[8] =
        shared_a + 32 * ((tile + kStages - 1) % kStages);
    uint4(*b_store)[8] =
        shared_b + 32 * ((tile + kStages - 1) % kStages);

    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ldmatrix_x4(a_registers[m],
                  a_load[m * 8 + warp_offset_a + load_row_a] + load_col_a);
      ldmatrix_x4(a_registers[m] + 4,
                  a_load[m * 8 + warp_offset_a + load_row_a] +
                      (load_col_a ^ 2));
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      ldmatrix_x2(b_registers[n],
                  b_load[n * 4 + warp_offset_b + load_row_b] + load_col_b);
      ldmatrix_x2(b_registers[n] + 2,
                  b_load[n * 4 + warp_offset_b + load_row_b] +
                      (load_col_b ^ 2));
    }

    int next_k_vector = (tile + kStages - 1) * 4;
    next_k_vector = min(next_k_vector, vectors_per_row - 4);
    cp_async_16(a_store[store_row] + store_col, a_address + next_k_vector);
    cp_async_16(b_store[store_row] + store_col, b_address + next_k_vector);
    asm volatile("cp.async.commit_group;\n" ::);

#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        mma_m16n8k16(a_registers[m], b_registers[n], accumulators[m][n]);
        mma_m16n8k16(a_registers[m] + 4, b_registers[n] + 2,
                     accumulators[m][n]);
      }
    }
  }

  const int group = lane >> 2;
  const int group_lane = lane & 3;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int row0 = block_row + m * 16 + 2 * warp_offset_a + group;
      const int row1 = row0 + 8;
      const int col = block_col + n * 8 + 2 * warp_offset_b + 2 * group_lane;
      *reinterpret_cast<float2*>(C + (size_t)row0 * N + col) =
          make_float2(accumulators[m][n][0], accumulators[m][n][1]);
      *reinterpret_cast<float2*>(C + (size_t)row1 * N + col) =
          make_float2(accumulators[m][n][2], accumulators[m][n][3]);
    }
  }
}

__launch_bounds__(kThreads, 1) __global__ void hgemm_mma_ada_pipeline(
    const half* A, const half* B_transposed, float* C, int M, int N, int K) {
  __shared__ __align__(16) uint4 shared_a[kStages * 64][8];
  __shared__ __align__(16) uint4 shared_b[kStages * 64][8];

  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  const int warp_offset_a = 32 * (warp / 4);
  const int warp_offset_b = 16 * (warp % 4);

  unsigned a_registers[4][8];
  unsigned b_registers[4][4];
  float accumulators[4][4][4] = {};

  const int store_row = warp * 4 + lane / 8;
  const int store_col = (lane % 8) ^ (lane / 8);
  const int load_row_a = (lane % 16) / 2;
  const int load_col_a =
      (lane / 16 + 4 * (lane % 2)) ^ (load_row_a % 4);
  const int load_row_b = (lane % 8) / 2;
  const int load_col_b =
      (lane / 8 + 4 * (lane % 2)) ^ (load_row_b % 4);

  const int vectors_per_row = K / 8;
  const uint4* global_a = reinterpret_cast<const uint4*>(A +
                                                         (size_t)block_row * K);
  const uint4* global_b = reinterpret_cast<const uint4*>(
      B_transposed + (size_t)block_col * K);
  const uint4* a_address =
      global_a + (warp * 8 + lane / 4) * vectors_per_row + lane % 4;
  const uint4* b_address =
      global_b + (warp * 8 + lane / 4) * vectors_per_row + lane % 4;

#pragma unroll
  for (int stage = 0; stage < kStages - 1; ++stage) {
    const int k_vector = stage * 4;
    uint4(*a_store)[8] = shared_a + 64 * stage;
    uint4(*b_store)[8] = shared_b + 64 * stage;
    cp_async_16(a_store[store_row] + store_col, a_address + k_vector);
    cp_async_16(a_store[store_row + 32] + store_col,
                a_address + 64 * vectors_per_row + k_vector);
    cp_async_16(b_store[store_row] + store_col, b_address + k_vector);
    cp_async_16(b_store[store_row + 32] + store_col,
                b_address + 64 * vectors_per_row + k_vector);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  const int reduction_tiles = K / kBlockK;
  for (int tile = 0; tile < reduction_tiles; ++tile) {
    uint4(*a_load)[8] = shared_a + 64 * (tile % kStages);
    uint4(*b_load)[8] = shared_b + 64 * (tile % kStages);
    uint4(*a_store)[8] =
        shared_a + 64 * ((tile + kStages - 1) % kStages);
    uint4(*b_store)[8] =
        shared_b + 64 * ((tile + kStages - 1) % kStages);

    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

#pragma unroll
    for (int m = 0; m < 4; ++m) {
      ldmatrix_x4(a_registers[m],
                  a_load[m * 8 + warp_offset_a + load_row_a] + load_col_a);
      ldmatrix_x4(a_registers[m] + 4,
                  a_load[m * 8 + warp_offset_a + load_row_a] +
                      (load_col_a ^ 2));
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      ldmatrix_x2(b_registers[n],
                  b_load[n * 4 + warp_offset_b + load_row_b] + load_col_b);
      ldmatrix_x2(b_registers[n] + 2,
                  b_load[n * 4 + warp_offset_b + load_row_b] +
                      (load_col_b ^ 2));
    }

    int next_k_vector = (tile + kStages - 1) * 4;
    next_k_vector = min(next_k_vector, vectors_per_row - 4);
    cp_async_16(a_store[store_row] + store_col, a_address + next_k_vector);
    cp_async_16(a_store[store_row + 32] + store_col,
                a_address + 64 * vectors_per_row + next_k_vector);
    cp_async_16(b_store[store_row] + store_col, b_address + next_k_vector);
    cp_async_16(b_store[store_row + 32] + store_col,
                b_address + 64 * vectors_per_row + next_k_vector);
    asm volatile("cp.async.commit_group;\n" ::);

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        mma_m16n8k16(a_registers[m], b_registers[n], accumulators[m][n]);
        mma_m16n8k16(a_registers[m] + 4, b_registers[n] + 2,
                     accumulators[m][n]);
      }
    }
  }

  const int group = lane >> 2;
  const int group_lane = lane & 3;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int row0 = block_row + m * 16 + 2 * warp_offset_a + group;
      const int row1 = row0 + 8;
      const int col = block_col + n * 8 + 2 * warp_offset_b + 2 * group_lane;
      *reinterpret_cast<float2*>(C + (size_t)row0 * N + col) =
          make_float2(accumulators[m][n][0], accumulators[m][n][1]);
      *reinterpret_cast<float2*>(C + (size_t)row1 * N + col) =
          make_float2(accumulators[m][n][2], accumulators[m][n][3]);
    }
  }
}

__global__ void transpose_rhs(const half* source, half* destination, int rows,
                              int columns) {
  __shared__ half tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    if (x < columns && y + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          source[(size_t)(y + offset) * columns + x];
    }
  }
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    if (x < rows && y + offset < columns) {
      destination[(size_t)(y + offset) * rows + x] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

struct PackedRhsCache {
  const half* source = nullptr;
  half* transposed = nullptr;
  int K = 0;
  int N = 0;
};

inline PackedRhsCache& rhs_cache() {
  static PackedRhsCache cache;
  return cache;
}

inline const half* prepare_rhs(const half* B, int K, int N,
                               cudaStream_t stream) {
  PackedRhsCache& cache = rhs_cache();
  if (cache.source == B && cache.K == K && cache.N == N && cache.transposed) {
    return cache.transposed;
  }
  if (cache.transposed) cudaFree(cache.transposed);
  cudaError_t status = cudaMalloc(&cache.transposed, (size_t)K * N * sizeof(half));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "RHS prepack allocation failed: %s\n",
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (K + 31) / 32);
  transpose_rhs<<<grid, block, 0, stream>>>(B, cache.transposed, K, N);
  cache.source = B;
  cache.K = K;
  cache.N = N;
  return cache.transposed;
}

inline void launch_ada(const half* A, const half* B, float* C, int M, int N,
                       int K, cudaStream_t stream) {
  if ((M % kBlockM) || (N % kBlockN) || (K % kBlockK) || K < 64) {
    launch_naive(A, B, C, M, N, K, stream);
    return;
  }
  const half* packed_b = prepare_rhs(B, K, N, stream);
  dim3 block(kThreads);
  dim3 grid(N / kBlockN, M / kBlockM);
  hgemm_mma_ada_pipeline<<<grid, block, 0, stream>>>(A, packed_b, C, M, N, K);
}

inline void launch_ada_64x64(const half* A, const half* B, float* C, int M,
                             int N, int K, cudaStream_t stream) {
  if ((M % 64) || (N % 64) || (K % kBlockK) || K < 64) {
    launch_naive(A, B, C, M, N, K, stream);
    return;
  }
  const half* packed_b = prepare_rhs(B, K, N, stream);
  dim3 block(kThreads);
  dim3 grid(N / 64, M / 64);
  hgemm_mma_ada_pipeline_64x64<<<grid, block, 0, stream>>>(A, packed_b, C, M,
                                                           N, K);
}

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t status_ = (call);                                              \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %d\n", static_cast<int>(status_));      \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                            \
  } while (0)

inline cublasHandle_t& cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) CUBLAS_CHECK(cublasCreate(&handle));
  return handle;
}

inline void launch_cublas(const half* A, const half* B, float* C, int M, int N,
                          int K, cudaStream_t stream) {
  cublasHandle_t handle = cublas_handle();
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
      CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace tensorcore_fp16
