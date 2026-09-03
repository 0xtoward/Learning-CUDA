#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace tensorcore_fp16_cutlass {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator,
    ElementAccumulator>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// SM80's mma.sync/cp.async mainloop is also the relevant CUTLASS path for
// consumer Ada (SM89); nvcc still emits an sm_89 cubin for the final kernel.
template <typename ThreadblockShape, typename WarpShape, int Stages = 3>
using GemmForShape = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, Epilogue, Swizzle, Stages, 8,
    8>;

using Gemm128x128 = GemmForShape<cutlass::gemm::GemmShape<128, 128, 32>,
                                cutlass::gemm::GemmShape<64, 64, 32>>;
using Gemm128x64 = GemmForShape<cutlass::gemm::GemmShape<128, 64, 32>,
                               cutlass::gemm::GemmShape<64, 32, 32>>;
using Gemm64x128 = GemmForShape<cutlass::gemm::GemmShape<64, 128, 32>,
                               cutlass::gemm::GemmShape<32, 64, 32>>;
using Gemm64x128Stages2 = GemmForShape<
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>, 2>;
using Gemm64x128Stages4 = GemmForShape<
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>, 4>;
using Gemm64x64 = GemmForShape<cutlass::gemm::GemmShape<64, 64, 32>,
                              cutlass::gemm::GemmShape<32, 32, 32>>;

template <typename Gemm>
inline void launch_gemm(const half* A, const half* B, float* C, int M, int N,
                        int K, cudaStream_t stream) {
  static Gemm operation;
  typename Gemm::Arguments arguments(
      {M, N, K},
      {reinterpret_cast<const ElementA*>(A), K},
      {reinterpret_cast<const ElementB*>(B), N},
      {C, N},
      {C, N},
      {1.0f, 0.0f});
  cutlass::Status status = operation(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS GEMM failed: %s\n",
                 cutlassGetStatusString(status));
    std::exit(EXIT_FAILURE);
  }
}

inline void launch_128x128(const half* A, const half* B, float* C, int M, int N,
                           int K, cudaStream_t stream) {
  launch_gemm<Gemm128x128>(A, B, C, M, N, K, stream);
}

inline void launch_128x64(const half* A, const half* B, float* C, int M, int N,
                          int K, cudaStream_t stream) {
  launch_gemm<Gemm128x64>(A, B, C, M, N, K, stream);
}

inline void launch_64x128(const half* A, const half* B, float* C, int M, int N,
                          int K, cudaStream_t stream) {
  launch_gemm<Gemm64x128>(A, B, C, M, N, K, stream);
}

inline void launch_64x128_stages2(const half* A, const half* B, float* C, int M,
                                  int N, int K, cudaStream_t stream) {
  launch_gemm<Gemm64x128Stages2>(A, B, C, M, N, K, stream);
}

inline void launch_64x128_stages4(const half* A, const half* B, float* C, int M,
                                  int N, int K, cudaStream_t stream) {
  launch_gemm<Gemm64x128Stages4>(A, B, C, M, N, K, stream);
}

inline void launch_64x64(const half* A, const half* B, float* C, int M, int N,
                         int K, cudaStream_t stream) {
  launch_gemm<Gemm64x64>(A, B, C, M, N, K, stream);
}

}  // namespace tensorcore_fp16_cutlass
