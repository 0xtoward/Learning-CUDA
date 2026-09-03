// The smallest useful CUTLASS lesson in this repository:
// instantiate a device-wide GEMM type, build its Arguments, and call it.
//
// This FP32 version intentionally uses CUTLASS's default OpClassSimt path.
// CUTLASS is a kernel-construction library; it does not imply Tensor Cores.

#include "gemm_common.cuh"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

using RowMajor = cutlass::layout::RowMajor;

// The remaining template parameters have sensible defaults. For FP32 these
// defaults select a tiled SIMT/CUDA-Core GEMM.
using CutlassSgemm = cutlass::gemm::device::Gemm<
    float, RowMajor,  // A element and layout
    float, RowMajor,  // B element and layout
    float, RowMajor   // C/D element and layout
    >;

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                 cudaStream_t stream) {
  static CutlassSgemm operation;

  // CUTLASS implements D = alpha * A * B + beta * C.
  CutlassSgemm::Arguments arguments(
      {M, N, K},  // problem size
      {A, K},      // row-major A: leading dimension K
      {B, N},      // row-major B: leading dimension N
      {C, N},      // source C
      {C, N},      // destination D (reuse the same allocation)
      {1.0f, 0.0f});

  cutlass::Status status = operation(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS GEMM failed: %s\n",
                 cutlassGetStatusString(status));
    std::exit(EXIT_FAILURE);
  }
}

#define KERNEL_NAME "cutlass_minimal_sgemm_simt"
#include "gemm_entrypoint.cuh"
