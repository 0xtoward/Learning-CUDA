#include "gemm_common.cuh"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error %d\n", static_cast<int>(status)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static cublasHandle_t handle;

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  static bool initialized = false;
  if (!initialized) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    initialized = true;
  }
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

#define KERNEL_NAME "cublas_sgemm_tf32"
#include "gemm_entrypoint.cuh"
