#include "gemm_common.cuh"
#include <cublas_v2.h>
#define CUBLAS_CHECK(x) do{cublasStatus_t s=(x);if(s!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS error %d\n",int(s));std::exit(1);}}while(0)

static cublasHandle_t handle;
void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  static bool init=false;
  if(!init){ CUBLAS_CHECK(cublasCreate(&handle)); init=true; }
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha=1.0f,beta=0.0f;
  // Row-major C=A*B is equivalent to column-major C^T=B^T*A^T.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha, B, N, A, K, &beta, C, N));
}
#define KERNEL_NAME "cublas_sgemm"
#include "gemm_entrypoint.cuh"
