// Minimal Tensor Core GEMM: one warp computes one 16x16 output tile.
// This is intentionally easy to read, not fast: every K step loads fragments
// directly from global memory, so the Tensor Cores spend much of their time
// waiting for data.

#include "hgemm_common.cuh"

#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;

__global__ void wmma_minimal_kernel(const half* A, const half* B, float* C,
                                    int M, int N, int K) {
  // The whole warp agrees on the tile origin.
  const int row = blockIdx.y * kM;
  const int col = blockIdx.x * kN;

  // These objects are logical warp-wide fragments. Their physical values are
  // distributed among the 32 lanes' private registers.
  wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kM, kN, kK, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // All 32 lanes must execute these warp-synchronous calls together.
  for (int k0 = 0; k0 < K; k0 += kK) {
    wmma::load_matrix_sync(a_frag, A + (size_t)row * K + k0, K);
    wmma::load_matrix_sync(b_frag, B + (size_t)k0 * N + col, N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(C + (size_t)row * N + col, acc_frag, N,
                          wmma::mem_row_major);
}

void launch_wmma(const half* A, const half* B, float* C, int M, int N, int K,
                 cudaStream_t stream) {
  // Keeping the minimal kernel branch-free also makes the warp-uniform WMMA
  // contract obvious. Production code needs a padded or predicated edge path.
  if ((M % 16) || (N % 16) || (K % 16)) {
    std::fprintf(stderr, "wmma_minimal requires M/N/K multiples of 16\n");
    std::exit(EXIT_FAILURE);
  }
  dim3 block(32);  // exactly one warp
  dim3 grid(N / 16, M / 16);
  wmma_minimal_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
  HgemmBenchConfig config = parse_hgemm_args(argc, argv, 1);
  return run_hgemm_benchmark("wmma_minimal_one_warp_per_tile", launch_wmma,
                             config);
}
