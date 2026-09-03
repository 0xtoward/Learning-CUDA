#include "gemm_common.cuh"

// Same spatial tiling as v7_warptiling, but with two SMEM stages.  While the
// warp tiles consume stage s, cp.async fills stage s^1 with the next K tile.
constexpr int BM = 64, BN = 64, BK = 8;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int WM = BM / WARPS_M;
constexpr int WN = BN / WARPS_N;
constexpr int TM = 4, TN = 8;
constexpr int THREADS = 128;

__device__ __forceinline__ void copy_async_16B(float* smem_dst,
                                                const float* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
               "r"(smem_addr), "l"(gmem_src));
#else
  *reinterpret_cast<float4*>(smem_dst) =
      *reinterpret_cast<const float4*>(gmem_src);
#endif
}

__device__ __forceinline__ void async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void issue_tile(
    const float* A, const float* B,
    float (*As)[BK], float (*Bs)[BN],
    int M, int N, int K, int block_row, int block_col, int k0) {
  int tid = threadIdx.x;

  // One 16-byte A chunk per thread.  XOR swaps the two half-rows for every
  // four rows: vector stores stay aligned, while the compute-side column
  // mapping avoids the stride-8 bank conflict without a padding column.
  int ar = tid / 2;
  int a_chunk = tid & 1;
  int physical_chunk = a_chunk ^ ((ar & 4) ? 1 : 0);
  int gr = block_row + ar;
  int gk = k0 + a_chunk * 4;
  if (gr < M && gk + 3 < K && (K % 4 == 0)) {
    copy_async_16B(&As[ar][physical_chunk * 4],
                   &A[(size_t)gr * K + gk]);
  } else {
#pragma unroll
    for (int q = 0; q < 4; ++q)
      As[ar][physical_chunk * 4 + q] =
          (gr < M && gk + q < K) ? A[(size_t)gr * K + gk + q] : 0.0f;
  }

  // One aligned 16-byte B chunk per thread.
  int bk = tid / 16;
  int b_chunk = tid & 15;
  int gc = block_col + b_chunk * 4;
  int bgk = k0 + bk;
  if (bgk < K && gc + 3 < N && (N % 4 == 0)) {
    copy_async_16B(&Bs[bk][b_chunk * 4],
                   &B[(size_t)bgk * N + gc]);
  } else {
#pragma unroll
    for (int q = 0; q < 4; ++q)
      Bs[bk][b_chunk * 4 + q] =
          (bgk < K && gc + q < N) ? B[(size_t)bgk * N + gc + q] : 0.0f;
  }
  async_commit();
}

__global__ void gemm_v3_cp_async_double_buffered(
    const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[2][BM][BK];
  __shared__ float Bs[2][BK][BN];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int warp_m = warp / WARPS_N;
  int warp_n = warp % WARPS_N;
  int lane_m = lane / 4;
  int lane_n = lane % 4;
  int lr0 = warp_m * WM + lane_m * TM;
  int lc0 = warp_n * WN + lane_n * TN;
  float acc[TM][TN] = {};

  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  int stage = 0;

  issue_tile(A, B, As[stage], Bs[stage], M, N, K,
             block_row, block_col, 0);
  async_wait_all();
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += BK) {
    int next_k0 = k0 + BK;
    int next_stage = stage ^ 1;
    if (next_k0 < K) {
      issue_tile(A, B, As[next_stage], Bs[next_stage], M, N, K,
                 block_row, block_col, next_k0);
    }

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float arx[TM], brx[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int r = lr0 + i;
        int physical_k = k ^ ((r & 4) ? 4 : 0);
        arx[i] = As[stage][r][physical_k];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) brx[j] = Bs[stage][k][lc0 + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(arx[i], brx[j], acc[i][j]);
    }

    if (next_k0 < K) async_wait_all();
    __syncthreads();
    stage = next_stage;
  }

  int gr0 = block_row + lr0;
  int gc0 = block_col + lc0;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = gr0 + i;
    if (r >= M) continue;
    if (gc0 + TN <= N && (N % 4 == 0)) {
      float4 o0{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
      float4 o1{acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
      *reinterpret_cast<float4*>(&C[(size_t)r * N + gc0]) = o0;
      *reinterpret_cast<float4*>(&C[(size_t)r * N + gc0 + 4]) = o1;
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j)
        if (gc0 + j < N) C[(size_t)r * N + gc0 + j] = acc[i][j];
    }
  }
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v3_cp_async_double_buffered<<<grid, block, 0, stream>>>(
      A, B, C, M, N, K);
}

#define KERNEL_NAME "v3_cp_async_double_buffered"
#include "gemm_entrypoint.cuh"
