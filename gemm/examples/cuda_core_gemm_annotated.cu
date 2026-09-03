// A readable, moderately optimized FP32 CUDA-Core GEMM.
//
// Data path:
//   global memory -> vectorized cooperative load -> shared-memory CTA tile
//   -> per-warp tile -> per-thread register micro-tile -> global memory
//
// No PTX is required. NVCC lowers float4 loads/stores and fmaf() to suitable
// PTX/SASS. Define GEMM_DEMO_INLINE_PTX_FMA only to demonstrate inline PTX;
// it is not expected to outperform fmaf() and may constrain scheduling.

#include "gemm_common.cuh"

constexpr int BM = 64;  // rows of C owned by one CTA
constexpr int BN = 64;  // columns of C owned by one CTA
constexpr int BK = 8;   // K values staged at a time

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WM = BM / WARPS_M;  // each warp owns a 32x32 C tile
constexpr int WN = BN / WARPS_N;

constexpr int TM = 4;  // each thread accumulates a 4x8 micro-tile
constexpr int TN = 8;
constexpr int THREADS = 128;  // four warps

__device__ __forceinline__ float multiply_add(float a, float b, float c) {
#ifdef GEMM_DEMO_INLINE_PTX_FMA
  float out;
  asm("fma.rn.f32 %0, %1, %2, %3;"
      : "=f"(out)
      : "f"(a), "f"(b), "f"(c));
  return out;
#else
  return fmaf(a, b, c);
#endif
}

__global__ void cuda_core_gemm_annotated(const float* A, const float* B,
                                         float* C, int M, int N, int K) {
  // The +1 changes A's shared-memory row stride from 8 to 9 floats. That
  // breaks an unfortunate periodic mapping onto the 32 shared-memory banks.
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  // Four warps form a 2x2 arrangement inside the 64x64 CTA tile.
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;

  // Within one 32x32 warp tile, 32 lanes form an 8x4 arrangement. Each lane
  // owns 4 rows x 8 columns, so 8*4 rows and 4*8 columns cover 32x32.
  const int lane_m = lane / 4;
  const int lane_n = lane % 4;
  const int local_row = warp_m * WM + lane_m * TM;
  const int local_col = warp_n * WN + lane_n * TN;

  // C stays in registers for the complete K loop. This is the main reason the
  // kernel reuses A/B much better than one-thread-one-output code.
  float accum[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile has 64*8 = 512 floats = 128 float4 values: one per thread.
    const int a_vector = tid;
    const int a_row = a_vector / (BK / 4);
    const int a_col4 = a_vector % (BK / 4);
    const int global_a_row = blockIdx.y * BM + a_row;
    const int global_a_col = k0 + a_col4 * 4;

    if (global_a_row < M && global_a_col + 3 < K && K % 4 == 0) {
      float4 value = *reinterpret_cast<const float4*>(
          A + (size_t)global_a_row * K + global_a_col);
      As[a_row][a_col4 * 4 + 0] = value.x;
      As[a_row][a_col4 * 4 + 1] = value.y;
      As[a_row][a_col4 * 4 + 2] = value.z;
      As[a_row][a_col4 * 4 + 3] = value.w;
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        As[a_row][a_col4 * 4 + q] =
            (global_a_row < M && global_a_col + q < K)
                ? A[(size_t)global_a_row * K + global_a_col + q]
                : 0.0f;
      }
    }

    // B tile also contains 128 float4 values. Consecutive lanes load
    // consecutive B columns, producing coalesced global-memory transactions.
    const int b_vector = tid;
    const int b_row = b_vector / (BN / 4);
    const int b_col4 = b_vector % (BN / 4);
    const int global_b_row = k0 + b_row;
    const int global_b_col = blockIdx.x * BN + b_col4 * 4;

    if (global_b_row < K && global_b_col + 3 < N && N % 4 == 0) {
      *reinterpret_cast<float4*>(&Bs[b_row][b_col4 * 4]) =
          *reinterpret_cast<const float4*>(
              B + (size_t)global_b_row * N + global_b_col);
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        Bs[b_row][b_col4 * 4 + q] =
            (global_b_row < K && global_b_col + q < N)
                ? B[(size_t)global_b_row * N + global_b_col + q]
                : 0.0f;
      }
    }

    // Every warp must see the complete CTA tile before consuming it.
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Load one A column slice and one B row slice from SMEM to registers.
      float a_reg[TM];
      float b_reg[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_reg[i] = As[local_row + i][k];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_reg[j] = Bs[k][local_col + j];
      }

      // Register outer product: TM A values x TN B values update 32 outputs.
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          accum[i][j] = multiply_add(a_reg[i], b_reg[j], accum[i][j]);
        }
      }
    }

    // The next iteration overwrites As/Bs, so all warps must finish first.
    __syncthreads();
  }

  const int global_c_row0 = blockIdx.y * BM + local_row;
  const int global_c_col0 = blockIdx.x * BN + local_col;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = global_c_row0 + i;
    if (row >= M) continue;

    if (global_c_col0 + TN <= N && N % 4 == 0) {
      const float4 left{accum[i][0], accum[i][1], accum[i][2], accum[i][3]};
      const float4 right{accum[i][4], accum[i][5], accum[i][6], accum[i][7]};
      *reinterpret_cast<float4*>(C + (size_t)row * N + global_c_col0) = left;
      *reinterpret_cast<float4*>(C + (size_t)row * N + global_c_col0 + 4) =
          right;
    } else {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        if (global_c_col0 + j < N) {
          C[(size_t)row * N + global_c_col0 + j] = accum[i][j];
        }
      }
    }
  }
}

void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                 cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  cuda_core_gemm_annotated<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

#ifdef GEMM_DEMO_INLINE_PTX_FMA
#define KERNEL_NAME "cuda_core_gemm_annotated_inline_ptx_fma"
#else
#define KERNEL_NAME "cuda_core_gemm_annotated"
#endif
#include "gemm_entrypoint.cuh"
