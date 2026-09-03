#include "gemm_common.cuh"
constexpr int BM = 64, BN = 64, BK = 8;
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int WM = BM / WARPS_M; // 32
constexpr int WN = BN / WARPS_N; // 32
constexpr int TM = 4, TN = 8;
constexpr int THREADS = 128; // 4 warps

__global__ void gemm_v7_warptiling(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
  // +1 padding breaks the row-stride/bank periodicity for the A tile.
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int warp_m = warp / WARPS_N;
  int warp_n = warp % WARPS_N;
  int lane_m = lane / 4; // 0..7
  int lane_n = lane % 4; // 0..3
  int lr0 = warp_m * WM + lane_m * TM;
  int lc0 = warp_n * WN + lane_n * TN;
  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Exactly 128 float4 chunks in A tile and 128 in B tile: one of each per thread.
    int av = tid;
    int ar = av / (BK / 4), avc = av % (BK / 4);
    int agr = blockIdx.y * BM + ar, agk = k0 + avc * 4;
    if (agr < M && agk + 3 < K && (K % 4 == 0)) {
      float4 x = *reinterpret_cast<const float4*>(&A[(size_t)agr * K + agk]);
      As[ar][avc*4+0] = x.x; As[ar][avc*4+1] = x.y;
      As[ar][avc*4+2] = x.z; As[ar][avc*4+3] = x.w;
    } else {
#pragma unroll
      for (int q=0;q<4;++q) As[ar][avc*4+q] = (agr < M && agk+q < K) ? A[(size_t)agr*K+agk+q] : 0.0f;
    }

    int bv = tid;
    int bk = bv / (BN / 4), bvc = bv % (BN / 4);
    int bgk = k0 + bk, bgc = blockIdx.x * BN + bvc * 4;
    if (bgk < K && bgc + 3 < N && (N % 4 == 0)) {
      *reinterpret_cast<float4*>(&Bs[bk][bvc*4]) = *reinterpret_cast<const float4*>(&B[(size_t)bgk*N+bgc]);
    } else {
#pragma unroll
      for (int q=0;q<4;++q) Bs[bk][bvc*4+q] = (bgk < K && bgc+q < N) ? B[(size_t)bgk*N+bgc+q] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float arx[TM], brx[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) arx[i] = As[lr0 + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) brx[j] = Bs[k][lc0 + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(arx[i], brx[j], acc[i][j]);
    }
    __syncthreads();
  }

  int gr0 = blockIdx.y * BM + lr0;
  int gc0 = blockIdx.x * BN + lc0;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int r = gr0 + i;
    if (r >= M) continue;
    if (gc0 + TN <= N && (N % 4 == 0)) {
      float4 o0{acc[i][0],acc[i][1],acc[i][2],acc[i][3]};
      float4 o1{acc[i][4],acc[i][5],acc[i][6],acc[i][7]};
      *reinterpret_cast<float4*>(&C[(size_t)r*N+gc0]) = o0;
      *reinterpret_cast<float4*>(&C[(size_t)r*N+gc0+4]) = o1;
    } else {
#pragma unroll
      for (int j=0;j<TN;++j) if (gc0+j<N) C[(size_t)r*N+gc0+j] = acc[i][j];
    }
  }
}

void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v7_warptiling<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#define KERNEL_NAME "v7_warptiling"
#include "gemm_entrypoint.cuh"
