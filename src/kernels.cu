#include <vector>
#include <stdexcept>
#include <cmath>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

constexpr int kRmsThreads = 256;
constexpr int kAttentionMaxThreads = 1024;

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half_rn(value);
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  const size_t offset = row * hidden_dim;

  float square_sum = 0.0f;
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const float value = toFloat(input[offset + col]);
    square_sum += value * value;
  }

  __shared__ float partial[kRmsThreads];
  partial[threadIdx.x] = square_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }

  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const float value = toFloat(input[offset + col]);
    output[offset + col] = fromFloat<T>(value * inv_rms * toFloat(weight[col]));
  }
}

template <typename T>
__global__ void flashAttentionKernel(const T* q, const T* k, const T* v, T* output,
                                     int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads, int head_dim,
                                     bool is_causal, float scale) {
  const int query_head = blockIdx.x % query_heads;
  const int query_pos = (blockIdx.x / query_heads) % target_seq_len;
  const int batch = blockIdx.x / (query_heads * target_seq_len);
  const int kv_head = query_head * kv_heads / query_heads;
  const int lane = threadIdx.x;

  const size_t q_offset =
      (static_cast<size_t>(batch) * target_seq_len * query_heads +
       static_cast<size_t>(query_pos) * query_heads + query_head) * head_dim;
  const size_t output_offset = q_offset;

  // Scores are recomputed for the max/sum/output passes instead of
  // materializing an O(sequence_length^2) score matrix.  A single lane uses
  // a fixed accumulation order for QK, which keeps the float result stable
  // across arbitrary head dimensions.
  float accumulator = 0.0f;
  __shared__ float shared_max;
  __shared__ float shared_sum;
  __shared__ float shared_score;

  if (lane == 0) {
    shared_max = -INFINITY;
    for (int key_pos = 0; key_pos < src_seq_len; ++key_pos) {
      if (is_causal && key_pos > query_pos) break;
      const size_t kv_offset =
          (static_cast<size_t>(batch) * src_seq_len * kv_heads +
           static_cast<size_t>(key_pos) * kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(toFloat(q[q_offset + dim]), toFloat(k[kv_offset + dim]), dot);
      }
      shared_max = fmaxf(shared_max, dot * scale);
    }

    shared_sum = 0.0f;
    for (int key_pos = 0; key_pos < src_seq_len; ++key_pos) {
      if (is_causal && key_pos > query_pos) break;
      const size_t kv_offset =
          (static_cast<size_t>(batch) * src_seq_len * kv_heads +
           static_cast<size_t>(key_pos) * kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(toFloat(q[q_offset + dim]), toFloat(k[kv_offset + dim]), dot);
      }
      shared_sum += expf(dot * scale - shared_max);
    }
  }
  __syncthreads();

  for (int key_pos = 0; key_pos < src_seq_len; ++key_pos) {
    if (is_causal && key_pos > query_pos) break;
    const size_t kv_offset =
        (static_cast<size_t>(batch) * src_seq_len * kv_heads +
         static_cast<size_t>(key_pos) * kv_heads + kv_head) * head_dim;
    if (lane == 0) {
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(toFloat(q[q_offset + dim]), toFloat(k[kv_offset + dim]), dot);
      }
      shared_score = dot * scale;
    }
    __syncthreads();
    const float probability = expf(shared_score - shared_max) / shared_sum;
    if (lane < head_dim) {
      accumulator = fmaf(probability, toFloat(v[kv_offset + lane]), accumulator);
    }
    __syncthreads();
  }

  if (lane < head_dim) {
    output[output_offset + lane] = fromFloat<T>(accumulator);
  }
}

int attentionThreads(int head_dim) {
  int threads = 1;
  while (threads < head_dim && threads < kAttentionMaxThreads) threads <<= 1;
  return threads;
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  if (rows == 0 || hidden_dim == 0) return;
  h_output.resize(rows * hidden_dim);

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  const size_t input_bytes = rows * hidden_dim * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
  RUNTIME_CHECK(cudaMalloc(&d_input, input_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_weight, weight_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_output, input_bytes));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

  rmsNormKernel<T><<<static_cast<unsigned int>(rows), kRmsThreads>>>(
      d_input, d_weight, d_output, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_input));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  if (batch_size == 0 || target_seq_len == 0 || src_seq_len == 0 ||
      query_heads == 0 || kv_heads == 0 || head_dim == 0) return;
  if (query_heads % kv_heads != 0 || head_dim > kAttentionMaxThreads) {
    throw std::invalid_argument("flashAttention requires query_heads divisible by kv_heads and head_dim <= 1024");
  }
  const size_t q_elements = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_elements = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  h_o.resize(q_elements);

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_elements * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_elements * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_elements * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_elements * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elements * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elements * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elements * sizeof(T), cudaMemcpyHostToDevice));

  const int threads = attentionThreads(head_dim);
  const int blocks = batch_size * target_seq_len * query_heads;
  flashAttentionKernel<T><<<blocks, threads>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, is_causal, 1.0f / sqrtf(static_cast<float>(head_dim)));
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_elements * sizeof(T), cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_o));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_q));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
