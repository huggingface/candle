// Derived from vLLM (Apache-2.0): https://github.com/vllm-project/vllm

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define WARP_SIZE 32

namespace {

template <typename T> __host__ __device__ constexpr T ceil_div(T x, T y) {
  return x / y + static_cast<T>(x % y != 0);
}

__device__ void moe_align_block_size(
    const int32_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks) {
  extern __shared__ int32_t shared_counts[];

  if (blockIdx.x % 2) {
    for (size_t i = threadIdx.x; i < static_cast<size_t>(max_num_tokens_padded);
         i += blockDim.x) {
      sorted_token_ids[i] = static_cast<int32_t>(numel);
    }
    return;
  }

  if (threadIdx.x < padded_num_experts) {
    shared_counts[threadIdx.x] = 0;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int expert_id = topk_ids[i];
    if (expert_id >= 0 && expert_id < num_experts) {
      const int warp_idx = expert_id / experts_per_warp;
      const int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
  }
  __syncthreads();

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  const int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    const int warp_idx = expert_id / experts_per_warp;
    const int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = ceil_div(expert_count, block_size) * block_size;
  }

  int cumsum_value;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_value);
  if (expert_id <= num_experts) {
    cumsum[expert_id] = cumsum_value;
  }
  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = cumsum_value;
  }
  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  const size_t fill_start = cumsum[num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start; i < static_cast<size_t>(max_num_m_blocks);
       i += blockDim.x) {
    expert_ids[i] = -1;
  }
}

__global__ void moe_align_block_size_kernel(
    const int32_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum, int32_t max_num_tokens_padded) {
  moe_align_block_size(topk_ids, sorted_token_ids, expert_ids,
                       total_tokens_post_pad, num_experts, padded_num_experts,
                       experts_per_warp, block_size, numel, cumsum,
                       max_num_tokens_padded,
                       ceil_div(max_num_tokens_padded, block_size));
}

__global__ void
count_and_sort_expert_tokens_kernel(const int32_t *__restrict__ topk_ids,
                                    int32_t *__restrict__ sorted_token_ids,
                                    int32_t *__restrict__ cumsum, size_t numel,
                                    int32_t num_experts) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;
  for (size_t i = tid; i < numel; i += stride) {
    const int32_t expert_id = topk_ids[i];
    if (expert_id >= 0 && expert_id < num_experts) {
      const int32_t rank_post_pad = atomicAdd(&cumsum[expert_id], 1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    }
  }
}

} // namespace

extern "C" int
candle_launch_moe_align(const int32_t *topk_ids, int32_t *sorted_token_ids,
                        int32_t *expert_ids, int32_t *num_tokens_post_pad,
                        int32_t *cumsum, int32_t num_experts,
                        int32_t block_size, int32_t numel,
                        int32_t max_num_tokens_padded, cudaStream_t stream) {
  const int32_t padded_num_experts =
      ceil_div(num_experts, WARP_SIZE) * WARP_SIZE;
  const int experts_per_warp = WARP_SIZE;
  const int threads = 1024;
  const size_t num_warps = ceil_div(padded_num_experts, experts_per_warp);
  const size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

  moe_align_block_size_kernel<<<2, threads, shared_mem_size, stream>>>(
      topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts,
      padded_num_experts, experts_per_warp, block_size,
      static_cast<size_t>(numel), cumsum, max_num_tokens_padded);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  const int block_threads = 256;
  const int n_blocks = ceil_div(numel, block_threads);
  const int actual_blocks = n_blocks < 65535 ? n_blocks : 65535;
  const dim3 grid_dims(1, actual_blocks);
  count_and_sort_expert_tokens_kernel<<<grid_dims, block_threads, 0, stream>>>(
      topk_ids, sorted_token_ids, cumsum, static_cast<size_t>(numel),
      num_experts);
  return static_cast<int>(cudaGetLastError());
}
