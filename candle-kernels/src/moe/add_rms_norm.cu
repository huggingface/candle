/**
 * @brief Fused (attn_out + residual) + RMSNorm for single-token decode.
 *
 * Replaces the two separate launches:
 *   xs       = attn_out + residual              // 1 launch (broadcast_add)
 *   x_normed = rms_norm(xs, gamma, eps)         // 1 launch
 * with a single kernel that produces BOTH outputs:
 *   xs       (attn residual + attn out — passed to MLP as residual)
 *   x_normed (gamma * xs * rsqrt(mean(xs^2) + eps) — input to MLP)
 *
 * Saves 1 launch per layer per token (~5–7 µs CPU overhead × 48 layers
 * × decode tok rate). On qwen3-coder this is ~240 µs/token = ~3%.
 *
 * Single-block design: hidden ≤ 8192 fits in 1024 threads × 8 elements.
 * Each thread loads `elements_per_thread` consecutive floats, accumulates
 * sum-of-squares, warp-reduces, then writes back the normalized output.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace ll_addnorm {

template <int block_size>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[block_size / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    // Warp reduce.
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, mask, WARP_SIZE);
    }
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    // First warp reduces warp sums.
    if (wid == 0) {
        val = lane < (block_size / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            val += __shfl_xor_sync(0xFFFFFFFFu, val, mask, WARP_SIZE);
        }
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();
    return warp_sums[0];
}

template <int block_size>
__global__ void add_rms_norm_kernel(
    const float* __restrict__ a,        // [hidden] attn_out
    const float* __restrict__ b,        // [hidden] residual
    const float* __restrict__ gamma,    // [hidden] norm weights
    float* __restrict__ xs_out,         // [hidden] = a + b
    float* __restrict__ normed_out,     // [hidden] = gamma * (a+b) / rms
    int hidden,
    float eps
) {
    const int tid = threadIdx.x;
    const int elements_per_thread = (hidden + block_size - 1) / block_size;

    // Pass 1: load + add + sum-of-squares.
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int idx = tid + i * block_size;
        if (i >= elements_per_thread || idx >= hidden) break;
        const float ai = a[idx];
        const float bi = b[idx];
        const float si = ai + bi;
        xs_out[idx] = si;
        sum_sq += si * si;
    }

    const float total_sq = block_reduce_sum<block_size>(sum_sq);
    const float inv_rms = rsqrtf(total_sq / (float)hidden + eps);

    // Pass 2: read back xs, scale, write normed.
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int idx = tid + i * block_size;
        if (i >= elements_per_thread || idx >= hidden) break;
        const float si = xs_out[idx];
        normed_out[idx] = si * inv_rms * gamma[idx];
    }
}

} // namespace ll_addnorm

extern "C" void add_rms_norm(
    const float* a,         // [hidden]
    const float* b,         // [hidden]
    const float* gamma,     // [hidden]
    float* xs_out,          // [hidden]
    float* normed_out,      // [hidden]
    int hidden,
    float eps,
    cudaStream_t stream
) {
    // Pick block size based on hidden dim. 1024 threads handles hidden up to
    // 1024 * 32 = 32768 (way more than needed). 512 is plenty for most models.
    constexpr int block_size = 512;
    dim3 grid(1, 1, 1);
    dim3 block(block_size, 1, 1);
    ll_addnorm::add_rms_norm_kernel<block_size>
        <<<grid, block, 0, stream>>>(a, b, gamma, xs_out, normed_out, hidden, eps);
}
