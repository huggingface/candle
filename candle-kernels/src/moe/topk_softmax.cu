/**
 * @brief Fused MoE routing kernel: softmax → top-k → optional renorm.
 *
 * Replaces the candle ops sequence
 *   softmax_last_dim → arg_sort_last_dim → narrow → contiguous →
 *   gather → sum_keepdim → broadcast_div
 * with a SINGLE CUDA launch.
 *
 * Per token we run on one warp:
 *   1. Each lane holds `experts_per_thread = ceil(n_experts / 32)` logits.
 *   2. Warp-local softmax (max, expf, warp_reduce_sum, divide).
 *   3. Iterate `n_expert_used` (top-k) times, each iteration finds the
 *      argmax across the warp via __shfl_xor reductions, records the
 *      (id, weight) pair and clobbers the picked logit with -INFINITY.
 *   4. If `with_norm`, divide the recorded weights by their sum.
 *
 * Adapted from llama.cpp's `topk_moe_cuda` (`ggml-cuda/topk-moe.cu`)
 * with two changes for candle compatibility:
 *   • output `ids` is u32 (candle's gather-arg dtype) rather than int32.
 *   • single fixed instantiation per supported (n_experts, with_norm)
 *     pair — see the `extern "C"` launchers below.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask, WARP_SIZE);
    }
    return v;
}

template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], int limit, int lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) max_val = fmaxf(max_val, vals[i]);
    }
    max_val = warp_reduce_max(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float v = __expf(vals[i] - max_val);
            vals[i] = v;
            sum += v;
        } else {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum(sum);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        vals[i] *= inv_sum;
    }
}

// Generic kernel — n_experts and with_norm are template parameters so
// the compiler can unroll inner loops at the right size.
template <int n_experts, bool with_norm>
__launch_bounds__(4 * WARP_SIZE, 1)
__global__ void topk_softmax_kernel(
    const float * __restrict__ logits,    // [n_rows, n_experts]
    float       * __restrict__ weights,   // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids,       // [n_rows, n_expert_used]
    int n_rows,
    int n_expert_used
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;

    logits  += (size_t)n_experts      * row;
    weights += (size_t)n_expert_used  * row;
    ids     += (size_t)n_expert_used  * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts)
            ? logits[expert]
            : -INFINITY;
    }

    softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);

    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] = 0.f;

    float wt_sum = 0.f;

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFFu, max_val,    mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFFu, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
            ids[k] = (uint32_t)max_expert;
            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum = warp_reduce_sum(wt_sum);
        const float inv_sum = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }
}

// Host launcher — picks the right (n_experts, with_norm) instantiation.
// Currently supports n_experts ∈ {32, 64, 128, 256} which covers every
// MoE model we care about (qwen3-coder=128, qwen3.5=256, others typical).
extern "C" void topk_softmax(
    const float* logits,
    float* weights,
    uint32_t* ids,
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    cudaStream_t stream
) {
    constexpr int rows_per_block = 4;
    dim3 grid((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3 block(WARP_SIZE, rows_per_block, 1);

#define DISPATCH(NE) \
    do { \
        if (with_norm) topk_softmax_kernel<NE, true>  <<<grid, block, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used); \
        else           topk_softmax_kernel<NE, false> <<<grid, block, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH(32);  break;
        case 64:  DISPATCH(64);  break;
        case 128: DISPATCH(128); break;
        case 256: DISPATCH(256); break;
        default:
            // Unsupported — caller must fall back to the unfused candle ops.
            break;
    }
#undef DISPATCH
}
