/**
 * @brief Fused MoE gate matmul + softmax + top-k.
 *
 * Replaces two launches per MoE layer:
 *   1. logits = gate.forward(x)   (F32 SGEMV via cublas)
 *   2. (weights, ids) = topk_softmax(logits)
 * with a single CUDA launch.
 *
 * Per token (one CUDA block, 32 threads / one warp):
 *   1. Load x[hidden] into shared memory (cooperatively).
 *   2. Each thread computes ceil(n_experts/32) logits, each = dot(x, W_row).
 *   3. Run the same warp-local softmax + iterative argmax + optional renorm
 *      as `topk_softmax_kernel` (kept inline to avoid template gymnastics).
 *
 * Layout:
 *   x:        [hidden]                    F32
 *   gate_w:   [n_experts, hidden]         F32 (row-major)
 *   weights:  [n_expert_used]             F32   (output)
 *   ids:      [n_expert_used]             u32   (output)
 *
 * Constraints:
 *   - hidden must be a multiple of 4 (we use float4 loads).
 *   - n_experts ∈ {32, 64, 128, 256}.
 *   - Single token per block (n_rows is a sweep dim, but typical use is
 *     per-token inside the MoE forward).
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32
#define MAX_HIDDEN_FOR_FUSED_GATE 8192

namespace ll_gate_topk {

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

template <int n_experts, bool with_norm>
__launch_bounds__(WARP_SIZE, 4)
__global__ void gate_topk_softmax_kernel(
    const float * __restrict__ x_in,        // [n_rows, hidden]
    const float * __restrict__ gate_w,      // [n_experts, hidden]
    float       * __restrict__ weights_out, // [n_rows, n_expert_used]
    uint32_t    * __restrict__ ids_out,     // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_expert_used
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float x_shared[];

    const int lane = threadIdx.x;

    // Cooperative load of x[hidden] into shared memory. Use float4 if hidden
    // is divisible by 4, otherwise fallback to scalar.
    const float * x_row = x_in + (size_t)row * hidden;
    if ((hidden & 3) == 0) {
        const float4 * x4 = reinterpret_cast<const float4 *>(x_row);
        float4       * s4 = reinterpret_cast<float4 *>(x_shared);
        const int hidden4 = hidden >> 2;
        for (int i = lane; i < hidden4; i += WARP_SIZE) {
            s4[i] = x4[i];
        }
    } else {
        for (int i = lane; i < hidden; i += WARP_SIZE) {
            x_shared[i] = x_row[i];
        }
    }
    __syncwarp();

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    // Compute per-thread logits: each lane handles experts_per_thread experts.
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        if (n_experts % WARP_SIZE == 0 || expert < n_experts) {
            const float * w_row = gate_w + (size_t)expert * hidden;
            float acc = 0.f;
            // Vectorized dot product: float4 chunks.
            if ((hidden & 3) == 0) {
                const float4 * w4 = reinterpret_cast<const float4 *>(w_row);
                const float4 * s4 = reinterpret_cast<const float4 *>(x_shared);
                const int hidden4 = hidden >> 2;
                #pragma unroll 4
                for (int k = 0; k < hidden4; k++) {
                    float4 a = w4[k];
                    float4 b = s4[k];
                    acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
            } else {
                for (int k = 0; k < hidden; k++) {
                    acc += w_row[k] * x_shared[k];
                }
            }
            wt[i] = acc;
        } else {
            wt[i] = -INFINITY;
        }
    }

    // ── Softmax (in-place over wt[]) ────────────────────────────────
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = (n_experts % WARP_SIZE == 0) || (idx < n_experts);
        if (active) max_val = fmaxf(max_val, wt[i]);
    }
    max_val = warp_reduce_max(max_val);

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
        const bool active = (n_experts % WARP_SIZE == 0) || (idx < n_experts);
        if (active) {
            const float v = __expf(wt[i] - max_val);
            wt[i] = v;
            sum += v;
        } else {
            wt[i] = 0.f;
        }
    }
    sum = warp_reduce_sum(sum);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] *= inv_sum;
    }

    // ── Iterative argmax to extract top-k ───────────────────────────
    float    output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] = 0.f;

    float * weights_row = weights_out + (size_t)row * n_expert_used;
    uint32_t * ids_row  = ids_out + (size_t)row * n_expert_used;

    float wt_sum = 0.f;
    for (int k = 0; k < n_expert_used; k++) {
        float max_v   = wt[0];
        int   max_exp = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_v) {
                max_v = wt[i];
                max_exp = expert;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float v_o = __shfl_xor_sync(0xFFFFFFFFu, max_v,   mask, WARP_SIZE);
            const int   e_o = __shfl_xor_sync(0xFFFFFFFFu, max_exp, mask, WARP_SIZE);
            if (v_o > max_v || (v_o == max_v && e_o < max_exp)) {
                max_v = v_o;
                max_exp = e_o;
            }
        }
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_v;
        }
        if ((max_exp & (WARP_SIZE - 1)) == lane) {
            wt[max_exp / WARP_SIZE] = -INFINITY;
            ids_row[k] = (uint32_t)max_exp;
            if constexpr (with_norm) {
                wt_sum += max_v;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum = warp_reduce_sum(wt_sum);
        const float inv_wt_sum = 1.0f / wt_sum;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_wt_sum;
        }
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights_row[idx] = output_weights[i];
        }
    }
}

} // namespace ll_gate_topk

extern "C" void gate_topk_softmax(
    const float * x_in,         // [n_rows, hidden] device ptr
    const float * gate_w,       // [n_experts, hidden] device ptr
    float       * weights_out,  // [n_rows, n_expert_used]
    uint32_t    * ids_out,      // [n_rows, n_expert_used]
    int hidden,
    int n_rows,
    int n_experts,
    int n_expert_used,
    int with_norm,
    cudaStream_t stream
) {
    if (hidden > MAX_HIDDEN_FOR_FUSED_GATE) {
        return; // Caller falls back to unfused path.
    }
    dim3 grid(n_rows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);
    int shared_bytes = hidden * (int)sizeof(float);

#define DISPATCH(NE) \
    do { \
        if (with_norm) ll_gate_topk::gate_topk_softmax_kernel<NE, true>  <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, weights_out, ids_out, hidden, n_rows, n_expert_used); \
        else           ll_gate_topk::gate_topk_softmax_kernel<NE, false> <<<grid, block, shared_bytes, stream>>>(x_in, gate_w, weights_out, ids_out, hidden, n_rows, n_expert_used); \
    } while (0)

    switch (n_experts) {
        case 32:  DISPATCH(32);  break;
        case 64:  DISPATCH(64);  break;
        case 128: DISPATCH(128); break;
        case 256: DISPATCH(256); break;
        default:
            return; // Unsupported — caller falls back.
    }
#undef DISPATCH
}
