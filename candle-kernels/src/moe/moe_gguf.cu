/**
 * @brief CUDA kernel for Mixture-of-Experts (MoE) GEMM using GGUF quantized weights.
 *
 * This kernel performs a dot-product between quantized input tokens and
 * quantized expert weight matrices, accumulating into float outputs.
 * It supports per-token top-k weighting and tiling along the K dimension
 * for efficient vectorized execution.
 *
 * Adapted from: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_gguf.cu
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cassert>
constexpr int MATRIX_ROW_PADDING = 512;

constexpr int pad(int size, int padding) {
    if (padding == 0) return size;  // avoid divide-by-zero
    return ((size + padding - 1) / padding) * padding;
}

// Optional helper if you want ceil division explicitly
constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

namespace vllm_rs {

/*
* Template Parameters:
 * @tparam T                 Type of output elements (float, half, etc.)
 * @tparam qk                Quantization block size for weights (e.g., 32)
 * @tparam qi                Quantization block size for inputs (e.g., 32)
 * @tparam block_q_t         Type of quantized weight block (e.g., block_q8_0)
 * @tparam vdr               Vectorization factor (number of elements per lane)
 * @tparam vec_dot_q_cuda    Function for computing vectorized dot-product between quantized blocks
 *
 * Kernel Parameters:
 * @param all_weights         Pointer to all expert weight matrices, [num_experts, N, K] (quantized)
 * @param all_inputs          Pointer to all input tokens, [M_total, K] (quantized)
 * @param sorted_token_ids    Sorted token indices for batch processing
 * @param expert_ids          Expert ID for each token
 * @param topk_weights        Optional top-k MoE weight per token
 * @param all_outputs         Output buffer [M_total, N] (float)
 * @param num_experts         Number of experts
 * @param topk                Top-k experts selected per token
 * @param size_m              Number of tokens processed (M dimension)
 * @param size_n              Output feature dimension (N dimension)
 * @param size_k              Input feature dimension (K dimension)
 * @param k_padded            Padded K dimension for GGUF stride
*/
// Multi-row MMVQ kernel — each warp computes ROWS_PER_WARP output rows
// for the same (token, expert). The input quantized vector (`y_ptr`) is
// shared across rows — loaded once via L1/L2 and reused by all
// `ROWS_PER_WARP` dot products inside the warp's inner loop. Halves
// (or quarters) the grid size compared to the 1-row-per-warp version
// while keeping the dot-product kernel structure unchanged.
//
// Pattern adapted from llama.cpp's `mul_mat_vec_q` `tmp[ncols_dst][rows_per_cuda_block]`
// accumulator. We keep ncols_dst==1 (each block still maps to one m_idx
// = one (token, expert)) since MoE expert routing makes m_idx values
// route to different experts, so weights can't be shared across m_idx.
// What we CAN share is the input across multiple output rows — that's
// what ROWS_PER_WARP does.
template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void moe_gemm_gguf_kernel(
    const void * __restrict__ all_weights,       // [num_experts, N, K] (quantized)
    const void * __restrict__ all_inputs,        // [M_total, K] (quantized, M_total is total tokens)
    const int32_t* __restrict__ sorted_token_ids,// [M] (M = num tokens processed)
    const int32_t* __restrict__ expert_ids,      // [M]
    const float* __restrict__ topk_weights,      // [M]
    float * __restrict__ all_outputs,            // [M_total, N] (float)
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k, // M, N, K are the logical dims
    int k_padded // Padded K-dim for GGUF stride
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    // First output row this warp computes. Block computes
    // nWraps * ROWS_PER_WARP rows; warp computes ROWS_PER_WARP
    // contiguous rows starting here.
    const int row0 = blockIdx.x * nWraps * ROWS_PER_WARP + wrapId * ROWS_PER_WARP;
    const int m_idx = blockIdx.y;

    if (row0 >= size_n || m_idx >= size_m) {
        return;
    }

    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx];
    const int expert = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    const float scale = (topk_weights) ? topk_weights[token_id] : 1.0f;

    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)all_weights + (size_t)expert * weight_expert_stride_bytes);

    const int input_index = topk_weights ? token_id : (token_id / topk);
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi;

    // Shared memory holds nWraps × ROWS_PER_WARP weight rows. Each warp
    // loads its ROWS_PER_WARP contiguous rows.
    extern __shared__ int8_t shared_bytes[];
    block_q_t* w_shared = reinterpret_cast<block_q_t*>(shared_bytes);
    block_q_t* w_shared_warp = w_shared + (size_t)wrapId * ROWS_PER_WARP * blocks_per_row_x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < size_n) {
            for (int i = laneId; i < blocks_per_row_x; i += WARP_SIZE) {
                w_shared_warp[r * blocks_per_row_x + i] = w_expert[row * blocks_per_row_x + i];
            }
        }
    }
    __syncthreads();

    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) acc[r] = 0.0f;

    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        // Same y_ptr block serves all ROWS_PER_WARP dot products — it
        // hits L1/L2 once and stays cached across the unrolled loop.
        const block_q8_1 * y_blk = &y_ptr[kby];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            acc[r] += vec_dot_q_cuda(
                &w_shared_warp[r * blocks_per_row_x + kbx],
                y_blk,
                kqs);
        }
    }

    float * __restrict__ out_ptr =
        all_outputs + ((size_t)token_id) * output_task_stride_elems;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < size_n) {
            const float v = warp_reduce_sum(acc[r]) * scale;
            if (laneId == 0) {
                out_ptr[row] = v;
            }
        }
    }
}

}

#define LAUNCH_MOE_GGUF(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    /* Shared mem: nWraps × ROWS_PER_WARP weight rows × bytes-per-row */ \
    const int shared_bytes = size_k / qk * sizeof(block_q_t) * nWraps * ROWS_PER_WARP + 1024;\
    vllm_rs::moe_gemm_gguf_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<grid_dim, block_dim, shared_bytes, stream>>>(\
        weights, y_q8_1,\
        sorted_token_ids, expert_ids, topk_weights,\
        outputs,\
        num_experts, topk,\
        size_m, size_n, size_k,\
        kx_padded\
    );\


extern "C" void moe_gemm_gguf(
    const float* inputs, //must be float
    const void* weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    float* outputs,
    int num_experts,
    int topk,
    int size_m,         // M (num tokens to process)
    int size_n,         // N (output dim)
    int size_k,         // K (input dim)
    int quant_type,     // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    int m = topk_weights ? size_m : size_m / topk;
    dim3 grid_dim_quant(num_blocks, m, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes =
        m * (kx_padded / QK8_1 * sizeof(block_q8_1));
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    // ROWS_PER_WARP > 1 doubles the per-block shared-memory cost
    // (each warp now caches that many weight rows). On Blackwell the
    // shared-mem-per-block ceiling clamps blocks-resident-per-SM to 1
    // for ROWS_PER_WARP=2 with K=2048 K-quants, which underutilizes
    // SMs when grid_x is small (e.g. moe_intermediate=768 → grid_x=96).
    // Keep =1; experiment with larger only when matmul shape favors it.
    const int nWraps = 4;
    constexpr int ROWS_PER_WARP = 1;
    dim3 grid_dim(ceil_div(size_n, nWraps * ROWS_PER_WARP), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
    switch (quant_type) {
        case 0: // Q8_0
        {
            LAUNCH_MOE_GGUF(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
            break;
        }
        case 1: // Q4K
        {
            LAUNCH_MOE_GGUF(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
            break;
        }
        case 2: // Q2_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
            break;
        }
        case 3: // Q3_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
            break;
        }
        case 4: // Q5_K
        {
            LAUNCH_MOE_GGUF(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
            break;
        }
        case 5: // Q6K
        {
            LAUNCH_MOE_GGUF(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
            break;
        }
        default:
            break;
    }
    cudaFreeAsync(y_q8_1, stream);
}

// ─────────────────────────────────────────────────────────────────────────
// MoE down-projection GEMM with topk reduction fused inline.
//
// Replaces the sequence
//   ys = moe_gemm_gguf(down_inputs, down_w, topk_weights)  // [M*topk, hidden]
//   ys = ys.reshape((M, topk, hidden))?.sum(D::Minus2)?    // [M, hidden]
// with a single CUDA launch that writes scaled partial results directly
// to a pre-zeroed [M, hidden] output via atomicAdd. Saves the explicit
// sum() launch and the [M*topk, hidden] intermediate.
//
// For decode (M=1 real token, topk=8): each (token, expert) pair adds
// its weighted contribution to the same output row. atomicAdd on F32
// has minimal contention at this scale (8 contributors per output
// position scattered across hidden=2048 lanes).
// ─────────────────────────────────────────────────────────────────────────
namespace vllm_rs {

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_gemm_gguf_down_reduce_kernel(
    const void * __restrict__ all_weights,       // [num_experts, N=hidden, K=moe_inter]
    const void * __restrict__ all_inputs,        // [M_total, K]
    const int32_t* __restrict__ sorted_token_ids,// [M]
    const int32_t* __restrict__ expert_ids,      // [M]
    const float*  __restrict__ topk_weights,     // [M_total]
    float * __restrict__ all_outputs,            // [num_real_tokens, N]
                                                 // pre-initialized to either zeros (alloc_zeros)
                                                 // or to residual (cast F16→F32) — caller's choice.
                                                 // Kernel just atomicAdds its contribution.
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k,
    int k_padded
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int row = blockIdx.x * nWraps + wrapId;
    const int m_idx = blockIdx.y;

    if (row >= size_n || m_idx >= size_m) return;

    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx];
    const int expert   = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    const float scale = topk_weights[token_id];

    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)all_weights + (size_t)expert * weight_expert_stride_bytes);

    // Down's input is per-(token, expert) — input_index is just token_id.
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)token_id * input_task_stride_bytes);

    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi;

    // No shared-mem weight cache (reads from global; L1 handles reuse).
    float acc = 0.0f;
    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        acc += vec_dot_q_cuda(
            &w_expert[row * blocks_per_row_x + kbx],
            &y_ptr[kby],
            kqs);
    }

    const float v = warp_reduce_sum(acc) * scale;
    if (laneId == 0) {
        const int real_token = token_id / topk;
        float * __restrict__ out_ptr =
            all_outputs + ((size_t)real_token) * output_task_stride_elems;
        atomicAdd(&out_ptr[row], v);
    }
}

} // namespace vllm_rs

#define LAUNCH_MOE_GGUF_DOWN_REDUCE(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    /* No shared-mem weight cache. */ \
    const int shared_bytes_dr = 0; \
    vllm_rs::moe_gemm_gguf_down_reduce_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<grid_dim, block_dim, shared_bytes_dr, stream>>>( \
        weights, y_q8_1, \
        sorted_token_ids, expert_ids, topk_weights, \
        outputs, \
        num_experts, topk, \
        size_m, size_n, size_k, \
        kx_padded \
    );

// Initialize an F32 output buffer from an F16 residual. Used to fuse
// the post-MLP residual add into the moe_gemm_gguf_down_reduce kernel:
// instead of alloc_zeros + atomicAdd contributions + later add_residual,
// we init the buffer to residual values then atomicAdd contributions —
// the residual add is "free" (one initial cast + write per element).
__global__ void init_f32_from_f16(float* __restrict__ dst, const __half* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void init_f32_from_bf16(float* __restrict__ dst, const __nv_bfloat16* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

extern "C" void cast_init_f32_from_dtype(
    float* dst,
    const void* src,
    int n,
    int dtype,                      // 0=f16, 1=bf16
    cudaStream_t stream
) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    if (dtype == 0) {
        init_f32_from_f16<<<grid, block, 0, stream>>>(dst, (const __half*)src, n);
    } else {
#ifndef NO_BF16_KERNEL
        init_f32_from_bf16<<<grid, block, 0, stream>>>(dst, (const __nv_bfloat16*)src, n);
#endif
    }
}

extern "C" void moe_gemm_gguf_down_reduce(
    const float* inputs,            // [M*topk, K] f32
    const void*  weights,           // [num_experts, N, K] quantized
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,      // [M*topk] f32 — required (not optional)
    float* outputs,                 // [num_real_tokens, N] f32 — caller MUST pre-zero
    int num_experts,
    int topk,
    int size_m,                     // M*topk (entries to process)
    int size_n,                     // hidden
    int size_k,                     // moe_inter
    int quant_type,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    int m_in = size_m; // down has its own per-(token,expert) input
    dim3 grid_dim_quant(num_blocks, m_in, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes = m_in * (kx_padded / QK8_1 * sizeof(block_q8_1));
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    // Smaller blocks → more parallelism across SMs (no shared-mem
    // barrier means small blocks are fine). Down has size_n=hidden,
    // typically much larger than gate/up's size_n=moe_inter, so the
    // grid is already big at nWraps=4; leaving =2 for symmetry and
    // marginally better SM occupancy on small batches.
    const int nWraps = 2;
    dim3 grid_dim(ceil_div(size_n, nWraps), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    switch (quant_type) {
        case 0: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break; }
        case 1: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break; }
        case 2: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break; }
        case 3: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break; }
        case 4: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break; }
        case 5: { LAUNCH_MOE_GGUF_DOWN_REDUCE(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break; }
        default: break;
    }
    cudaFreeAsync(y_q8_1, stream);
}

// ─────────────────────────────────────────────────────────────────────────
// Fused gate+up MoE GEMM with SiLU activation and elementwise multiply.
//
// Computes:    output[m, n] = silu(dot(gate_w[expert][n], input[m])) *
//                                    dot(up_w[expert][n],   input[m])
//
// Compared to running moe_gemm_gguf twice (once for gate, once for up) and
// then a separate silu+mul elementwise kernel, this fused kernel:
//   • shares the quantize_q8_1 of the input (one launch instead of two —
//     the input was already shared, but the host-side kernel invocations
//     each did a separate quantize alloc),
//   • shares the *load* of `y_ptr` block bytes from L1/L2 across the gate
//     and up partial-sum loops (one trip through global memory for the
//     input vector, used twice),
//   • writes a single [M, N] output instead of two [M, N] intermediates +
//     one [M, N] final, halving global-memory write bandwidth and
//     eliminating the activation-and-multiply launches.
//
// llama.cpp's MMVQ does this via a `vgate` / `tmp_gate` template parameter;
// here we keep the existing kernel template and add a sibling that takes
// gate and up weights as separate pointers.
// ─────────────────────────────────────────────────────────────────────────
namespace vllm_rs {

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void moe_gemm_gguf_gate_up_silu_mul_kernel(
    const void * __restrict__ gate_weights,      // [num_experts, N, K] (quantized)
    const void * __restrict__ up_weights,        // [num_experts, N, K] (quantized)
    const void * __restrict__ all_inputs,        // [M_total, K] (quantized)
    const int32_t* __restrict__ sorted_token_ids,// [M]
    const int32_t* __restrict__ expert_ids,      // [M]
    float * __restrict__ all_outputs,            // [M_total, N] (float, fused output)
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k,
    int k_padded
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    // Each warp computes ROWS_PER_WARP contiguous output rows. Block
    // computes nWraps × ROWS_PER_WARP rows total.
    const int row0 = blockIdx.x * nWraps * ROWS_PER_WARP + wrapId * ROWS_PER_WARP;
    const int m_idx = blockIdx.y;

    if (row0 >= size_n || m_idx >= size_m) {
        return;
    }

    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx];
    const int expert   = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    const block_q_t * __restrict__ wg_expert =
        (const block_q_t *)((const char *)gate_weights + (size_t)expert * weight_expert_stride_bytes);
    const block_q_t * __restrict__ wu_expert =
        (const block_q_t *)((const char *)up_weights   + (size_t)expert * weight_expert_stride_bytes);

    // Gate/up don't apply topk_weights (only down does) — input_index is
    // the same as in the unfused gate/up kernel call: token_id / topk.
    const int input_index = token_id / topk;
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi;

    // No shared-mem weight cache. Reading weights directly from global
    // hits L1/L2 — for typical hidden sizes the working set fits and
    // the saved shared mem lets MORE blocks reside per SM, improving
    // occupancy. Pattern matches llama.cpp's MMVQ kernel.
    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }

    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        const block_q8_1 * y_blk = &y_ptr[kby];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            const int row = row0 + r;
            if (row < size_n) {
                acc_g[r] += vec_dot_q_cuda(&wg_expert[row * blocks_per_row_x + kbx], y_blk, kqs);
                acc_u[r] += vec_dot_q_cuda(&wu_expert[row * blocks_per_row_x + kbx], y_blk, kqs);
            }
        }
    }

    float * __restrict__ out_ptr =
        all_outputs + ((size_t)token_id) * output_task_stride_elems;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < size_n) {
            const float g = warp_reduce_sum(acc_g[r]);
            const float u = warp_reduce_sum(acc_u[r]);
            if (laneId == 0) {
                const float silu_g = g / (1.0f + __expf(-g));
                out_ptr[row] = silu_g * u;
            }
        }
    }
}

} // namespace vllm_rs

#define LAUNCH_MOE_GGUF_GATE_UP(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    /* Kernel reads weights from global; only need a tiny stub for the */ \
    /* dynamic-shared-mem extern declaration. */ \
    const int shared_bytes_gu = 0; \
    vllm_rs::moe_gemm_gguf_gate_up_silu_mul_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<grid_dim, block_dim, shared_bytes_gu, stream>>>( \
        gate_weights, up_weights, y_q8_1, \
        sorted_token_ids, expert_ids, \
        outputs, \
        num_experts, topk, \
        size_m, size_n, size_k, \
        kx_padded \
    );

extern "C" void moe_gemm_gguf_gate_up_silu_mul(
    const float* inputs,
    const void* gate_weights,
    const void* up_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    float* outputs,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int quant_type,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    int m_in = size_m / topk;
    dim3 grid_dim_quant(num_blocks, m_in, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes = m_in * (kx_padded / QK8_1 * sizeof(block_q8_1));
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    // Without shared-mem barriers we can use smaller blocks to spread
    // work across more SMs. For typical MoE intermediate sizes (768) at
    // nWraps=4 we'd only generate 192 blocks (1.5/SM on a 128-SM card).
    // nWraps=2 doubles block count to 384, ~3 blocks/SM — better.
    const int nWraps = 2;
    constexpr int ROWS_PER_WARP = 1;
    dim3 grid_dim(ceil_div(size_n, nWraps * ROWS_PER_WARP), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    switch (quant_type) {
        case 0: { LAUNCH_MOE_GGUF_GATE_UP(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break; }
        case 1: { LAUNCH_MOE_GGUF_GATE_UP(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break; }
        case 2: { LAUNCH_MOE_GGUF_GATE_UP(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break; }
        case 3: { LAUNCH_MOE_GGUF_GATE_UP(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break; }
        case 4: { LAUNCH_MOE_GGUF_GATE_UP(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break; }
        case 5: { LAUNCH_MOE_GGUF_GATE_UP(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break; }
        default: break;
    }
    cudaFreeAsync(y_q8_1, stream);
}

// ───────────────────────────────────────────────────────────────────────
// MoE gate||up GEMM with GELU(tanh) activation and elementwise multiply,
// for the gemma4-MoE concat weight layout where gate and up share one
// [num_experts, 2*N, K] tensor (gate = rows 0..N, up = rows N..2N).
//
// Replaces the gemma4 MoE FFN sequence:
//   gu = moe_gemm_gguf(input, gate_up_exps)         // [M*topk, 2N]
//   gate_act = gelu_tanh(gu[:, :N]); down_in = gate_act * gu[:, N:]
// with a single fused matmul that emits the activated [M*topk, N] output
// directly, saving the [2N] intermediate write + the activation+mul
// elementwise launches.
// ───────────────────────────────────────────────────────────────────────
namespace vllm_rs {

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void moe_gemm_gguf_gate_up_gelu_mul_concat_kernel(
    const void * __restrict__ gate_up_weights,   // [num_experts, 2*N, K] (quantized, gate||up)
    const void * __restrict__ all_inputs,        // [M_total, K] (quantized)
    const int32_t* __restrict__ sorted_token_ids,// [M]
    const int32_t* __restrict__ expert_ids,      // [M]
    float * __restrict__ all_outputs,            // [M_total, N] (gelu(gate) * up)
    int num_experts,
    int topk,
    int size_m, int size_n, int size_k,
    int k_padded
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int row0 = blockIdx.x * nWraps * ROWS_PER_WARP + wrapId * ROWS_PER_WARP;
    const int m_idx = blockIdx.y;

    if (row0 >= size_n || m_idx >= size_m) {
        return;
    }

    // 2*N rows per expert in the concat layout.
    const int two_n = size_n << 1;
    const size_t weight_expert_stride_bytes = (size_t)(two_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx];
    const int expert   = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    // Gate weights = rows [0..N), up weights = rows [N..2N) within this
    // expert slab. Both share the same expert-stride base pointer.
    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)gate_up_weights + (size_t)expert * weight_expert_stride_bytes);
    // Up half starts N rows in (= N * blocks_per_row blocks).
    const int blocks_per_row_x = size_k / qk;
    const block_q_t * __restrict__ wu_expert = w_expert + (size_t)size_n * blocks_per_row_x;

    const int input_index = token_id / topk;
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    const int blocks_per_iter = vdr * WARP_SIZE / qi;

    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }

    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        const block_q8_1 * y_blk = &y_ptr[kby];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            const int row = row0 + r;
            if (row < size_n) {
                acc_g[r] += vec_dot_q_cuda(&w_expert[row * blocks_per_row_x + kbx],   y_blk, kqs);
                acc_u[r] += vec_dot_q_cuda(&wu_expert[row * blocks_per_row_x + kbx], y_blk, kqs);
            }
        }
    }

    float * __restrict__ out_ptr =
        all_outputs + ((size_t)token_id) * output_task_stride_elems;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < size_n) {
            const float g = warp_reduce_sum(acc_g[r]);
            const float u = warp_reduce_sum(acc_u[r]);
            if (laneId == 0) {
                // GELU(tanh-approx) — matches gemma4's gelu_pytorch_tanh:
                //   0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3)))
                const float g2 = g * g;
                const float g3 = g2 * g;
                const float k0 = 0.7978845608028654f;
                const float k1 = 0.044715f;
                const float gelu_g = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g3)));
                out_ptr[row] = gelu_g * u;
            }
        }
    }
}

} // namespace vllm_rs

#define LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    const int shared_bytes_gug = 0; \
    vllm_rs::moe_gemm_gguf_gate_up_gelu_mul_concat_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<grid_dim, block_dim, shared_bytes_gug, stream>>>( \
        gate_up_weights, y_q8_1, \
        sorted_token_ids, expert_ids, \
        outputs, \
        num_experts, topk, \
        size_m, size_n, size_k, \
        kx_padded \
    );

extern "C" void moe_gemm_gguf_gate_up_gelu_mul_concat(
    const float* inputs,
    const void* gate_up_weights,                 // [num_experts, 2*size_n, size_k]
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    float* outputs,                              // [size_m, size_n] = gelu(gate) * up
    int num_experts,
    int topk,
    int size_m,                                  // M*topk (sorted pairs count)
    int size_n,                                  // expert ffn dim (= half of stored 2N)
    int size_k,
    int quant_type,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    int m_in = size_m / topk;
    dim3 grid_dim_quant(num_blocks, m_in, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes = m_in * (kx_padded / QK8_1 * sizeof(block_q8_1));
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(inputs, y_q8_1, size_k, kx_padded);

    const int nWraps = 2;
    constexpr int ROWS_PER_WARP = 1;
    dim3 grid_dim(ceil_div(size_n, nWraps * ROWS_PER_WARP), size_m, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    switch (quant_type) {
        case 0: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break; }
        case 1: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break; }
        case 2: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break; }
        case 3: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break; }
        case 4: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break; }
        case 5: { LAUNCH_MOE_GGUF_GATE_UP_GELU_CONCAT(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break; }
        default: break;
    }
    cudaFreeAsync(y_q8_1, stream);
}

// ───────────────────────────────────────────────────────────────────────
// Dense (non-MoE) gate+up+silu*mul kernel. Same fusion pattern as the
// MoE variant above but stripped of expert routing for batch=1 dense
// FFN (qwen3 base, qwen2, etc.).
//
//   acc_g[r] = dot(gate_w[r], x_q8_1)
//   acc_u[r] = dot(up_w[r],   x_q8_1)
//   output[r] = silu(acc_g[r]) * acc_u[r]
//
// Replaces 3 launches (ffn_up matmul on [2N,K] + narrow + fused_silu_mul)
// with 1 quantize + 1 fused matmul. Saves the [2N] intermediate write.
// ───────────────────────────────────────────────────────────────────────
namespace vllm_rs {

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ROWS_PER_WARP>
__global__ void dense_gate_up_silu_mul_kernel(
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ y_q8_1,
    float       * __restrict__ output,
    int size_n,
    int size_k
) {
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int row0 = blockIdx.x * nWraps * ROWS_PER_WARP + wrapId * ROWS_PER_WARP;
    if (row0 >= size_n) return;

    const block_q_t  * __restrict__ wg = (const block_q_t *)gate_w;
    const block_q_t  * __restrict__ wu = (const block_q_t *)up_w;
    const block_q8_1 * __restrict__ yp = (const block_q8_1 *)y_q8_1;

    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi;

    float acc_g[ROWS_PER_WARP];
    float acc_u[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }

    #pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        const block_q8_1 * y_blk = &yp[kby];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            const int row = row0 + r;
            if (row < size_n) {
                acc_g[r] += vec_dot_q_cuda(&wg[row * blocks_per_row_x + kbx], y_blk, kqs);
                acc_u[r] += vec_dot_q_cuda(&wu[row * blocks_per_row_x + kbx], y_blk, kqs);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        const int row = row0 + r;
        if (row < size_n) {
            const float g = warp_reduce_sum(acc_g[r]);
            const float u = warp_reduce_sum(acc_u[r]);
            if (laneId == 0) {
                const float silu_g = g / (1.0f + __expf(-g));
                output[row] = silu_g * u;
            }
        }
    }
}

} // namespace vllm_rs

#define LAUNCH_DENSE_GATE_UP(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    vllm_rs::dense_gate_up_silu_mul_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda, ROWS_PER_WARP> \
        <<<grid_dim, block_dim, 0, stream>>>( \
        gate_w, up_w, y_q8_1, \
        output, size_n, size_k);

extern "C" void dense_gate_up_silu_mul_v2(
    const float* input,
    const void* gate_w,
    const void* up_w,
    float* output,
    int size_n,
    int size_k,
    int gguf_dtype,
    cudaStream_t stream
) {
    const int QUANTIZE_BLOCK_SIZE = CUDA_QUANTIZE_BLOCK_SIZE;
    const int kx_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int num_blocks = ceil_div(kx_padded, QUANTIZE_BLOCK_SIZE);
    dim3 grid_dim_quant(num_blocks, 1, 1);
    dim3 block_dim_quant(QUANTIZE_BLOCK_SIZE, 1, 1);
    int y_size_in_bytes = (kx_padded / QK8_1) * sizeof(block_q8_1);
    void* y_q8_1 = nullptr;
    cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(input, y_q8_1, size_k, kx_padded);

    constexpr int ROWS_PER_WARP = 1;
    const int nWraps = 2;
    dim3 grid_dim(ceil_div(size_n, nWraps * ROWS_PER_WARP), 1, 1);
    dim3 block_dim(WARP_SIZE, nWraps, 1);

    switch (gguf_dtype) {
        case 0: { LAUNCH_DENSE_GATE_UP(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break; }
        case 1: { LAUNCH_DENSE_GATE_UP(QK_K,  QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break; }
        case 2: { LAUNCH_DENSE_GATE_UP(QK_K,  QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break; }
        case 3: { LAUNCH_DENSE_GATE_UP(QK_K,  QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break; }
        case 4: { LAUNCH_DENSE_GATE_UP(QK_K,  QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break; }
        case 5: { LAUNCH_DENSE_GATE_UP(QK_K,  QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break; }
        default: break;
    }
    cudaFreeAsync(y_q8_1, stream);
}