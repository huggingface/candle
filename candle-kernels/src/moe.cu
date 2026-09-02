/**
 * @file moe.cu
 * @brief Dynamic-loading compatible PTX CUDA kernels for Mixture of Experts (MoE).
 *
 * Contains:
 * 1. Token routing utilities: count_tokens_per_expert, expert_prefix_sum
 * 2. Dense WMMA MoE GEMM kernels (F16 and BF16, prefill and decode)
 * 3. Quantized GGUF MoE MMVQ kernels (Decode: Q8_0, Q4_K, Q2_K, Q3_K, Q5_K, Q6_K)
 * 4. Quantized GGUF MoE WMMA kernels (Prefill: F16/BF16 activations x Q8_0, Q4_K, Q2_K, Q3_K, Q5_K, Q6_K)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "moe/gguf.cuh"

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// ============================================================================
// 1. Token Routing Utilities
// ============================================================================

extern "C" __global__ void count_tokens_per_expert_kernel(
    const uint32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_counts,
    int size_m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

#define ALIGN16(x) (((x) + 15) & ~15)

// Single-block or multi-warp parallel exclusive scan supporting any num_experts up to 65536.
extern "C" __global__ void expert_prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts
) {
    extern __shared__ int32_t temp_storage[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Phase 1: Local prefix sum in chunks
    int running_sum = 0;
    if (tid == 0) {
        offsets[0] = 0;
    }

    for (int chunk_start = 0; chunk_start < num_experts; chunk_start += block_size) {
        int idx = chunk_start + tid;
        int val = (idx < num_experts) ? counts[idx] : 0;
        temp_storage[tid] = val;
        __syncthreads();

        for (int offset = 1; offset < block_size; offset <<= 1) {
            int temp_val = 0;
            if (tid >= offset) {
                temp_val = temp_storage[tid - offset];
            }
            __syncthreads();
            if (tid >= offset) {
                temp_storage[tid] += temp_val;
            }
            __syncthreads();
        }

        if (idx < num_experts) {
            offsets[idx + 1] = running_sum + temp_storage[tid];
        }
        __syncthreads();

        running_sum += temp_storage[block_size - 1];
        __syncthreads();
    }
}

namespace vllm_moe {

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ void from_float(half& dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

inline __device__ void from_float(__nv_bfloat16& dst, float src) {
  dst = __float2bfloat16(src);
}

constexpr int WMMA_K = 16;
using VecT = float4;
constexpr int VEC_SIZE = 8;
constexpr int NUM_VECS = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_THREADS = 128;
constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int K_BLK = WMMA_K;

template<typename T, int WMMA_M, int WMMA_N, int WARPS_N>
__device__ void moe_gemm_grouped_device(
    const T* __restrict__ input,
    const T* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    T* __restrict__ output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k
) {
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;

    if (expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const T* expert_w = weights + (size_t)expert_id * size_n * size_k;

    extern __shared__ uint8_t smem_bytes[];
    size_t a_bytes = ALIGN16(M_BLK * K_BLK * sizeof(T));
    size_t b_bytes = ALIGN16(N_BLK * K_BLK * sizeof(T));
    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    T* B_sh = reinterpret_cast<T*>(smem_bytes + a_bytes);
    float* C_sh = reinterpret_cast<float*>(smem_bytes + a_bytes + b_bytes);

    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const int M_WARP = M_BLK / (WARPS_PER_BLOCK / WARPS_N);
    const int N_WARP = N_BLK / WARPS_N;

    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[M_WARP / WMMA_M][N_WARP / WMMA_N];
        #pragma unroll
        for (int i = 0; i < M_WARP / WMMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < N_WARP / WMMA_N; j++) {
                fill_fragment(c_frag[i][j], 0.0f);
            }
        }

        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            __syncthreads();

            // Load A_sh: [M_BLK, K_BLK]
            int tid = threadIdx.x;
            if (tid < NUM_VECS) {
                int a_row_offset = tid / 2;
                int a_col_offset = (tid % 2) * VEC_SIZE;
                int token_idx_in_segment = m_base + a_row_offset;
                if (token_idx_in_segment < num_rows_in_segment) {
                    int original_token_idx = sorted_token_ids[segment_start + token_idx_in_segment];
                    if (original_token_idx < size_m) {
                        int token_to_fetch = (topk_weights == nullptr) ? (original_token_idx / topk) : original_token_idx;
                        const T* in_ptr = input + (size_t)token_to_fetch * size_k + k_base + a_col_offset;
                        *reinterpret_cast<VecT*>(&A_sh[a_row_offset * K_BLK + a_col_offset]) = *reinterpret_cast<const VecT*>(in_ptr);
                    } else {
                        *reinterpret_cast<VecT*>(&A_sh[a_row_offset * K_BLK + a_col_offset]) = zero_vec;
                    }
                } else {
                    *reinterpret_cast<VecT*>(&A_sh[a_row_offset * K_BLK + a_col_offset]) = zero_vec;
                }
            }

            // Load B_sh: [N_BLK, K_BLK]
            if (tid < NUM_VECS) {
                int b_row_offset = tid / 2;
                int b_col_offset = (tid % 2) * VEC_SIZE;
                int global_n = n_base + b_row_offset;
                if (global_n < size_n) {
                    const T* w_ptr = expert_w + (size_t)global_n * size_k + k_base + b_col_offset;
                    *reinterpret_cast<VecT*>(&B_sh[b_row_offset * K_BLK + b_col_offset]) = *reinterpret_cast<const VecT*>(w_ptr);
                } else {
                    *reinterpret_cast<VecT*>(&B_sh[b_row_offset * K_BLK + b_col_offset]) = zero_vec;
                }
            }

            __syncthreads();

            // WMMA computation
            #pragma unroll
            for (int i = 0; i < M_WARP / WMMA_M; i++) {
                int m_offset = warp_m_idx * M_WARP + i * WMMA_M;
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
                load_matrix_sync(a_frag, &A_sh[m_offset * K_BLK], K_BLK);

                #pragma unroll
                for (int j = 0; j < N_WARP / WMMA_N; j++) {
                    int n_offset = warp_n_idx * N_WARP + j * WMMA_N;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;
                    load_matrix_sync(b_frag, &B_sh[n_offset * K_BLK], K_BLK);
                    mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        } // end k_base loop

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < M_WARP / WMMA_M; i++) {
            int m_offset = warp_m_idx * M_WARP + i * WMMA_M;
            #pragma unroll
            for (int j = 0; j < N_WARP / WMMA_N; j++) {
                int n_offset = warp_n_idx * N_WARP + j * WMMA_N;
                store_matrix_sync(&C_sh[m_offset * N_BLK + n_offset], c_frag[i][j], N_BLK, mem_row_major);
            }
        }

        __syncthreads();

        int total_c_elements = M_BLK * N_BLK;
        int elements_per_thread = total_c_elements / BLOCK_THREADS;
        int thread_offset = threadIdx.x * elements_per_thread;

        #pragma unroll
        for (int elem = 0; elem < elements_per_thread; elem++) {
            int i = thread_offset + elem;
            int m_local_c = i / N_BLK;
            int n_local_c = i % N_BLK;

            int m_seg = m_base + m_local_c;
            int n_global = n_base + n_local_c;

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c];
                    if (topk_weights) {
                        val *= topk_weights[token_pair_index];
                    }
                    from_float(output[(size_t)token_index * size_n + n_global], val);
                }
            }
        }
    } // end m_base loop
}

} // namespace vllm_moe

// ============================================================================
// 2. Dense WMMA MoE Kernels Entry Points
// ============================================================================

extern "C" __global__ void moe_gemm_wmma_f16_prefill(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    half* __restrict__ output,
    int num_experts, int topk, int size_m, int size_n, int size_k
) {
    vllm_moe::moe_gemm_grouped_device<half, 16, 16, 2>(
        input, weights, sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k
    );
}

extern "C" __global__ void moe_gemm_wmma_f16_decode(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    half* __restrict__ output,
    int num_experts, int topk, int size_m, int size_n, int size_k
) {
    vllm_moe::moe_gemm_grouped_device<half, 8, 32, 1>(
        input, weights, sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k
    );
}

#ifndef NO_BF16_KERNEL
extern "C" __global__ void moe_gemm_wmma_bf16_prefill(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int num_experts, int topk, int size_m, int size_n, int size_k
) {
    vllm_moe::moe_gemm_grouped_device<__nv_bfloat16, 16, 16, 2>(
        input, weights, sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k
    );
}

extern "C" __global__ void moe_gemm_wmma_bf16_decode(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int num_experts, int topk, int size_m, int size_n, int size_k
) {
    vllm_moe::moe_gemm_grouped_device<__nv_bfloat16, 8, 32, 1>(
        input, weights, sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k
    );
}
#endif

// ============================================================================
// 3. Quantized GGUF MoE MMVQ Kernels (Decode: single or small token count)
// ============================================================================

namespace vllm_moe_gguf {

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemm_gguf_decode_device(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const uint32_t* __restrict__ sorted_token_ids,
    const uint32_t* __restrict__ expert_ids,
    const float* __restrict__ topk_weights,
    float * __restrict__ all_outputs,
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

    if (row >= size_n || m_idx >= size_m) {
        return;
    }

    const int token_pair_index = m_idx;
    const int token_idx = (sorted_token_ids != nullptr) ? sorted_token_ids[token_pair_index] : token_pair_index;
    const int expert_id = expert_ids[token_pair_index];

    if (expert_id < 0 || expert_id >= num_experts) {
        return;
    }

    int input_token_idx = (topk_weights == nullptr) ? (token_idx / topk) : token_idx;
    if (sorted_token_ids == nullptr) {
        input_token_idx = (topk_weights == nullptr) ? (m_idx / topk) : m_idx;
    }

    const size_t input_stride_bytes = (size_t)(k_padded / QK8_1) * sizeof(block_q8_1);
    const void * input_ptr = (const char *)all_inputs + (size_t)input_token_idx * input_stride_bytes;

    const size_t weight_row_stride_bytes = (size_t)(size_k / qk) * sizeof(block_q_t);
    const size_t expert_matrix_stride_bytes = (size_t)size_n * weight_row_stride_bytes;
    const void * weight_ptr = (const char *)all_weights
                            + (size_t)expert_id * expert_matrix_stride_bytes
                            + (size_t)row * weight_row_stride_bytes;

    extern __shared__ char shared_mem[];
    const int bytes_per_warp = (size_k / qk) * sizeof(block_q_t);
    void * shared_weight_ptr = shared_mem + wrapId * bytes_per_warp;

    const int num_blocks_per_row = size_k / qk;
    const int num_ints = (num_blocks_per_row * sizeof(block_q_t)) / sizeof(int);

    const int* global_src = (const int*)weight_ptr;
    int* shared_dst = (int*)shared_weight_ptr;

    for (int i = laneId; i < num_ints; i += 32) {
        shared_dst[i] = global_src[i];
    }
    __syncwarp();

    float tmp = 0.0f;
    const block_q_t * w = (const block_q_t *)shared_weight_ptr;
    const block_q8_1 * x = (const block_q8_1 *)input_ptr;

    const int blocks_per_row_x = size_k / qk;
    constexpr int blocks_per_iter = vdr * 32 / qi;

    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        tmp += vec_dot_q_cuda(&w[kbx], &x[kby], kqs);
    }

    tmp = warp_reduce_sum(tmp);

    if (laneId == 0) {
        if (topk_weights != nullptr) {
            tmp *= topk_weights[token_pair_index];
        }
        float * out_ptr = all_outputs + (size_t)token_idx * size_n;
        out_ptr[row] = tmp;
    }
}

} // namespace vllm_moe_gguf

#define DEFINE_MOE_GGUF_DECODE(NAME, QK, QI, BLOCK_T, VDR, VEC_DOT) \
extern "C" __global__ void moe_gemm_gguf_##NAME( \
    const void * __restrict__ all_weights, \
    const void * __restrict__ all_inputs, \
    const uint32_t* __restrict__ sorted_token_ids, \
    const uint32_t* __restrict__ expert_ids, \
    const float* __restrict__ topk_weights, \
    float * __restrict__ all_outputs, \
    int num_experts, int topk, \
    int size_m, int size_n, int size_k, \
    int k_padded \
) { \
    vllm_moe_gguf::moe_gemm_gguf_decode_device<QK, QI, BLOCK_T, VDR, VEC_DOT>( \
        all_weights, all_inputs, sorted_token_ids, expert_ids, topk_weights, all_outputs, \
        num_experts, topk, size_m, size_n, size_k, k_padded \
    ); \
}

DEFINE_MOE_GGUF_DECODE(q8_0, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
DEFINE_MOE_GGUF_DECODE(q4_k, QK_K,  QI4_K, block_q4_K,  VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_MOE_GGUF_DECODE(q2_k, QK_K,  QI2_K, block_q2_K,  VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DEFINE_MOE_GGUF_DECODE(q3_k, QK_K,  QI3_K, block_q3_K,  VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_MOE_GGUF_DECODE(q5_k, QK_K,  QI5_K, block_q5_K,  VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_MOE_GGUF_DECODE(q6_k, QK_K,  QI6_K, block_q6_K,  VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

// ============================================================================
// 4. Quantized GGUF MoE WMMA Kernels (Prefill: sequence / batch > 1)
// ============================================================================

namespace vllm_moe_wmma_gguf {

template<typename T>
__forceinline__ __device__ void dequantize_block_warp(
    T* dequant_out,
    const uint8_t* quant_in,
    int gguf_dtype
) {
    using namespace nvcuda;
    switch (gguf_dtype) {
        case 0: { // qk = 32, q8_0
            int laneId = threadIdx.x;
            const half* d_ptr = (const half*)quant_in;
            const int8_t* qs = (const int8_t*)(quant_in + 2);

            half d_val = (laneId == 0) ? *d_ptr : (half)0.0f;
            d_val = __shfl_sync(0xFFFFFFFF, d_val, 0);
            float d_f = __half2float(d_val);

            if (laneId < QK8_0) {
                dequant_out[laneId] = T((float)qs[laneId] * d_f);
            }
            break;
        }
        case 1: {
            dequantize_block_q4_K<T>(quant_in, dequant_out);
            break;
        }
        case 2: {
            dequantize_block_q2_K<T>(quant_in, dequant_out);
            break;
        }
        case 3: {
            dequantize_block_q3_K<T>(quant_in, dequant_out);
            break;
        }
        case 4: {
            dequantize_block_q5_K<T>(quant_in, dequant_out);
            break;
        }
        case 5: {
            dequantize_block_q6_K<T>(quant_in, dequant_out);
            break;
        }
        default:
            break;
    }
}

template<typename T, int qk, typename block_q_t, int wrap_size>
__device__ void moe_gemm_gguf_prefill_device(
    const T* __restrict__ input,
    const uint8_t* __restrict__ weights,
    const uint32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    float* __restrict__ output,
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k,
    const int gguf_dtype
) {
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;

    if (expert_id < 0 || expert_id >= num_experts) return;
    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;
    
    const int n_base = n_tile_idx * vllm_moe::N_BLK;
    if (n_base >= size_n) return;

    const size_t block_size_bytes = sizeof(block_q_t);
    const size_t expert_w_row_stride_bytes = (size_k / qk) * block_size_bytes;
    const uint8_t* expert_w = weights + (size_t)expert_id * size_n * expert_w_row_stride_bytes;

    extern __shared__ uint8_t smem_bytes[];
    
    // 1. A tile: [M_BLK, qk] (dequantized, 16-byte aligned)
    size_t A_sh_bytes = ALIGN16((size_t)vllm_moe::M_BLK * qk * sizeof(T));
    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    
    // 2. B tile: [N_BLK, qk] (dequantized, 16-byte aligned)
    size_t B_sh_bytes = ALIGN16((size_t)vllm_moe::N_BLK * qk * sizeof(T));
    T* B_sh = reinterpret_cast<T*>(smem_bytes + A_sh_bytes);
    
    // 3. B quantized tile: [N_BLK * block_size_bytes] (raw GGUF, 16-byte aligned)
    size_t B_quant_sh_bytes = ALIGN16((size_t)vllm_moe::N_BLK * block_size_bytes);
    uint8_t* B_quant_sh = smem_bytes + A_sh_bytes + B_sh_bytes;

    // 4. C tile: [M_BLK, N_BLK] (float accumulator, 16-byte aligned)
    float* C_sh = reinterpret_cast<float*>(smem_bytes + A_sh_bytes + B_sh_bytes + B_quant_sh_bytes);

    const int laneId = threadIdx.x;
    const int warpId = threadIdx.y;
    const int warp_m_idx = warpId / 2;
    const int warp_n_idx = warpId % 2;

    const size_t A_ELEMS_PER_BLOCK = (size_t)vllm_moe::M_BLK * qk;
    const size_t VEC_ELEMS_A = A_ELEMS_PER_BLOCK / vllm_moe::VEC_SIZE;
    vllm_moe::VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;
    
    for (int m_base = 0; m_base < num_rows_in_segment; m_base += vllm_moe::M_BLK) {
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += qk) {
            __syncthreads();

            // Load A tile
            for (size_t v_idx = warpId * wrap_size + laneId; v_idx < VEC_ELEMS_A; v_idx += vllm_moe::BLOCK_THREADS) {
                const int m_local = v_idx / (qk / vllm_moe::VEC_SIZE);
                const int k_local = (v_idx % (qk / vllm_moe::VEC_SIZE)) * vllm_moe::VEC_SIZE;
                const int m_seg = m_base + m_local;

                if (m_seg < num_rows_in_segment) {
                    const int token_pair_index = segment_start + m_seg;
                    if (token_pair_index < size_m) {
                        const int token_index = sorted_token_ids[token_pair_index];
                        const int token_to_fetch = (topk_weights == nullptr) ? (token_index / topk) : token_index;
                        const size_t in_offset = (size_t)token_to_fetch * size_k + k_base + k_local;
                        *reinterpret_cast<vllm_moe::VecT*>(&A_sh[m_local * qk + k_local]) =
                            *reinterpret_cast<const vllm_moe::VecT*>(&input[in_offset]);
                    } else {
                        *reinterpret_cast<vllm_moe::VecT*>(&A_sh[m_local * qk + k_local]) = zero_vec;
                    }
                } else {
                    *reinterpret_cast<vllm_moe::VecT*>(&A_sh[m_local * qk + k_local]) = zero_vec;
                }
            }

            // Load B quantized tile
            const int block_idx_in_row = k_base / qk;
            const size_t total_b_quant_bytes = (size_t)vllm_moe::N_BLK * block_size_bytes;
            const size_t b_quant_u32_count = total_b_quant_bytes / sizeof(uint32_t);

            for (size_t u32_idx = warpId * wrap_size + laneId; u32_idx < b_quant_u32_count; u32_idx += vllm_moe::BLOCK_THREADS) {
                const int n_local = u32_idx / (block_size_bytes / sizeof(uint32_t));
                const int byte_offset_in_block = (u32_idx % (block_size_bytes / sizeof(uint32_t))) * sizeof(uint32_t);
                const int global_n = n_base + n_local;

                if (global_n < size_n) {
                    const size_t w_offset = (size_t)global_n * expert_w_row_stride_bytes
                                          + (size_t)block_idx_in_row * block_size_bytes
                                          + byte_offset_in_block;
                    *reinterpret_cast<uint32_t*>(&B_quant_sh[n_local * block_size_bytes + byte_offset_in_block]) =
                        *reinterpret_cast<const uint32_t*>(&expert_w[w_offset]);
                } else {
                    *reinterpret_cast<uint32_t*>(&B_quant_sh[n_local * block_size_bytes + byte_offset_in_block]) = 0;
                }
            }

            __syncthreads();

            // Dequantize B
            const int total_blocks = vllm_moe::N_BLK;
            for (int block_idx = warpId; block_idx < total_blocks; block_idx += vllm_moe::WARPS_PER_BLOCK) {
                const int n_local = block_idx;
                const int global_n = n_base + n_local;

                if (global_n < size_n) {
                    T* dequant_ptr = B_sh + n_local * qk;
                    const uint8_t* quant_ptr = B_quant_sh + n_local * block_size_bytes;
                    dequantize_block_warp(dequant_ptr, quant_ptr, gguf_dtype);
                } else {
                    if (laneId < qk) {
                        B_sh[n_local * qk + laneId] = (T)0.0f;
                    }
                }
            }

            __syncthreads();

            // WMMA sub-steps inside the GGUF block
            #pragma unroll
            for (int k_sub = 0; k_sub < qk; k_sub += 16) {
                const int m_offset = warp_m_idx * 16;
                const int n_offset = warp_n_idx * 16;

                fragment<matrix_a, 16, 16, 16, T, row_major> a_frag;
                load_matrix_sync(a_frag, &A_sh[m_offset * qk + k_sub], qk);

                fragment<matrix_b, 16, 16, 16, T, col_major> b_frag;
                load_matrix_sync(b_frag, &B_sh[n_offset * qk + k_sub], qk);

                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        } // end k_base loop

        __syncthreads();

        const int m_offset = warp_m_idx * 16;
        const int n_offset = warp_n_idx * 16;
        store_matrix_sync(&C_sh[m_offset * vllm_moe::N_BLK + n_offset], c_frag, vllm_moe::N_BLK, mem_row_major);

        __syncthreads();

        // Write output
        const int total_c_elements = vllm_moe::M_BLK * vllm_moe::N_BLK;
        const int elements_per_thread = total_c_elements / vllm_moe::BLOCK_THREADS;
        const int thread_offset = (warpId * wrap_size + laneId) * elements_per_thread;

        #pragma unroll
        for (int elem = 0; elem < elements_per_thread; elem++) {
            const int i = thread_offset + elem;
            const int m_local_c = i / vllm_moe::N_BLK;
            const int n_local_c = i % vllm_moe::N_BLK;

            const int m_seg = m_base + m_local_c;
            const int n_global = n_base + n_local_c;

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                const int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    const int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * vllm_moe::N_BLK + n_local_c];
                    if (topk_weights) {
                        val *= topk_weights[token_pair_index];
                    }
                    output[(size_t)token_index * size_n + n_global] = val;
                }
            }
        }
    } // end m_base loop
}

} // namespace vllm_moe_wmma_gguf

#define DEFINE_MOE_GGUF_PREFILL(NAME, T_TYPE, QK, BLOCK_T, WRAP_SZ, GGUF_TYPE) \
extern "C" __global__ void moe_gemm_gguf_prefill_##T_TYPE##_##NAME( \
    const T_TYPE* __restrict__ input, \
    const uint8_t* __restrict__ weights, \
    const uint32_t* __restrict__ sorted_token_ids, \
    const int32_t* __restrict__ expert_offsets, \
    const float* __restrict__ topk_weights, \
    float* __restrict__ output, \
    const int num_experts, const int topk, \
    const int32_t size_m, const int32_t size_n, const int32_t size_k \
) { \
    vllm_moe_wmma_gguf::moe_gemm_gguf_prefill_device<T_TYPE, QK, BLOCK_T, WRAP_SZ>( \
        input, weights, sorted_token_ids, expert_offsets, topk_weights, output, \
        num_experts, topk, size_m, size_n, size_k, GGUF_TYPE \
    ); \
}

DEFINE_MOE_GGUF_PREFILL(q8_0, half, QK8_0, block_q8_0, 32, 0)
DEFINE_MOE_GGUF_PREFILL(q4_k, half, QK_K,  block_q4_K,  32, 1)
DEFINE_MOE_GGUF_PREFILL(q2_k, half, QK_K,  block_q2_K,  64, 2)
DEFINE_MOE_GGUF_PREFILL(q3_k, half, QK_K,  block_q3_K,  64, 3)
DEFINE_MOE_GGUF_PREFILL(q5_k, half, QK_K,  block_q5_K,  64, 4)
DEFINE_MOE_GGUF_PREFILL(q6_k, half, QK_K,  block_q6_K,  64, 5)

#ifndef NO_BF16_KERNEL
DEFINE_MOE_GGUF_PREFILL(q8_0, __nv_bfloat16, QK8_0, block_q8_0, 32, 0)
DEFINE_MOE_GGUF_PREFILL(q4_k, __nv_bfloat16, QK_K,  block_q4_K,  32, 1)
DEFINE_MOE_GGUF_PREFILL(q2_k, __nv_bfloat16, QK_K,  block_q2_K,  64, 2)
DEFINE_MOE_GGUF_PREFILL(q3_k, __nv_bfloat16, QK_K,  block_q3_K,  64, 3)
DEFINE_MOE_GGUF_PREFILL(q5_k, __nv_bfloat16, QK_K,  block_q5_K,  64, 4)
DEFINE_MOE_GGUF_PREFILL(q6_k, __nv_bfloat16, QK_K,  block_q6_K,  64, 5)
#endif
