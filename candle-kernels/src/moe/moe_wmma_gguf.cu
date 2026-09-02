/**
 * @brief CUDA kernel for Mixture-of-Experts (MoE) GEMM with GGUF quantized weights and Tensor Core.
 *
 * This kernel performs batched GEMM where the weight matrix is stored in GGUF
 * quantized format (uint8_t blocks). It supports top-k expert selection and
 * segmented expert layouts. Uses shared memory tiles and WMMA (tensor cores)
 * for efficient computation.
 *
 * Adapted from: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_wmma_gguf.cu
 */
#include "gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <vector>
#include <cassert>
#include <cstring>
#include "moe_utils.cuh"
using namespace nvcuda::wmma;

// Constants from original kernel
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16; // This is fixed by the hardware instruction
using VecT = float4;

constexpr int VEC_SIZE = 8;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 4 warps

constexpr int M_BLK = WARPS_M * WMMA_M; // 32
constexpr int N_BLK = WARPS_N * WMMA_N; // 32

// Helper for ceiling division
#define CEILDIV(A, B) (((A) + (B)-1) / (B))

// --- GGUF Dequantization Function (Warp-level) ---
/**
 * @brief Dequantizes a single GGUF block using one warp (32 threads).
 *
 * @tparam T           Output type (half or nv_bfloat16)
 * @param dequant_out  Pointer to output in shared mem [qk]
 * @param quant_in     Pointer to input GGUF block in shared mem
 * @param type         GGUF type
 * @param qk           Quantization group size (32 or 256)
 * @param laneId       threadIdx.x % 32
 */
template<typename T>
__forceinline__ __device__ void dequantize_block_warp(
    T* dequant_out,
    const uint8_t* quant_in,
    int gguf_dtype //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
) {
    using namespace nvcuda;
    switch (gguf_dtype) {
        case 0: { // qk = 32, q8_0
            // Block: half d (2B), int8_t qs[32] (32B)
            int laneId = threadIdx.x;
            const half* d_ptr = (const half*)quant_in;
            const int8_t* qs = (const int8_t*)(quant_in + 2);

            // Lane 0 loads scale and broadcasts to all other lanes
            half d_val = (laneId == 0) ? *d_ptr : (half)0.0f;
            d_val = __shfl_sync(0xFFFFFFFF, d_val, 0);
            float d_f = __half2float(d_val);

            // 32 lanes dequantize 32 values
            if (laneId < QK8_0) { // qk should be 32
                dequant_out[laneId] = T( (float)qs[laneId] * d_f );
            }
            break;
        }
        case 1: { // q4k, 32 lanes
            dequantize_block_q4_K<T>(quant_in, dequant_out);
            break;
        }
        case 2: { // q2k, 64 lanes
            dequantize_block_q2_K<T>(quant_in, dequant_out);
            break;
        }
        case 3: { // q3k, 64 lanes
            dequantize_block_q3_K<T>(quant_in, dequant_out);
            break;
        }
        case 4: { // q5k, 64 lanes
            dequantize_block_q5_K<T>(quant_in, dequant_out);
            break;
        }
        case 5: { // q6k, 64 lanes
            dequantize_block_q6_K<T>(quant_in, dequant_out);
            break;
        }
        default:
            break;
    }
}

/*
* Template Parameters:
 * @tparam T         Type of input/output (float, half, etc.)
 * @tparam qk        Quantization block size (e.g., 32)
 * @tparam block_q_t Type representing a single GGUF block (e.g., block_q8_0)
 * @tparam wrap_size Warp size used for thread tiling (usually 32)
 *
 * Kernel Parameters:
 * @param input             Input matrix [size_m, size_k]
 * @param weights           GGUF quantized weights buffer (uint8_t blocks)
 * @param sorted_token_ids  Array of sorted token indices for MoE routing
 * @param expert_offsets   [num_experts] array of {start, len} tokens indices for each expert
 * @param topk_weights      Top-k MoE weights per token (optional)
 * @param output            Output matrix [size_m, size_n]
 * @param num_experts       Number of experts in the MoE
 * @param topk              Number of top experts selected per token
 * @param size_m            Number of input rows / tokens
 * @param size_n            Output feature dimension
 * @param size_k            Input feature dimension
 * @param gguf_dtype        GGUF quantization type ID (e.g., Q8_0)
*/
template<typename T, int qk, typename block_q_t, int wrap_size>
__global__ void moe_gemm_gguf_prefill_kernel(
    const T* __restrict__ input,
    const uint8_t* __restrict__ weights, // Now uint8_t*
    const int32_t* __restrict__ sorted_token_ids,
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
    constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * wrap_size; // 128 threads
    
    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const size_t block_size_bytes = sizeof(block_q_t);
    const size_t expert_w_row_stride_bytes = (size_k / qk) * block_size_bytes;
    const uint8_t* expert_w = weights + (size_t)expert_id * size_n * expert_w_row_stride_bytes;

    extern __shared__ uint8_t smem_bytes[];
    
    // 1. A tile: [M_BLK, qk] (dequantized)
    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    size_t A_sh_bytes = (size_t)M_BLK * qk * sizeof(T);
    
    // 2. B tile: [N_BLK, qk] (dequantized)
    uint8_t* B_sh_ptr = smem_bytes + A_sh_bytes;
    size_t B_sh_bytes = (size_t)N_BLK * qk * sizeof(T);
    
    // 3. B quantized tile: [N_BLK * block_size_bytes] (raw GGUF)
    uint8_t* B_quant_sh_ptr = B_sh_ptr + B_sh_bytes;
    size_t B_quant_sh_bytes = (size_t)N_BLK * block_size_bytes;

    // 4. C tile: [M_BLK, N_BLK] (float accumulator)
    uint8_t* C_sh_ptr = B_quant_sh_ptr + B_quant_sh_bytes;
    size_t C_sh_offset = reinterpret_cast<uintptr_t>(C_sh_ptr) % alignof(float);
    if (C_sh_offset != 0) C_sh_ptr += (alignof(float) - C_sh_offset);
    
    // Final aligned shared memory pointers
    T* B_sh = reinterpret_cast<T*>(B_sh_ptr);
    uint8_t* B_quant_sh = reinterpret_cast<uint8_t*>(B_quant_sh_ptr);
    float* C_sh = reinterpret_cast<float*>(C_sh_ptr);

    const int laneId = threadIdx.x;
    const int warpId = threadIdx.y;
    const int threadId = warpId * wrap_size + laneId;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const size_t A_ELEMS_PER_BLOCK = (size_t)M_BLK * qk;
    const size_t VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE;
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;
    
    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        
        // Per-warp accumulator fragment
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        // K-Loop: Strides by GGUF block size `qk`
        for (int k_base = 0; k_base < size_k; k_base += qk) {
            
            // Load A Tile (Inputs) into A_sh
            #pragma unroll
            for (size_t i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                size_t idx = i * VEC_SIZE; // element index
                size_t m_local = idx / qk;
                size_t k_local = idx % qk;

                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;

                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg; 
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = token_index / (topk_weights? 1: topk);
                    *reinterpret_cast<VecT*>(&A_sh[m_local * qk + k_local]) = *reinterpret_cast<const VecT*>(
                        &input[(size_t)input_index * size_k + k_global]
                    );
                } else {
                    *reinterpret_cast<VecT*>(&A_sh[m_local * qk + k_local]) = zero_vec;
                }
            }

            // Load B Tile (Quantized) into B_quant_sh
            const size_t k_base_offset_bytes = (k_base / qk) * block_size_bytes;
            constexpr int ROWS_PER_WARP = N_BLK / WARPS_PER_BLOCK;
            
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                int n_local = warpId * ROWS_PER_WARP + row;
                int n_global = n_base + n_local;
                if (n_local < N_BLK && n_global < size_n) {
                    block_q_t* dest_ptr = reinterpret_cast<block_q_t*>(B_quant_sh + n_local * block_size_bytes);
                    const block_q_t* src_ptr = reinterpret_cast<const block_q_t*>(expert_w + (size_t)n_global * expert_w_row_stride_bytes + k_base_offset_bytes);
                    *dest_ptr = *src_ptr;
                }
            }
            
            __syncthreads();

            // Dequantize B from B_quant_sh to B_sh
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                int n_local = warpId * ROWS_PER_WARP + row;
                int n_global = n_base + n_local;
                if (n_local < N_BLK && n_global < size_n) {
                    const uint8_t* quant_ptr = B_quant_sh + n_local * block_size_bytes;
                    T* dequant_ptr = B_sh + n_local * qk; // Stride by qk
                    // Dequantize one block using this warp
                    dequantize_block_warp(dequant_ptr, quant_ptr, gguf_dtype);
                }
            }

            __syncthreads();

            // Inner WMMA Loop
            // A_sh and B_sh are now dequantized and in shared mem
            // We loop over the K-dim (now `qk`) using the hardware `WMMA_K`
            #pragma unroll
            for (int k_tile = 0; k_tile < qk; k_tile += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

                // Point to the correct 16x16 tile inside the [M_BLK, qk] / [N_BLK, qk] buffers
                const T* A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * qk) + k_tile;
                const T* B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * qk) + k_tile;

                load_matrix_sync(a_frag, A_sh_ptr, qk); // Stride is qk
                load_matrix_sync(b_frag, B_sh_ptr, qk); // Stride is qk
                
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        } // end k_base loop

        // Store C_frag to C_sh
        float* C_sh_ptr_warp = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        store_matrix_sync(C_sh_ptr_warp, c_frag, N_BLK, mem_row_major);
        __syncthreads();

        // Cooperative Store to Global
        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        #pragma unroll
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
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
                        val *= topk_weights[token_index];
                    }
                    output[(size_t)token_index * size_n + n_global] = val;
                }
            }
        }
    } // end m_base loop
}

#define LAUNCH_MOE_GGUF_PREFILL(DTYPE) \
    if (gguf_type == 0) {\
        dim3 block(32, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK8_0, block_q8_0, 32><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    } else if (gguf_type == 1) {\
        dim3 block(32, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK_K, block_q4_K, 32><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    } else if (gguf_type == 2) {\
        dim3 block(64, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK_K, block_q2_K, 64><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    } else if (gguf_type == 3) {\
        dim3 block(64, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK_K, block_q3_K, 64><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    } else if (gguf_type == 4) { \
        dim3 block(64, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK_K, block_q5_K, 64><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    } else if (gguf_type == 5) { \
        dim3 block(64, WARPS_PER_BLOCK, 1);\
        moe_gemm_gguf_prefill_kernel<DTYPE, QK_K, block_q6_K, 64><<<grid, block, smem_bytes, stream>>>(\
            reinterpret_cast<const DTYPE*>(input),\
            reinterpret_cast<const uint8_t*>(weights),\
            sorted_token_ids, expert_offsets, topk_weights,\
            output, num_experts, topk, size_m, size_n, size_k, gguf_type\
        );\
    }


extern "C" void moe_gemm_gguf_prefill(
    const void* input,
    const uint8_t* weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    float* output,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int input_dtype,      // 0 = half, 1 = bfloat16
    int gguf_type, //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5,
    cudaStream_t stream
) {
    int32_t* expert_counts;
    cudaMallocAsync(&expert_counts, num_experts * sizeof(int32_t), stream);

    int32_t* expert_offsets;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    calculate_expert_offsets(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    
    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    
    size_t qk = QK_K;
    size_t block_size_bytes = sizeof(block_q6_K);
    if (gguf_type == 0) { //Q8_0: 0,
        block_size_bytes = sizeof(block_q8_0);
        qk = QK8_0;
    } else if (gguf_type == 1) {// Q4K: 1,
        block_size_bytes = sizeof(block_q4_K);
    } else if (gguf_type == 2) {// Q2K: 2,
        block_size_bytes = sizeof(block_q2_K);
    } else if (gguf_type == 3) {//Q3K: 3,
        block_size_bytes = sizeof(block_q3_K);
    } else if (gguf_type == 4) {//Q5K: 4,
        block_size_bytes = sizeof(block_q5_K);
    }

    // 1. A tile: [M_BLK, qk] (dequantized)
    size_t A_sh_bytes = (size_t)M_BLK * qk * 2; // 2 for half/bfloat16
    
    // 2. B tile: [N_BLK, qk] (dequantized)
    size_t B_sh_bytes = (size_t)N_BLK * qk * 2;
    
    // 3. B quantized tile: [N_BLK * block_size_bytes]
    size_t B_quant_sh_bytes = (size_t)N_BLK * block_size_bytes;

    // 4. C tile: [M_BLK, N_BLK] (float accumulator)
    size_t C_sh_bytes = (size_t)M_BLK * N_BLK * sizeof(float);
    
    // Add up, with padding for C
    size_t smem_bytes = A_sh_bytes + B_sh_bytes + B_quant_sh_bytes;
    size_t C_sh_offset = smem_bytes % alignof(float);
    if (C_sh_offset != 0) smem_bytes += (alignof(float) - C_sh_offset);
    smem_bytes += C_sh_bytes;
    
    if (input_dtype == 0) {
        LAUNCH_MOE_GGUF_PREFILL(half);
    } else {
#ifndef NO_BF16_KERNEL
        LAUNCH_MOE_GGUF_PREFILL(nv_bfloat16);
#endif
    }
    cudaFreeAsync(expert_counts, stream);
    cudaFreeAsync(expert_offsets, stream);
}
