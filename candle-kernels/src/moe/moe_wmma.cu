/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  Each block computes a tile of the output corresponding to:
 *    - One expert segment (group of tokens routed to the same expert)
 *    - One N-dimension tile (a sub-block of the expert's output features)
 *
 *  The kernel loads input activations and expert weights in tiles using shared memory,
 *  performs matrix multiplication using Tensor Cores (WMMA), and accumulates results
 *  into a shared C tile. The final results are written atomically into the global
 *  output buffer to support multi-expert (top-k > 1) routing where tokens appear in
 *  multiple experts’ outputs.
 *
 *  Adapted from https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_wmma.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>
#include "moe_utils.cuh"
using namespace nvcuda::wmma;

namespace vllm_rs {

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))

constexpr int WMMA_K = 16;
using VecT = float4;

// Vectorized load size (float4 = 128 bits = 8 half/bfloat16 values)
constexpr int VEC_SIZE = 8;
constexpr int NUM_VECS = 32;

// We use 4 Warps (128 threads) per block
constexpr int WARPS_PER_BLOCK = 4; // 4 warps
constexpr int BLOCK_THREADS = 128; // 128 threads

constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int K_BLK = WMMA_K;           // 16


/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  @tparam T               Data type: half or nv_bfloat16
 *
 *  @param input            [size_m or size_m/topk, size_k]
 *  @param weights          [num_experts, size_n, size_k] compacted expert weights
 *  @param sorted_token_ids [size_m] mapping of per-token row indices (sorted by expert)
 *  @param expert_offsets   [num_experts] array of {start, len} tokens indices for each expert
 *  @param topk_weights     [size_m] optional per-token scaling weights (nullptr if unused)
 *  @param output           [size_m, size_n] global output buffer (must be zero-initialized)
 *  @param num_experts      Total number of experts
 *  @param topk             Number of experts each token is routed to
 *  @param size_m           Number of tokens
 *  @param size_n           Output hidden dimension (per expert)
 *  @param size_k           Input hidden dimension
*/
template<typename T, int WMMA_M, int WMMA_N, int WARPS_N>
__global__ void moe_gemm_grouped_kernel(
    const T* __restrict__ input,           // [size_m, size_k]
    const T* __restrict__ weights,         // [num_experts, size_n, size_k]
    const int32_t* __restrict__ sorted_token_ids, // [size_m]
    const int32_t* __restrict__ expert_offsets,   // [num_experts]
    const float* __restrict__ topk_weights, // [size_m]
    T* __restrict__ output,                 // [size_m, size_n] (Zero-initialized)
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k
) {
    // Get Segment and N-Tile for this Block
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;
    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const T* expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

    extern __shared__ uint8_t smem_bytes[];
    
    // A tile: [M_BLK, K_BLK] (row-major)
    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    // B tile: [N_BLK, K_BLK] (row-major)
    T* B_sh = reinterpret_cast<T*>(A_sh + M_BLK * K_BLK);
    uint8_t* C_ptr = reinterpret_cast<uint8_t*>(B_sh + N_BLK * K_BLK);

    // align next pointer to float alignment
    size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
    if (offset != 0) {
        C_ptr += (alignof(float) - offset);
    }
    float* C_sh = reinterpret_cast<float*>(C_ptr); // shared scratch for final per-block tile writes

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int laneId = threadId % 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const int B_ELEMS_PER_BLOCK = N_BLK * K_BLK;
    const int VEC_ELEMS_B = B_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
    const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
    const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;
    
    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        // We'll accumulate full-K results in per-warp fragments (initialized here)
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        // For every k_block we will load B_sh and A_sh for this m_base subsequently
        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // Load B Tile (Weights) into B_sh
            for (int i = threadId; i < VEC_ELEMS_B; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE; // element index (0..511)
                int n_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int n_global = n_base + n_local;
                int k_global = k_base + k_local;

                // this should be always satisfied since k dim aligned to 8
                if (n_global < size_n && k_global < size_k) {
                    *reinterpret_cast<VecT*>(&B_sh[n_local * K_BLK + k_local]) = *reinterpret_cast<const VecT*>(
                        &expert_w[(size_t)n_global * size_k + k_global]
                    );
                } else {
                    *reinterpret_cast<VecT*>(&B_sh[n_local * K_BLK + k_local]) = zero_vec;
                }
            }

            // Load A Tile (Inputs) into A_sh for this m_base and this k_base
            for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE; // element index
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int m_seg = m_base + m_local; // row index within segment
                int k_global = k_base + k_local;

                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg; 
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = token_index / (topk_weights? 1: topk);
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = *reinterpret_cast<const VecT*>(
                        &input[(size_t)input_index * size_k + k_global]
                    );
                } else {
                    // in case m dim in this segment not aligned to 8
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = zero_vec;
                }
            }

            __syncthreads();

            // Compute (Warp-level) : update c_frag for this k_block
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

            // Point this warp to its tile in shared memory
            const T* A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
            const T* B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

            load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
            load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

            // Accumulate into c_frag (which persists across k_base iterations)
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads(); // Fix shared memory mismatch on V100
        } // end k_base loop (we have a fully-accumulated c_frag for this m_base tile)

        // Store the accumulated c_frag to C_sh (shared) once per warp
        // Point this warp to its 16x16 tile *within* the 32x32 C_sh
        float* C_sh_ptr = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        // store the full accumulated 16x16 tile (note ld = N_BLK, result in row-major in C_sh)
        store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

        __syncthreads();

        // Cooperative Store from C_sh to Global
        // 128 threads write [M_BLK, N_BLK] = [32, 32] = 1024 elements
        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
            int m_local_c = i / N_BLK; // row in C_sh (0..31)
            int n_local_c = i % N_BLK; // col in C_sh (0..31)

            int m_seg = m_base + m_local_c;    // row index within segment
            int n_global = n_base + n_local_c; // col index in output

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c]; 
                    if (topk_weights) {
                        val *= topk_weights[token_index];
                    }
                    from_float(output[(size_t)token_index * size_n + n_global], val);
                }
            }
        }
    } // end m_base loop
}

}

#define LAUNCH_MOE_WMMA(DTYPE, WMMA_M, WMMA_N, WARPS_N)\
    vllm_rs::moe_gemm_grouped_kernel<DTYPE, WMMA_M, WMMA_N, WARPS_N><<<grid, block, smem_bytes, stream>>>(\
        reinterpret_cast<const DTYPE*>(input),\
        reinterpret_cast<const DTYPE*>(weights),\
        sorted_token_ids,\
        expert_offsets,\
        topk_weights,\
        reinterpret_cast<DTYPE*>(output),\
        num_experts, topk,\
        size_m, size_n, size_k \
    );\

extern "C" void moe_gemm_wmma(
    const void* input,                // [size_m, size_k]
    const void* weights,              // [num_experts, size_n, size_k]
    const int32_t* sorted_token_ids,  // [size_m] (Device)
    const int32_t* expert_ids,   // [size_m * topk]
    const float* topk_weights,        // [size_m] (Device, can be nullptr)
    void* output,                     // [size_m, size_n]
    int32_t* expert_counts, // prealloc [num_experts]
    int32_t* expert_offsets, // prealloc [num_experts + 1]
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int data_type,                    // 0 = half, 1 = bfloat16
    bool is_prefill,
    cudaStream_t stream
) {
    if (is_prefill) {
        calculate_expert_offsets(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    } else {
        calculate_expert_offsets_light(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    }

    int grid_n = CEILDIV(size_n, vllm_rs::N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(vllm_rs::BLOCK_THREADS, 1, 1);

    // Shared memory: A_sh[M_BLK, K_BLK] + B_sh[N_BLK, K_BLK]
    size_t A_sh_bytes = vllm_rs::M_BLK * vllm_rs::K_BLK * 2; // (32*16 * 2) = 1024
    size_t B_sh_bytes = vllm_rs::N_BLK * vllm_rs::K_BLK * 2; // (32*16 * 2) = 1024
    size_t C_sh_bytes = vllm_rs::M_BLK * vllm_rs::N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16; 
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes; // ~6KB total needed

    if (data_type == 0) { // half
        if (is_prefill) {
            LAUNCH_MOE_WMMA(half, 16, 16, 2)
        } else {
            // we use smaller M_tile and larger N_tile for decoding
            LAUNCH_MOE_WMMA(half, 8, 32, 1)
        }
    } else if (data_type == 1) { // bfloat16
        if (is_prefill) {
            LAUNCH_MOE_WMMA(nv_bfloat16, 16, 16, 2)
        } else {
            LAUNCH_MOE_WMMA(nv_bfloat16, 8, 32, 1)
        }
    }
}