// Fused gather + matmul kernel for MoE (Mixture of Experts) acceleration
// This kernel fuses index_select on weights + matmul into a single kernel
// to reduce kernel launch overhead and memory bandwidth.
//
// Copyright © 2024-2025 Based on MLX concepts from Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// GatherMM param classes
///////////////////////////////////////////////////////////////////////////////

struct GatherMMParams {
    int M;                    // Number of rows (num_tokens * num_experts_per_tok)
    int N;                    // Output features
    int K;                    // Hidden dim / input features
    int num_experts;          // Number of expert weight matrices
    int expert_stride;        // Stride between experts in B (N * K)
};

///////////////////////////////////////////////////////////////////////////////
// Simple gather_mm kernel: one thread per output element
// A: [M, K] - input (M = num_tokens * num_experts_per_tok, flattened)
// B: [num_experts, N, K] - expert weights (transposed: each expert is [N, K])
// indices: [M] - expert index for each row
// D: [M, N] - output
///////////////////////////////////////////////////////////////////////////////

template <typename T>
[[kernel]] void gather_mm_simple(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {

    const int row = tid.y;
    const int col = tid.x;

    if (row >= params->M || col >= params->N) {
        return;
    }

    const int K = params->K;
    const int N = params->N;
    const uint32_t expert = indices[row];

    // B is [num_experts, N, K] (transposed weights)
    // B[expert, col, :] gives us the weight vector for this output column
    const device T* A_row = A + row * K;
    const device T* B_col = B + expert * params->expert_stride + col * K;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += float(A_row[k]) * float(B_col[k]);
    }

    D[row * N + col] = T(sum);
}

///////////////////////////////////////////////////////////////////////////////
// Tiled gather_mm kernel: uses threadgroup memory for better performance
// Each threadgroup computes one row of output (all N columns)
// Uses tiling in K dimension
///////////////////////////////////////////////////////////////////////////////

template <typename T, int BK = 64, int TN = 4>
[[kernel]] void gather_mm_tiled(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint tid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {

    const int row = tid_y;
    if (row >= params->M) {
        return;
    }

    const int K = params->K;
    const int N = params->N;
    const uint32_t expert = indices[row];

    const device T* A_row = A + row * K;
    const device T* B_expert = B + expert * params->expert_stride;
    device T* D_row = D + row * N;

    // Each thread handles TN output columns
    const int col_start = tid_x * TN;
    if (col_start >= N) {
        return;
    }

    // Accumulators for TN outputs
    float sums[TN] = {0.0f};

    // Threadgroup memory for A tile
    threadgroup T A_shared[BK];

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_base = kt * BK;

        // Cooperatively load A tile into shared memory
        for (int i = tid_x; i < BK && (k_base + i) < K; i += threads_per_group) {
            A_shared[i] = A_row[k_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot products
        const int k_end = min(BK, K - k_base);

        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN && (col_start + j) < N; j++) {
            const device T* B_col = B_expert + (col_start + j) * K + k_base;
            for (int k = 0; k < k_end; k++) {
                sums[j] += float(A_shared[k]) * float(B_col[k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results
    for (int j = 0; j < TN && (col_start + j) < N; j++) {
        D_row[col_start + j] = T(sums[j]);
    }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD-optimized gather_mm kernel using simdgroup operations
// Each simdgroup handles multiple output elements per row
///////////////////////////////////////////////////////////////////////////////

template <typename T, int BK = 32>
[[kernel]] void gather_mm_simd(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device T* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

    const int row = tgid.y;
    const int col_block = tgid.x * 128; // Each threadgroup handles 128 columns (4 simdgroups × 32 lanes)

    if (row >= params->M) {
        return;
    }

    const int K = params->K;
    const int N = params->N;
    const uint32_t expert = indices[row];

    const device T* A_row = A + row * K;
    const device T* B_expert = B + expert * params->expert_stride;
    device T* D_row = D + row * N;

    // Each simdgroup handles 32 columns
    const int col = col_block + simd_group_id * 32 + simd_lane_id;
    if (col >= N) {
        return;
    }

    const device T* B_col = B_expert + col * K;

    // Compute dot product using SIMD reduction
    float sum = 0.0f;

    // Process K in chunks
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        float4 a_vec = float4(A_row[k], A_row[k+1], A_row[k+2], A_row[k+3]);
        float4 b_vec = float4(B_col[k], B_col[k+1], B_col[k+2], B_col[k+3]);
        sum += dot(a_vec, b_vec);
    }

    // Handle remaining elements
    for (; k < K; k++) {
        sum += float(A_row[k]) * float(B_col[k]);
    }

    D_row[col] = T(sum);
}

///////////////////////////////////////////////////////////////////////////////
// Kernel instantiations
///////////////////////////////////////////////////////////////////////////////

// Simple kernel - correct but basic
template [[host_name("gather_mm_simple_f32")]]
[[kernel]] void gather_mm_simple<float>(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device float* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]);

template [[host_name("gather_mm_simple_f16")]]
[[kernel]] void gather_mm_simple<half>(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device half* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]);

#if defined(__HAVE_BFLOAT__)
template [[host_name("gather_mm_simple_bf16")]]
[[kernel]] void gather_mm_simple<bfloat>(
    const device bfloat* A [[buffer(0)]],
    const device bfloat* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device bfloat* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]);
#endif

// Tiled kernel - better memory access patterns
template [[host_name("gather_mm_tiled_f32")]]
[[kernel]] void gather_mm_tiled<float, 64, 4>(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device float* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint tid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

template [[host_name("gather_mm_tiled_f16")]]
[[kernel]] void gather_mm_tiled<half, 64, 4>(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device half* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint tid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

#if defined(__HAVE_BFLOAT__)
template [[host_name("gather_mm_tiled_bf16")]]
[[kernel]] void gather_mm_tiled<bfloat, 64, 4>(
    const device bfloat* A [[buffer(0)]],
    const device bfloat* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device bfloat* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint tid_y [[threadgroup_position_in_grid]],
    uint tid_x [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);
#endif

// SIMD kernel - vectorized loads
template [[host_name("gather_mm_simd_f32")]]
[[kernel]] void gather_mm_simd<float, 32>(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device float* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);

template [[host_name("gather_mm_simd_f16")]]
[[kernel]] void gather_mm_simd<half, 32>(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device half* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#if defined(__HAVE_BFLOAT__)
template [[host_name("gather_mm_simd_bf16")]]
[[kernel]] void gather_mm_simd<bfloat, 32>(
    const device bfloat* A [[buffer(0)]],
    const device bfloat* B [[buffer(1)]],
    const device uint32_t* indices [[buffer(2)]],
    device bfloat* D [[buffer(3)]],
    const constant GatherMMParams* params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
#endif
