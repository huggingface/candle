// SPDX-License-Identifier: MIT
// C wrapper for AMD Composable Kernel Flash Attention
// Created by: TEAM-509 (ROCm Flash Attention C wrapper)

#include <hip/hip_runtime.h>
#include <stdint.h>

// C interface for Flash Attention
extern "C" {

// Forward declaration of CK Flash Attention function
// This will be linked from the CK library
void run_fmha_fwd_ck(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* softmax_lse_ptr,
    const void* alibi_slopes_ptr,
    
    const int32_t* cu_seqlens_q_ptr,
    const int32_t* cu_seqlens_k_ptr,
    
    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,
    
    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    
    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    
    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,
    
    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,
    
    int32_t is_bf16,
    int32_t is_causal,
    int32_t unpadded_lse,
    
    int32_t window_size_left,
    int32_t window_size_right,
    
    float softcap,
    
    hipStream_t stream
);

// Rust FFI-compatible wrapper
void run_mha_rocm(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* softmax_lse_ptr,
    const void* alibi_slopes_ptr,
    
    const int32_t* cu_seqlens_q_ptr,
    const int32_t* cu_seqlens_k_ptr,
    
    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,
    
    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    
    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    
    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,
    
    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,
    
    int32_t is_bf16,
    int32_t is_causal,
    int32_t unpadded_lse,
    
    int32_t window_size_left,
    int32_t window_size_right,
    
    float softcap
) {
    // Get current HIP stream (or use default)
    hipStream_t stream = 0; // Default stream for now
    
    // Call CK Flash Attention
    run_fmha_fwd_ck(
        q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr, alibi_slopes_ptr,
        cu_seqlens_q_ptr, cu_seqlens_k_ptr,
        q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride, alibi_slopes_batch_stride,
        q_row_stride, k_row_stride, v_row_stride, o_row_stride,
        q_head_stride, k_head_stride, v_head_stride, o_head_stride,
        b, h, h_k, d, d_rounded, softmax_scale,
        seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
        is_bf16, is_causal, unpadded_lse,
        window_size_left, window_size_right,
        softcap,
        stream
    );
}

} // extern "C"
