//! FFI bindings to AMD Composable Kernel Flash Attention
//!
//! Created by: TEAM-509 (ROCm Flash Attention FFI)
//! CUDA parity: candle-flash-attn/src/ffi.rs
//!
//! This module provides Rust FFI bindings to the C wrapper (csrc/fmha_wrapper.cpp)
//! which calls AMD's Composable Kernel Flash Attention implementation.

use core::ffi::{c_int, c_void};

extern "C" {
    /// Run Multi-Head Attention using AMD's Composable Kernel implementation
    /// 
    /// This function is implemented in csrc/fmha_wrapper.cpp and links to
    /// AMD's Composable Kernel library.
    /// 
    /// # Arguments
    /// 
    /// * `q_ptr` - Query tensor device pointer (shape: [b, h, s_q, d])
    /// * `k_ptr` - Key tensor device pointer (shape: [b, h_k, s_k, d])
    /// * `v_ptr` - Value tensor device pointer (shape: [b, h_k, s_k, d_v])
    /// * `o_ptr` - Output tensor device pointer (shape: [b, h, s_q, d_v])
    /// * `softmax_lse_ptr` - Log-sum-exp output for training (can be null)
    /// * `alibi_slopes_ptr` - ALiBi slopes for positional bias (can be null)
    /// * `cu_seqlens_q_ptr` - Cumulative sequence lengths for Q (group mode, can be null)
    /// * `cu_seqlens_k_ptr` - Cumulative sequence lengths for K/V (group mode, can be null)
    /// * `q_batch_stride` - Stride between batches in Q
    /// * `k_batch_stride` - Stride between batches in K
    /// * `v_batch_stride` - Stride between batches in V
    /// * `o_batch_stride` - Stride between batches in O
    /// * `alibi_slopes_batch_stride` - Stride between batches in ALiBi slopes
    /// * `q_row_stride` - Stride between rows (sequence) in Q
    /// * `k_row_stride` - Stride between rows (sequence) in K
    /// * `v_row_stride` - Stride between rows (sequence) in V
    /// * `o_row_stride` - Stride between rows (sequence) in O
    /// * `q_head_stride` - Stride between heads in Q
    /// * `k_head_stride` - Stride between heads in K
    /// * `v_head_stride` - Stride between heads in V
    /// * `o_head_stride` - Stride between heads in O
    /// * `b` - Batch size
    /// * `h` - Number of query heads
    /// * `h_k` - Number of key/value heads (for GQA/MQA)
    /// * `d` - Head dimension for Q/K
    /// * `d_rounded` - Rounded head dimension (multiple of 32)
    /// * `softmax_scale` - Scaling factor for attention scores (typically 1/sqrt(d))
    /// * `seqlen_q` - Query sequence length
    /// * `seqlen_k` - Key/Value sequence length
    /// * `seqlen_q_rounded` - Rounded query sequence length (multiple of 128)
    /// * `seqlen_k_rounded` - Rounded key sequence length (multiple of 128)
    /// * `is_bf16` - 1 if using BF16, 0 if using F16
    /// * `is_causal` - 1 for causal masking, 0 otherwise
    /// * `unpadded_lse` - 1 to use unpadded LSE, 0 otherwise
    /// * `window_size_left` - Left window size for sliding window attention (-1 for no limit)
    /// * `window_size_right` - Right window size for sliding window attention (-1 for no limit)
    /// * `softcap` - Softcapping value for attention logits (0.0 for no softcap)
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because:
    /// - All pointers must be valid HIP device pointers
    /// - Tensor dimensions must match the strides
    /// - Memory must be properly allocated on the device
    /// - This function assumes the current HIP stream
    pub(crate) fn run_mha_rocm(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,
        unpadded_lse: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_signature() {
        // This test just ensures the FFI signature compiles
        // Actual testing requires ROCm hardware
        let _ = run_mha_rocm as *const ();
    }
}
