use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha_fwd(
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
        use_gqa_packing: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        total_q: u32,
        total_k: u32,

        rescale_threshold: f32,
        deterministic: c_int,
        use_2cta_mode: c_int,
        num_sm: c_int,

        stream: *mut c_void,
    );

    pub(crate) fn run_mha_bwd(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        dout_ptr: *const c_void,
        dq_ptr: *const c_void,
        dk_ptr: *const c_void,
        dv_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        dsoftmax_sum_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        dout_batch_stride: u32,
        dq_batch_stride: u32,
        dk_batch_stride: u32,
        dv_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,
        dout_row_stride: u32,
        dq_row_stride: u32,
        dk_row_stride: u32,
        dv_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,
        dout_head_stride: u32,
        dq_head_stride: u32,
        dk_head_stride: u32,
        dv_head_stride: u32,

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
        use_gqa_packing: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        total_q: u32,
        total_k: u32,

        rescale_threshold: f32,
        deterministic: c_int,
        use_2cta_mode: c_int,
        num_sm: c_int,

        stream: *mut c_void,
    );
}
