use core::ffi::c_void;

extern "C" {
    pub(crate) fn run_decode_attention(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        out_ptr: *const c_void,
        dtype: i32,
        batch: i32,
        num_heads: i32,
        num_heads_k: i32,
        seqlen_k: i32,
        head_dim: i32,
        q_b_stride: i64,
        q_h_stride: i64,
        k_b_stride: i64,
        k_h_stride: i64,
        k_l_stride: i64,
        v_b_stride: i64,
        v_h_stride: i64,
        v_l_stride: i64,
        o_b_stride: i64,
        o_h_stride: i64,
        scale: f32,
        stream: *mut c_void,
    );
}
