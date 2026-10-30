pub mod err;
pub mod kernel;
pub mod kernels;
pub mod push_constants;
pub mod source;

pub use err::VulkanKernelError;
pub use kernel::{trace_counts, KernelName, Kernels};
pub use kernels::{
    call_add_rms_norm_slang, call_affine_coopmat_slang, call_affine_slang, call_binary_slang,
    call_cmp_slang, call_const_fill_bytes, call_const_set_bf16, call_const_set_f16, call_const_set_f32, call_const_set_f64,
    call_const_set_f8e4m3, call_const_set_i64, call_const_set_u32, call_const_set_u8, call_conv_slang,
    call_copy2d_slang, call_decode_attention_slang, call_dequantize_slang,
    call_fused_quant_gemm_slang, call_gather_idx_slang, call_gemm_slang, call_index_add_slang,
    call_kv_row_copy_slang, call_layer_norm_slang, call_pool2d_slang,
    call_post_rms_norm_add_slang, call_reduce_slang, call_rms_norm_slang,
    call_rope_slang, call_rotary_emb_int_slang, call_scatter_slang, call_sigmoid_slang,
    call_softmax_slang, call_test_fill_bf16_emulated, call_test_fill_bf16_native,
    call_test_fill_f16, call_test_fill_f32, call_test_fill_f4, call_test_fill_f64,
    call_test_fill_f6e2m3, call_test_fill_f6e3m2, call_test_fill_f8e4m3, call_test_fill_f8e8m0,
    call_test_fill_i16, call_test_fill_i32, call_test_fill_i64, call_test_fill_u32,
    call_test_fill_u8, call_test_printf, call_to_dtype_slang, call_unary_slang,
    call_upsample_slang, call_where_slang, call_zero_fill, DECODE_ATTENTION_CHUNK,
};
pub use push_constants::{
    AffineParams, ConstFillBytesParams, ConstSetParams, DecodeAttentionParams,
    KvRowCopyParams, NormParams,
    ToDtypeParams, UnaryParams, UpsampleParams,
};
pub use source::Source;
