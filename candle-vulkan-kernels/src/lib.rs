pub mod err;
pub mod kernel;
pub mod kernels;
pub mod source;

pub use err::VulkanKernelError;
pub use kernel::{trace_counts, KernelName, Kernels};
pub use kernels::{
    call_affine_slang_f32, call_const_set_f32,
    call_gemm_slang_f32, call_test_fill_bf16_emulated, call_test_fill_bf16_native,
    call_test_fill_f16, call_test_fill_f32, call_test_fill_f4, call_test_fill_f64,
    call_test_fill_f6e2m3, call_test_fill_f6e3m2, call_test_fill_f8e4m3, call_test_fill_f8e8m0,
    call_test_fill_i16, call_test_fill_i32, call_test_fill_i64, call_test_fill_u32,
    call_test_fill_u8, call_binary_slang_f32, call_cmp_slang_f32, call_unary_slang_f32,
    call_where_slang_f32,
    call_copy2d_slang_f32,
    call_gather_idx_slang_f32,
    call_reduce_slang_f32,
    call_pool2d_slang_f32,
    call_upsample_slang_f32,
    call_conv_slang_f32,
    call_scatter_slang_f32,
};
pub use source::Source;
