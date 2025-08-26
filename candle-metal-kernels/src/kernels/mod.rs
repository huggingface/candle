pub mod binary;
pub mod cast;
mod macros;
pub mod mlx_gemm;
pub mod quantized;
pub mod sdpa;
pub mod sort;
pub mod unary;

pub use binary::{call_binary_contiguous, call_binary_strided};
pub use cast::{call_cast_contiguous, call_cast_strided};
pub use mlx_gemm::{call_mlx_gemm, GemmDType};
pub use quantized::{call_quantized_matmul_mm_t, call_quantized_matmul_mv_t, GgmlDType};
pub use sdpa::{call_sdpa_full, call_sdpa_vector, call_sdpa_vector_2pass};
pub use sort::{call_arg_sort, call_mlx_arg_sort};
pub use unary::{
    call_const_set_contiguous, call_const_set_contiguous_tiled, call_const_set_strided,
    call_unary_contiguous, call_unary_contiguous_tiled, call_unary_strided,
};
