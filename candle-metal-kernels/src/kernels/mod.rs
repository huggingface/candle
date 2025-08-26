mod macros;
pub mod mlx_gemm;
pub mod sort;
pub mod unary;

pub use mlx_gemm::{call_mlx_gemm, GemmDType};
pub use sort::{call_arg_sort, call_mlx_arg_sort};
pub use unary::{
    call_const_set_contiguous, call_const_set_contiguous_tiled, call_const_set_strided,
    call_unary_contiguous, call_unary_contiguous_tiled, call_unary_strided,
};
