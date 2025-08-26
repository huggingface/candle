pub mod mlx_gemm;
pub mod sort;

pub use mlx_gemm::{call_mlx_gemm, GemmDType};
pub use sort::{call_arg_sort, call_mlx_arg_sort};
