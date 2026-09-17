//! Compute kernel dispatch facades.

pub mod affine;
pub mod copy;
pub mod elementwise;
pub mod gather;
pub mod gemm;
pub mod reduce;

pub use affine::call_affine_f32;
pub use copy::call_copy_f32;
pub use elementwise::{call_elem_binary_f32, call_elem_unary_f32};
pub use gather::call_gather_f32;
pub use gemm::call_gemm_f32;
pub use reduce::{call_reduce_max_f32, call_reduce_sum_f32};
