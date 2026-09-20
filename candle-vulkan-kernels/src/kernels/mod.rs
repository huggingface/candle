//! Compute kernel dispatch facades.

pub mod affine;
pub mod copy;
pub mod elementwise;
pub mod gather;
pub mod gemm;
pub mod gemv;
pub mod q4k;
pub mod q5q8;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod softmax;

pub use affine::call_affine_f32;
pub use copy::call_copy_f32;
pub use elementwise::{call_elem_binary_f32, call_elem_unary_f32};
pub use gather::call_gather_f32;
pub use gemm::call_gemm_f32;
pub use gemv::{call_gemv_f32, call_gemv_t_f32};
pub use q4k::{call_q4k_dequant_f32, call_q4k_qmatvec_f32, call_q6k_dequant_f32};
pub use q5q8::{call_q50_dequant_f32, call_q5k_dequant_f32, call_q80_dequant_f32};
pub use reduce::{call_reduce_max_f32, call_reduce_sum_f32};
pub use rms_norm::call_rms_norm_f32;
pub use rope::call_rope_f32;
pub use softmax::call_softmax_last_dim_f32;
