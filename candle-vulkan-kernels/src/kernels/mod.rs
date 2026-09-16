//! Compute kernel dispatch facades.

pub mod affine;
pub mod reduce;

pub use affine::call_affine_f32;
pub use reduce::call_reduce_sum_f32;
