//! AIR emission for Apple's Metal Performance Primitives (MPP).
//! Grants access to faster primitives like `matmul2d`.
//!
//! Entry points are `air.externally_defined` declarations the driver resolves at pipeline
//! creation. Kernels are emitted as air ops ([`mpp_air`]) and lowered normally.

// TODO: MPP can do more than just GEMM.

mod abi;
mod air;
mod config;
mod lower;

pub use abi::{MppAbi, mpp_abi};
pub use air::mpp_air;
pub use config::{MppDtype, MppGemm};
pub use lower::{lower_mpp_gemm_air, lower_mpp_gemm_splitk};
