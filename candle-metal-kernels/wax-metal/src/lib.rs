//! Lowers the wax IR (`wax_ir::Module`) to Apple AIR (LLVM IR targeting
//! `air64_vNN-apple-<os>`, resolved for the build's platform and deployment target by
//! [`profile::TargetProfile`]).
//!
//! # Pipeline
//!
//! ```text
//! #[wax::entry] Rust kernel
//!   -> KernelCompiler::compile() -> cutile_ir::Module
//!        -> lower_to_metallib()  -> .metallib            (in process, no tools)
//!        -> lower_to_air()       -> AIR text -> metal-as -> wrap_metallib()
//! ```
//!
//! [`lower_to_metallib`] runs no external tools: [`bitcode`] writes the LLVM bitcode and
//! [`metallib`] the container, so kernels compile where no Metal toolchain exists. It
//! emits the pre-opaque-pointer dialect Apple's AIR reader requires; a stock LLVM writes
//! opaque pointers that reader cannot consume, so this is not replaceable by an LLVM dep.
//!
//! [`lower_to_air`] is kept as the reference the in-process writer is checked against:
//! `wax-metal benches/tests/bitcode_vs_metal_as.rs` compiles both ways and requires
//! bit-identical GPU results.

#[cfg(feature = "pliron-air")]
pub mod air_dialect;
/// The wax IR's own dialect, re-exported from the crate that defines it.
pub use wax_ir::dialect;
pub mod bitcode;
pub mod gemm;
pub mod ir_builder;
pub mod launch;
pub mod lower;
pub mod metallib;
pub mod mpp;
pub mod profile;
mod types;

pub use gemm::{Backend, GemmSpec, Reason, Routing, route};
pub use launch::{Launch, LoweredKernel};
#[cfg(feature = "pliron-air")]
pub use lower::lower_to_metallib_via_air;
pub use lower::{
    lower_to_air, lower_to_air_for, lower_to_metallib, lower_to_metallib_for,
    lower_to_metallib_from_wax_ir,
};
pub use metallib::{Function as MetallibFunction, wrap as wrap_metallib};
pub use profile::{Platform, TargetProfile};

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("unimplemented op: {0}")]
    Unimplemented(String),
    #[error("missing attribute '{attr}' on {op}")]
    MissingAttr { op: String, attr: String },
    #[error("type error: {0}")]
    TypeError(String),
    #[error("malformed AIR: {0}")]
    MalformedIr(String),
}
