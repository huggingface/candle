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
//! ```
//!
//! [`lower_to_metallib`] runs no external tools: [`bitcode`] writes the LLVM bitcode and
//! [`metallib`] the container, so kernels compile where no Metal toolchain exists. It
//! emits the pre-opaque-pointer dialect Apple's AIR reader requires. A stock LLVM writes
//! opaque pointers that reader cannot consume, so this is not replaceable by an LLVM dep.
//!
//! The route is the pliron dialect conversion in [`wax_to_air`]: the entry function's
//! `wax` ops are rewritten into `air` ops, and [`lower`]::to_bitcode emits the bitcode.

pub mod air_dialect;
pub mod wax_to_air;
/// The wax IR's own dialect, re-exported from the crate that defines it.
pub use wax_ir::dialect;
pub mod bitcode;
pub mod gemm;
pub mod launch;
pub mod lower;
pub mod metallib;
pub mod mpp;
pub mod profile;
pub mod types;

pub use gemm::{GemmSpec, Plan, Reason, Routing, route};
pub use launch::{Launch, LoweredKernel, SecondaryDispatch};
pub use lower::{lower_to_metallib, lower_to_metallib_for};
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
