//! Wax IR - Cuda Tile compatible tile-level IR
//!
//! The wax IR is based on the IR of cutile-rs. The Rust DSL is identical, and the tile IR concepts
//! are meant to be equal to those defined by CUDA Tile IR spec itself.
//!
//! A wax [`Module`] can be built from the Rust DSL through [`builder::OpBuilder`].
//! The [`WaxSource`] trait acts as a bridge between components that process wax IR:
//! the same op readers run over the builder, over a pliron module, or over a dialect view.

pub mod attr;
pub mod build;
pub mod builder;
pub mod cache;
pub mod dialect;
pub mod from_source;
pub mod location;
pub mod opcode;
pub mod passes;
pub mod source;
pub mod types;

pub use attr::{
    Attribute, Bounded, DenseElements, DivBy, FloatBits, OptimizationHints, SameElements,
};
pub use build::{BlockId, Module, OpId, Value};
pub use cache::{CompileKey, cached_module};
pub use location::{DebugInfoLoc, DebugScope, Location};
pub use opcode::Opcode;
pub use source::WaxSource;
pub use types::*;
