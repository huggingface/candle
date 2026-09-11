//! Wax IR - Cuda Tile compatible tile-level IR
//!
//! The wax IR is based on the IR of cutile-rs. The Rust DSL is identical, and the tile IR concepts
//! are meant to be equal to those defined by CUDA Tile IR spec itself.
//!
//! A wax [`Module`] can be built from the Rust DSL through [`builder::OpBuilder`].
//! The [`WaxSource`] trait acts as a bridge between components that process wax IR.
//! The IR is materialized into [`WaxIr`], which can be lowered to any wax compatible backend.

pub mod attr;
pub mod build;
pub mod builder;
pub mod dialect;
pub mod global;
pub mod ir;
pub mod location;
pub mod opcode;
pub mod passes;
pub mod source;
pub mod types;

pub use attr::{
    Attribute, Bounded, DenseElements, DivBy, FloatBits, OptimizationHints, SameElements,
};
pub use build::{BlockId, Module, OpId, Region, RegionId, Value};
pub use global::{Global, SymbolVisibility};
pub use ir::{BlockRef, OpData, OpRef, Producer, RegionRef, ValRef, WaxIr};
pub use location::{DebugInfoLoc, DebugScope, Location};
pub use opcode::Opcode;
pub use source::WaxSource;
pub use types::*;
