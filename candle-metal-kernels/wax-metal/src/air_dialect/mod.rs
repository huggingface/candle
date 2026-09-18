//! Phase A of the pliron migration: AIR modelled as a pliron dialect.
//!
//! Scope is the backend only - the emitter's instruction set, not the wax IR. The gate
//! is `tests/pliron_air_parity.rs`: the same kernels through the existing `cfg.rs` path
//! and through this one must produce BYTE-IDENTICAL metallibs.
//!
//! Off by default behind the `pliron-air` feature until that gate passes.

pub mod attrs;
pub mod from_cfg;
pub mod ops;
pub mod to_bitcode;
pub mod types;
