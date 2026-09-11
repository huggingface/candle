//! The `wax` IR modelled as a pliron dialect.
//!
//! An in-tree representation of a tile-level module. Tiles, views, structured ontrol flow.
//! [`crate::WaxSource`] let's us drive a backend directly, and is what the frontend builds into.
//!
//! Modelling as a dialect rather than a closed opcode enum keeps the vocabulary open.
//! Ops carry operands, results, attributes and regions in pliron's structures, and
//! [`crate::Opcode`] labels an op rather than defining it.
//! Adding a backend-specific op is a registration rather than a change to a shared enum.
//!
//! `ops` and `attrs` register the ops, `types` and `attr_mirror` bridge this crate's [`crate::Type`]
//! and [`crate::Attribute`] to pliron, `make` builds ops from an [`crate::Opcode`], and `source`
//! implements [`crate::WaxSource`] over it.
pub mod attr_mirror;
pub mod attrs;
pub mod make;
pub mod ops;
pub mod source;
pub mod types;
