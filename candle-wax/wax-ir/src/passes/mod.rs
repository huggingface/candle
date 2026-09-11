//! Tile-level analyses and transforms.
//!
//! The pliron pass types are re-exported here so a backend can drive these passes without
//! taking its own dependency on pliron. This crate owns the framework relationship; a backend
//! naming `pliron::pass::*` directly would be coupled to our choice of IR framework rather
//! than to our IR.
pub use pliron::pass::{Analysis, AnalysisManager, Pass, PassResult};

// On attribute naming
//
// Currently passes record their conclusions on the IR by writing in `WaxAttrs`, which is then
// stored as a pliron attribute with the name `wax_attrs`. The plan is to flatten this to only
// use pliron attributes directly.
//
// Pliron attribute names must satisfy the regex `[a-zA-Z_][a-zA-Z0-9_]*`.
//
// Convention:
//  Attribute names from wax passes should be written `wax_<concept>_<field>`, where <concept>
//  names what the attribute decides (not which pass wrote it). For example `cross_loop_reuse`
//  adds the attribute `wax_cache_producer`, `wax_cache_bytes`, `wax_cache_consumer` and
//  `wax_cache_id`, because what they record is a caching decision.
//
// IR attributes outlive the pass. The backend reads the `wax_cache_producer` decision and caches,
// regardless of which pass produced the decision. If another pass reaches the same decision from
// another route the lowering does not need to be updated.
// There is nothing inherently wrong with using the same attribute names in different passes. In
// fact it simplifies lowering. It does, however, require coordination.

pub mod cross_loop_reuse;
