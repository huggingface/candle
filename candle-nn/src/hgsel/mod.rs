//! HGSEL — Hash-based Sparse Expert Layer
//! 
//! Deterministic sparse Mixture of Experts layer for candle-nn.

mod multi_hash;
mod expert_bank;
mod combine;
mod layer;

pub use multi_hash::{MultiHashRouter, RoutingStats};
pub use expert_bank::{ExpertBank, ExpertView};
pub use combine::{CombineMode, CombineStrategy, CombineFactory};
pub use layer::{HGSELLayer, HGSELDiagnostics, ExpertLoadStats};