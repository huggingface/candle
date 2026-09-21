//! Split-K reduction passes.
//!
//! [`rewrite`] edits an existing K-loop kernel to produce `splits` slices.
//! [`reduce`] builds a split-k reduce kernel, consuming `splits` slices.
//!
//! Both use [`Partials`], which is stamped by the first pass, and read by the second.
//!
//! Building the split-k kernel can be done directly, without the first pass. See [`build`].

pub mod partials;
pub mod reduce;
pub mod rewrite;

pub use partials::{ATTR_PARTIALS, ATTR_REDUCE, Partials};
pub use reduce::{SplitKReduce, SplitKReducePass, build};
pub use rewrite::SplitKPass;
