//! The partials produced by a split K-loop and consumed by a split-K reduce.
//!
//! (Exact hyphen placement is intentional)

use crate::build::OpId;
use crate::types::ScalarType;
use pliron::context::Context;
use pliron::derive::pliron_attr;

pliron::dict_key!(
    /// Key to the partials attribute.
    ATTR_PARTIALS,
    "wax_split_partials"
);

pliron::dict_key!(
    /// Marks the function that reduces partials.
    ATTR_REDUCE,
    "wax_split_reduce"
);

/// Marks a function as the reduce half of a split.
#[pliron_attr(name = "wax.split_reduce", verifier = "succ", format)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ReduceMark;

/// The partials the reduce consumes.
#[pliron_attr(name = "wax.split_partials", verifier = "succ", format)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partials {
    /// Output dtype. Partials are f32.
    pub out_elem: ScalarType,
    /// Output rows.
    pub rows: i64,
    /// Output columns.
    pub cols: i64,
    /// Output row stride. Partials are packed.
    pub ldc: i64,
    pub splits: i32,
}

impl Partials {
    /// Stamp the partials attribute.
    pub fn stamp(&self, ctx: &Context, func: OpId) {
        func.deref_mut(ctx)
            .attributes
            .set(ATTR_PARTIALS.clone(), self.clone());
    }

    /// Extract the partials attribute, if present
    pub fn of(ctx: &Context, func: OpId) -> Option<Partials> {
        func.deref(ctx)
            .attributes
            .get::<Partials>(&ATTR_PARTIALS)
            .cloned()
    }

    /// Scratch size required. Always f32.
    pub fn scratch_elems(&self) -> usize {
        (self.splits as i64 * self.rows * self.cols) as usize
    }
}
