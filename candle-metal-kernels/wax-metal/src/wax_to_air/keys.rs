//! Attribute keys this crate uses.
//!
//! `dict_key!` registers each name globally. Keys live here since wax-metal is the only user.
//! If that changes they should move to, for example, `wax-ir`.

pub(crate) use wax_ir::passes::cross_loop_reuse::{
    ATTR_CACHE_BYTES, ATTR_CACHE_CONSUMER, ATTR_CACHE_ID, ATTR_CACHE_PRODUCER,
};

pliron::dict_key!(
    /// Comparison predicate.
    ATTR_COMPARISON_PREDICATE,
    "comparison_predicate"
);
pliron::dict_key!(
    /// Dim a `wax.reduce` or `wax.scan` collapses.
    ATTR_DIM,
    "dim"
);
pliron::dict_key!(
    /// Atomic memory scope.
    ATTR_MEMORY_SCOPE,
    "memory_scope"
);
pliron::dict_key!(
    /// Atomic ordering mode.
    ATTR_MODE,
    "mode"
);
pliron::dict_key!(
    /// If a `wax.scan` runs backwards.
    ATTR_REVERSE,
    "reverse"
);
pliron::dict_key!(
    /// Rounding mode on `DivI`.
    ATTR_ROUNDING,
    "rounding"
);
pliron::dict_key!(
    /// Whether an integer op is signed or not.
    ATTR_SIGNEDNESS,
    "signedness"
);
pliron::dict_key!(
    /// Rows of threadgroup scratch.
    ATTR_TG_SCRATCH_ROWS,
    "tg_scratch_rows"
);
