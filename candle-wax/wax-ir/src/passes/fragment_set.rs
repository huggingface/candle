//! Detect values that should be represented as matrix-fragment registers.
//!
//! The requirement is simply that the values must have been used by an eligible operation,
//! namely `MmaF`, `MmaI`, or `MmaFScaled`.
//!
//! A value assembled or consumed elementwise, with no MMA in the chain, is invisible to the pass.

use crate::Opcode;
use crate::dialect::source::{
    opcode_of, operands_of, ops_and_blocks_under, results_of, value_type,
};
use crate::types::Type;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::pass::{Analysis, AnalysisManager};
use pliron::result::Result;
use pliron::utils::union_find::UnionFind;
use pliron::value::Value;
use std::collections::HashSet;

/// Ops whose operands and results are fragments by definition.
fn is_seed(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::MmaF | Opcode::MmaI | Opcode::MmaFScaled)
}

/// The shape of `v`, when `v` is a tile. The only shape this pass joins on.
fn tile_shape(ctx: &Context, v: Value) -> Option<Vec<i64>> {
    match value_type(ctx, v)? {
        Type::Tile(tt) => Some(tt.shape),
        _ => None,
    }
}

/// All iter values of a `wax.for` op.
// TODO: Consider moving to utils. Consider `wax.loop` once implemented.
#[derive(Default)]
pub(crate) struct IterValues {
    /// One group per iter value, init first.
    pub groups: Vec<Vec<Value>>,
    /// Indicates that the body had a terminator to read the values from.
    // TODO: both `Yield` and `Continue` are valid. Rename to `has_terminator`?
    pub saw_yield: bool,
}

/// Extract iter value names from `was.for` op.
pub(crate) fn iter_value_names(ctx: &Context, for_op: Ptr<Operation>) -> IterValues {
    use pliron::linked_list::ContainsLinkedList;

    /// Past `[start, end, step, `.
    const FIRST_INIT: usize = 3;
    /// Past the induction variable.
    const FIRST_ARG: usize = 1;

    let operands = operands_of(ctx, for_op);
    let results = results_of(ctx, for_op);
    let n = results.len();
    // A loop with more results than it has inits doesn't work. Would index out of bounds.
    if operands.len() < FIRST_INIT + n {
        return IterValues::default();
    }
    let Some(body_region) = for_op.deref(ctx).regions().next() else {
        return IterValues::default();
    };
    let Some(body_block) = body_region.deref(ctx).iter(ctx).next() else {
        return IterValues::default();
    };
    let args: Vec<Value> = body_block.deref(ctx).arguments().collect();
    let yielded: Option<Vec<Value>> = body_block
        .deref(ctx)
        .iter(ctx)
        .find(|&o| matches!(opcode_of(ctx, o), Opcode::Continue | Opcode::Yield))
        .map(|o| operands_of(ctx, o));

    IterValues {
        saw_yield: yielded.is_some(),
        groups: (0..n)
            .map(|i| {
                let mut names = vec![operands[FIRST_INIT + i]];
                names.extend(args.get(FIRST_ARG + i).copied());
                names.push(results[i]);
                names.extend(yielded.as_ref().and_then(|y| y.get(i)).copied());
                names
            })
            .collect(),
    }
}

/// The values that must be represented as fragment registers.
pub fn fragment_values(ctx: &Context, root: Ptr<Operation>) -> HashSet<Value> {
    let mut uf: UnionFind<Value> = UnionFind::default();
    let mut seeds: Vec<Value> = Vec::new();

    let (ops, blocks) = ops_and_blocks_under(ctx, root);
    for op in ops {
        let opcode = opcode_of(ctx, op);
        // Every tile value this op touches, plus its shape.
        let mut touched: Vec<(Value, Vec<i64>)> = Vec::new();
        for v in operands_of(ctx, op).into_iter().chain(results_of(ctx, op)) {
            if let Some(s) = tile_shape(ctx, v) {
                touched.push((v, s));
            }
        }
        if is_seed(opcode) {
            seeds.extend(touched.iter().map(|(v, _)| *v));
        }
        // We want to union/join similarly shaped tiles related to this op, however loads and
        // stores are explicitly excluded since they interact with memory - not registers.
        if matches!(
            opcode,
            Opcode::LoadViewTko | Opcode::StoreViewTko | Opcode::LoadPtrTko | Opcode::StorePtrTko
        ) {
            continue;
        }
        // Joining tiles
        for i in 1..touched.len() {
            if touched[i].1 == touched[0].1 {
                uf.union(touched[0].0, touched[i].0);
            }
        }

        // Join together iter value names in the for loop groups.
        if opcode == Opcode::For {
            for names in iter_value_names(ctx, op).groups {
                for &v in &names[1..] {
                    uf.union(names[0], v);
                }
            }
        }
    }

    // Tile block args are tracked as their own sets so that unjoined iter values are not lost.
    // `find` registers these values.
    for b in blocks {
        let args: Vec<Value> = b.deref(ctx).arguments().collect();
        for arg in args {
            if let Some(Type::Tile(_)) = value_type(ctx, arg) {
                uf.find(arg);
            }
        }
    }

    // A value belongs in a fragment register when it is connected to a seed, here tracked by
    // `set_members`. A seed is always a member of its own set.
    let mut out: HashSet<Value> = HashSet::new();
    for &v in &seeds {
        out.extend(uf.set_members(v).iter().copied());
    }
    out
}

pub struct FragmentValues {
    pub values: HashSet<Value>,
}

impl Analysis for FragmentValues {
    fn name(&self) -> &str {
        "wax-fragment-values"
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self> {
        Ok(FragmentValues {
            values: fragment_values(ctx, op),
        })
    }
}
