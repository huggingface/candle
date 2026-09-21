//! A SIMD group is uniform if every lane holds identical values. Otherwise it is diverse.
//!
//! If no divergence is found the op is fully uniform.

use crate::Opcode;
use crate::dialect::source::{
    attr_i64, opcode_of, operands_of, ops_and_blocks_under, results_of, value_type,
};
use crate::passes::fragment_set::iter_value_names;
use crate::types::Type;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::pass::{Analysis, AnalysisManager};
use pliron::result::Result;
use pliron::value::Value;
use std::collections::HashSet;

/// Element count of `v`'s tile type, or 1 for a scalar.
fn n_elems(ctx: &Context, v: Value) -> i64 {
    match value_type(ctx, v) {
        Some(Type::Tile(tt)) => tt.shape.iter().product(),
        _ => 1,
    }
}

/// A reduce that collapses the lane axis is uniform.
///
/// The tile model defines the innermost axis as the lane axis. A reduce with no
/// `dim` implies tile to scalar - uniform by construction.
fn is_convergence_point(ctx: &Context, op: Ptr<Operation>) -> bool {
    if opcode_of(ctx, op) != Opcode::Reduce {
        return false;
    }
    let Some(&source) = operands_of(ctx, op).first() else {
        return false;
    };
    let Some(Type::Tile(tt)) = value_type(ctx, source) else {
        return true;
    };
    match attr_i64(ctx, op, "dim") {
        None => true,
        Some(d) => usize::try_from(d).is_ok_and(|d| d == tt.shape.len() - 1),
    }
}

/// Ops whose output can not be proven uniform.
fn is_divergence_seed(ctx: &Context, op: Ptr<Operation>) -> bool {
    match opcode_of(ctx, op) {
        Opcode::Iota => true,
        // A generic load can obviously usually not be proven uniform, but a single element tile
        // means every lane gets the same value.
        Opcode::LoadViewTko | Opcode::LoadPtrTko => {
            results_of(ctx, op).iter().any(|&r| n_elems(ctx, r) > 1)
        }
        _ => false,
    }
}

/// Values that differ amongst the lanes of a SIMD group.
pub fn divergent_values(ctx: &Context, root: Ptr<Operation>) -> HashSet<Value> {
    let (ops, _) = ops_and_blocks_under(ctx, root);

    let mut div: HashSet<Value> = HashSet::new();
    for &op in &ops {
        if is_divergence_seed(ctx, op) {
            div.extend(results_of(ctx, op));
        }
    }
    // Loop until divergent value set is unchanged.
    loop {
        let before = div.len();
        for &op in &ops {
            if is_convergence_point(ctx, op) {
                continue;
            }
            let opcode = opcode_of(ctx, op);
            // `For` handled separately because otherwise uniform ops in a loop alongside divergent
            // accumulators would be considered divergent.
            if opcode != Opcode::For && operands_of(ctx, op).iter().any(|o| div.contains(o)) {
                div.extend(results_of(ctx, op));
            }
            // `If` also has to be handled seperately because the terminator results live in
            // regions, not its operands.
            if opcode == Opcode::If {
                use pliron::linked_list::ContainsLinkedList;
                let regions: Vec<_> = op.deref(ctx).regions().collect();
                let results = results_of(ctx, op);
                let mut opaque = regions.len() < 2; // no else arm: cannot see it
                for &region in regions.iter() {
                    let Some(blk) = region.deref(ctx).iter(ctx).next() else {
                        opaque = true;
                        continue;
                    };
                    match blk
                        .deref(ctx)
                        .iter(ctx)
                        .find(|&o| matches!(opcode_of(ctx, o), Opcode::Yield | Opcode::Continue))
                    {
                        Some(term) => {
                            let term_ops = operands_of(ctx, term);
                            for (i, &r) in results.iter().enumerate() {
                                if term_ops.get(i).is_some_and(|v| div.contains(v)) {
                                    div.insert(r);
                                }
                            }
                        }
                        None => opaque = true,
                    }
                }
                // An opaque arm cannot be proven uniform.
                if opaque {
                    div.extend(results);
                }
            }
            if opcode == Opcode::For {
                let iter_values = iter_value_names(ctx, op);
                for names in iter_values.groups {
                    // No terminator means we can't evaluate, so we assume divergent.
                    // Divergence in one iter value is a divergence in all.
                    if !iter_values.saw_yield || names.iter().any(|v| div.contains(v)) {
                        div.extend(names);
                    }
                }
            }
        }
        if div.len() == before {
            return div;
        }
    }
}

pub struct DivergentValues {
    pub values: HashSet<Value>,
}

impl Analysis for DivergentValues {
    fn name(&self) -> &str {
        "wax-divergent-values"
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self> {
        Ok(DivergentValues {
            values: divergent_values(ctx, op),
        })
    }
}
