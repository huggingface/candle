//! SIMD-group uniformity: which values every lane of a SIMD group holds identically.
//!
//! This is a divergence analysis run the usual way round - seed the values that are
//! per-lane by construction, propagate forward to a least fixpoint, and call everything
//! left over uniform. Doing it as a fixpoint rather than a peephole is what lets a loop
//! carry come out uniform: the decode SDPA's running max is a `constant` joined with a
//! within-SIMD `reduce`, and neither is divergent, so the carry converges to uniform
//! only once the cycle has been walked to stability.
//!
//! Soundness direction matters. Calling a divergent value uniform silently drops work
//! that was load-bearing, so every rule here errs towards divergent: an op is uniform
//! only if it is a convergence point or all of its inputs are uniform.
use std::collections::HashSet;

use wax_ir::{Opcode, Type};

use super::ir::{ValRef, WaxIr};

/// Element count of `v`'s tile type, or 1 for a scalar.
fn n_elems(ir: &WaxIr, v: ValRef) -> i64 {
    match ir.value_type(v) {
        Type::Tile(tt) => tt.shape.iter().product(),
        _ => 1,
    }
}

/// A within-SIMD reduce lands the same value in every lane, so its result is uniform
/// whatever it read. A COLUMN reduce (`dim = 0` over a `[rows, 32]` tile) does not - it
/// leaves one result per lane - so it is not a convergence point.
fn is_convergence_point(ir: &WaxIr, op: &super::ir::OpData) -> bool {
    if op.opcode != Opcode::Reduce {
        return false;
    }
    let Some(&src) = op.operands.first() else {
        return false;
    };
    let column = matches!(
        ir.value_type(src),
        Type::Tile(tt) if tt.shape.len() == 2 && tt.shape[1] == 32 && tt.shape[0] > 1
    ) && super::attr_i32(op, "dim") == Some(0);
    !column
}

/// Ops that produce a different value in every lane no matter what they are given.
fn is_divergence_seed(ir: &WaxIr, op: &super::ir::OpData) -> bool {
    match op.opcode {
        Opcode::Iota => true,
        // A load is per-lane exactly when its tile spreads over the lanes. A one-element
        // tile has no lane term, so every lane reads the same address and agrees.
        Opcode::LoadViewTko | Opcode::LoadPtrTko => {
            op.results.iter().flatten().any(|&r| n_elems(ir, r) > 1)
        }
        _ => false,
    }
}

/// The values that differ across the lanes of a SIMD group.
pub(crate) fn divergent_values(ir: &WaxIr) -> HashSet<ValRef> {
    let mut div: HashSet<ValRef> = HashSet::new();
    for op in ir.ops.iter() {
        if is_divergence_seed(ir, op) {
            div.extend(op.results.iter().flatten().copied());
        }
    }
    // Least fixpoint: divergence only ever spreads, so iterate until it stops.
    loop {
        let before = div.len();
        for op in ir.ops.iter() {
            if is_convergence_point(ir, op) {
                continue;
            }
            // `For` is excluded from the blanket rule on purpose: its operands are all
            // the carry inits at once, so "any operand divergent" would tie every carry
            // to every other one. The decode SDPA carries a uniform running max
            // alongside four divergent accumulators, and conflating them costs exactly
            // the optimization this analysis exists for. Carries are tied positionally
            // below instead.
            if op.opcode != Opcode::For && op.operands.iter().any(|o| div.contains(o)) {
                div.extend(op.results.iter().flatten().copied());
            }
            // An `If` yields its results from the ARMS' TERMINATORS, which live in its
            // REGIONS - not in `op.operands`. The blanket rule above only inspects
            // operands, so an arm that computed something divergent and yielded it left
            // the result marked UNIFORM. A column reduce over such a value then elided its
            // redistribution and handed every lane the same number: silently wrong, and
            // only where the elision is reachable at all, which is why it hid behind a
            // threadgroup width.
            if op.opcode == Opcode::If {
                let n_res = op.result_types.len();
                let mut opaque = op.regions.len() < 2; // no else arm: cannot see it
                for &region in op.regions.iter() {
                    let Some(&blk) = ir.region(region).blocks.first() else {
                        opaque = true;
                        continue;
                    };
                    match ir
                        .block(blk)
                        .ops
                        .iter()
                        .map(|&o| ir.op(o))
                        .find(|o| matches!(o.opcode, Opcode::Yield | Opcode::Continue))
                    {
                        Some(term) => {
                            for i in 0..n_res {
                                if term.operands.get(i).is_some_and(|v| div.contains(v))
                                    && let Some(r) = op.results.get(i).and_then(|r| *r)
                                {
                                    div.insert(r);
                                }
                            }
                        }
                        None => opaque = true,
                    }
                }
                // An arm we cannot see into could yield anything, so assume the worst -
                // that costs an elision, where guessing uniform costs correctness.
                if opaque {
                    div.extend(op.results.iter().flatten().copied());
                }
            }
            // A loop carry is one value under four names (see `sgmma_set`): the init
            // operand, the body block argument, the value yielded at the bottom, and the
            // loop result. Divergence in any of them is divergence in all.
            if op.opcode == Opcode::For {
                let n_carry = op.result_types.len();
                let Some(&body_region) = op.regions.first() else {
                    continue;
                };
                let Some(&body_block) = ir.region(body_region).blocks.first() else {
                    continue;
                };
                let args = ir.block(body_block).args.clone();
                // The body's terminator carries the values fed back round. It is
                // `Continue` here, not `Yield` - looking only for `Yield` finds nothing,
                // and then divergence computed inside the loop never reaches the carry,
                // which is silently unsound rather than merely imprecise.
                let yielded: Option<Vec<ValRef>> = ir
                    .block(body_block)
                    .ops
                    .iter()
                    .map(|&o| ir.op(o))
                    .find(|o| matches!(o.opcode, Opcode::Continue | Opcode::Yield))
                    .map(|o| o.operands.clone());
                // No terminator found means no way to see what the body feeds back, so
                // every carry has to be assumed divergent.
                if yielded.is_none() {
                    for i in 0..n_carry {
                        div.extend(op.operands.get(3 + i).copied());
                        div.extend(args.get(1 + i).map(|(a, _)| *a));
                        div.extend(op.results.get(i).and_then(|r| *r));
                    }
                    continue;
                }
                // The induction variable is uniform; the carries are tied positionally.
                for i in 0..n_carry {
                    let init = op.operands.get(3 + i).copied();
                    let arg = args.get(1 + i).map(|(a, _)| *a);
                    let yld = yielded.as_ref().and_then(|y| y.get(i).copied());
                    let res = op.results.get(i).and_then(|r| *r);
                    let tied: Vec<ValRef> = [init, arg, yld, res].into_iter().flatten().collect();
                    if tied.iter().any(|v| div.contains(v)) {
                        div.extend(tied);
                    }
                }
            }
        }
        if div.len() == before {
            return div;
        }
    }
}
