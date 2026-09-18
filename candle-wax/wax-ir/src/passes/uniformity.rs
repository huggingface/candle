//! SIMD-group uniformity: which values every lane of a SIMD group holds identically.
//!
//! This is a divergence analysis run the usual way round - seed the values that are per-lane
//! by construction, propagate forward to a least fixpoint, and call everything left over
//! uniform. Doing it as a fixpoint rather than a peephole is what lets a loop carry come out
//! uniform: a running max joined with a `constant` init stays uniform even though it is
//! rewritten every iteration.
//!
//! Generic over [`WaxSource`] for the same reason as `fragment_set`: the answer is a property
//! of the IR, and a backend walking its own materialised form must not reach a different
//! conclusion from a conversion walking the dialect.

use std::collections::HashSet;

use crate::Opcode;
use crate::attr::Attribute;
use crate::source::WaxSource;
use crate::types::Type;

use super::fragment_set::for_each_op;

/// Element count of `v`'s tile type, or 1 for a scalar.
fn n_elems<S: WaxSource>(src: &S, v: S::Val) -> i64 {
    match src.value_type(v) {
        Type::Tile(tt) => tt.shape.iter().product(),
        _ => 1,
    }
}

/// An integer attribute of `op`.
fn attr_i64<S: WaxSource>(src: &S, op: S::Op, key: &str) -> Option<i64> {
    src.attributes(op).into_iter().find_map(|(k, v)| {
        if k != key {
            return None;
        }
        match v {
            Attribute::Integer(i, _) => Some(i),
            _ => None,
        }
    })
}

/// Every used result of `op`.
fn results_of<S: WaxSource>(src: &S, op: S::Op) -> Vec<S::Val> {
    let n = src.result_types(op).len();
    (0..n).filter_map(|i| src.op_result(op, i as u32)).collect()
}

/// A within-SIMD reduce lands the same value in every lane, so its result is uniform whatever
/// it read. A COLUMN reduce (`dim = 0` over a `[rows, 32]` tile) does not - it leaves one
/// result per lane - so it is not a convergence point.
fn is_convergence_point<S: WaxSource>(src: &S, op: S::Op) -> bool {
    if src.opcode(op) != Opcode::Reduce {
        return false;
    }
    let Some(&source) = src.operands(op).first() else {
        return false;
    };
    let column = matches!(
        src.value_type(source),
        Type::Tile(tt) if tt.shape.len() == 2 && tt.shape[1] == 32 && tt.shape[0] > 1
    ) && attr_i64(src, op, "dim") == Some(0);
    !column
}

/// Ops that produce a different value in every lane no matter what they are given.
fn is_divergence_seed<S: WaxSource>(src: &S, op: S::Op) -> bool {
    match src.opcode(op) {
        Opcode::Iota => true,
        // A load is per-lane exactly when its tile spreads over the lanes. A one-element tile
        // has no lane term, so every lane reads the same address and agrees.
        Opcode::LoadViewTko | Opcode::LoadPtrTko => {
            results_of(src, op).iter().any(|&r| n_elems(src, r) > 1)
        }
        _ => false,
    }
}

/// The values that differ across the lanes of a SIMD group.
pub fn divergent_values<S: WaxSource>(src: &S, root: S::Op) -> HashSet<S::Val> {
    let mut ops: Vec<S::Op> = Vec::new();
    for_each_op(src, root, &mut |op| ops.push(op));

    let mut div: HashSet<S::Val> = HashSet::new();
    for &op in &ops {
        if is_divergence_seed(src, op) {
            div.extend(results_of(src, op));
        }
    }
    // Least fixpoint: divergence only ever spreads, so iterate until it stops.
    loop {
        let before = div.len();
        for &op in &ops {
            if is_convergence_point(src, op) {
                continue;
            }
            let opcode = src.opcode(op);
            // `For` is excluded from the blanket rule on purpose: its operands are all the
            // carry inits at once, so "any operand divergent" would tie every carry to every
            // other one. A kernel carrying a uniform running max alongside divergent
            // accumulators loses exactly the optimisation this analysis exists for. Carries
            // are tied positionally below instead.
            if opcode != Opcode::For && src.operands(op).iter().any(|o| div.contains(o)) {
                div.extend(results_of(src, op));
            }
            // An `If` yields its results from the ARMS' TERMINATORS, which live in its
            // REGIONS - not in its operands. The blanket rule above only inspects operands, so
            // an arm that computed something divergent and yielded it left the result marked
            // UNIFORM. A column reduce over such a value then elided its redistribution and
            // handed every lane the same number: silently wrong, and only where the elision is
            // reachable at all, which is why it hid behind a threadgroup width.
            if opcode == Opcode::If {
                let regions = src.regions(op);
                let n_res = src.result_types(op).len();
                let mut opaque = regions.len() < 2; // no else arm: cannot see it
                for &region in regions.iter() {
                    let Some(&blk) = src.region_blocks(region).first() else {
                        opaque = true;
                        continue;
                    };
                    match src
                        .block_ops(blk)
                        .into_iter()
                        .find(|&o| matches!(src.opcode(o), Opcode::Yield | Opcode::Continue))
                    {
                        Some(term) => {
                            let term_ops = src.operands(term);
                            for i in 0..n_res {
                                if term_ops.get(i).is_some_and(|v| div.contains(v))
                                    && let Some(r) = src.op_result(op, i as u32)
                                {
                                    div.insert(r);
                                }
                            }
                        }
                        None => opaque = true,
                    }
                }
                // An arm we cannot see into could yield anything, so assume the worst - that
                // costs an elision, where guessing uniform costs correctness.
                if opaque {
                    div.extend(results_of(src, op));
                }
            }
            // A loop carry is one value under four names (see `fragment_set`): the init
            // operand, the body block argument, the value yielded at the bottom, and the loop
            // result. Divergence in any of them is divergence in all.
            if opcode == Opcode::For {
                let operands = src.operands(op);
                let n_carry = src.result_types(op).len();
                let Some(&body_region) = src.regions(op).first() else {
                    continue;
                };
                let Some(&body_block) = src.region_blocks(body_region).first() else {
                    continue;
                };
                let args = src.block_args(body_block);
                // The body's terminator carries the values fed back round. It may be
                // `Continue` rather than `Yield` - looking only for `Yield` finds nothing, and
                // then divergence computed inside the loop never reaches the carry, which is
                // silently unsound rather than merely imprecise.
                let yielded: Option<Vec<S::Val>> = src
                    .block_ops(body_block)
                    .into_iter()
                    .find(|&o| matches!(src.opcode(o), Opcode::Continue | Opcode::Yield))
                    .map(|o| src.operands(o));
                // No terminator found means no way to see what the body feeds back, so every
                // carry has to be assumed divergent.
                if yielded.is_none() {
                    for i in 0..n_carry {
                        div.extend(operands.get(3 + i).copied());
                        div.extend(args.get(1 + i).map(|(a, _)| *a));
                        div.extend(src.op_result(op, i as u32));
                    }
                    continue;
                }
                // The induction variable is uniform; the carries are tied positionally.
                for i in 0..n_carry {
                    let init = operands.get(3 + i).copied();
                    let arg = args.get(1 + i).map(|(a, _)| *a);
                    let yld = yielded.as_ref().and_then(|y| y.get(i).copied());
                    let res = src.op_result(op, i as u32);
                    let tied: Vec<S::Val> = [init, arg, yld, res].into_iter().flatten().collect();
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
