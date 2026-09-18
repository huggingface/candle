//! Which values must be represented as matrix-fragment registers.
//!
//! This used to be inferred from SHAPE alone: any 2-D f32-ish tile whose extents were both
//! multiples of 8 was assumed to be a fragment. That is not a property of the shape - `[32,
//! 32]` is equally the shape of a threadgroup-spanning tile with one element per thread - so
//! the inference claimed tiles that were never near an MMA and made them impossible to load.
//!
//! A fragment is what an MMA touches, so that is what this asks. Seed from the MMA ops, then
//! close over ops that PRESERVE the tile shape: a loop carry ties the accumulator's `constant`
//! init, the block argument, the yielded value and the loop result into one value, and an
//! elementwise op ties its operands to its result. Everything in a connected component shares
//! one register representation, so the component is a fragment if any member is.
//!
//! Generic over [`WaxSource`] on purpose. The classification is a property of the IR, not of a
//! backend, and two copies of it would drift: the Metal backend asks this of its materialised
//! `WaxIr` while a dialect conversion asks it of the pliron module, and both must get the same
//! answer or they disagree about what a register holds.

use std::collections::{HashMap, HashSet};

use crate::Opcode;
use crate::source::WaxSource;
use crate::types::Type;

/// Ops whose operands and results are fragments by definition.
fn is_seed(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::MmaF | Opcode::MmaI | Opcode::MmaFScaled)
}

/// The values that must be represented as fragment registers.
pub fn fragment_values<S: WaxSource>(src: &S, root: S::Op) -> HashSet<S::Val> {
    let mut parent: HashMap<S::Val, S::Val> = HashMap::new();

    fn find<V: Copy + Eq + std::hash::Hash>(parent: &mut HashMap<V, V>, v: V) -> V {
        let mut r = v;
        while let Some(&p) = parent.get(&r) {
            if p == r {
                break;
            }
            r = p;
        }
        // Path compression, so a long carry chain does not make this quadratic.
        let mut cur = v;
        while let Some(&p) = parent.get(&cur) {
            if p == cur {
                break;
            }
            parent.insert(cur, r);
            cur = p;
        }
        r
    }

    fn union<V: Copy + Eq + std::hash::Hash>(parent: &mut HashMap<V, V>, a: V, b: V) {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    let tile_shape = |v: S::Val| -> Option<Vec<i64>> {
        match src.value_type(v) {
            Type::Tile(tt) => Some(tt.shape.clone()),
            _ => None,
        }
    };

    let mut seeds: Vec<S::Val> = Vec::new();
    let mut blocks: Vec<S::Block> = Vec::new();
    for_each_op(src, root, &mut |op| {
        let opcode = src.opcode(op);
        // Every tile value this op touches, with its shape.
        let mut touched: Vec<(S::Val, Vec<i64>)> = Vec::new();
        for o in src.operands(op) {
            if let Some(s) = tile_shape(o) {
                touched.push((o, s));
            }
        }
        let n_results = src.result_types(op).len();
        for i in 0..n_results {
            if let Some(r) = src.op_result(op, i as u32)
                && let Some(s) = tile_shape(r)
            {
                touched.push((r, s));
            }
        }
        if is_seed(opcode) {
            seeds.extend(touched.iter().map(|(v, _)| *v));
        }
        // An op that names its fragment operand explicitly.
        if src.attributes(op).iter().any(|(k, _)| k == "sgmma_slot")
            && let Some(&first) = src.operands(op).first()
        {
            seeds.push(first);
        }
        // Join same-shaped tiles this op relates. Loads and stores are excluded: a load's
        // result comes from memory, not from another register, so joining it to anything
        // would leak the classification across a memory round trip.
        if matches!(
            opcode,
            Opcode::LoadViewTko | Opcode::StoreViewTko | Opcode::LoadPtrTko | Opcode::StorePtrTko
        ) {
            return;
        }
        for i in 1..touched.len() {
            if touched[i].1 == touched[0].1 {
                union(&mut parent, touched[0].0, touched[i].0);
            }
        }

        // A loop carry is one value wearing four names: the init operand, the body's block
        // argument, the value yielded at the bottom, and the loop's result. They must share a
        // representation, so tie them POSITIONALLY - shape alone would also join unrelated
        // same-shaped tiles that happen to be carried alongside.
        //
        // `For` operands are `[start, end, step, init_0..]`; body args are `[iv, carry_0..]`.
        // The body's terminator may be `Continue` rather than `Yield`; matching only `Yield`
        // silently never ties the yielded value at all.
        if opcode == Opcode::For {
            let operands = src.operands(op);
            let n_carry = n_results;
            if operands.len() >= 3 + n_carry
                && let Some(&body_region) = src.regions(op).first()
                && let Some(&body_block) = src.region_blocks(body_region).first()
            {
                let args = src.block_args(body_block);
                let yielded: Option<Vec<S::Val>> = src
                    .block_ops(body_block)
                    .into_iter()
                    .find(|&o| matches!(src.opcode(o), Opcode::Continue | Opcode::Yield))
                    .map(|o| src.operands(o));
                for i in 0..n_carry {
                    let init = operands[3 + i];
                    if let Some((arg, _)) = args.get(1 + i) {
                        union(&mut parent, init, *arg);
                    }
                    if let Some(res) = src.op_result(op, i as u32) {
                        union(&mut parent, init, res);
                    }
                    if let Some(y) = yielded.as_ref().and_then(|y| y.get(i)) {
                        union(&mut parent, init, *y);
                    }
                }
            }
        }
    });

    // Every tile-typed block argument is at least its own component, so a carry that nothing
    // joins still appears in the output rather than being absent.
    for_each_block(src, root, &mut |b| blocks.push(b));
    for b in blocks {
        for (arg, ty) in src.block_args(b) {
            if let Type::Tile(_) = ty {
                parent.entry(arg).or_insert(arg);
            }
        }
    }

    let seed_roots: HashSet<S::Val> = seeds
        .iter()
        .map(|&v| {
            parent.entry(v).or_insert(v);
            find(&mut parent, v)
        })
        .collect();

    let all: Vec<S::Val> = parent.keys().copied().collect();
    let mut out: HashSet<S::Val> = HashSet::new();
    for v in all {
        if seed_roots.contains(&find(&mut parent, v)) {
            out.insert(v);
        }
    }
    for &v in &seeds {
        out.insert(v);
    }
    out
}

/// Every operation under `root`, outermost first.
pub(crate) fn for_each_op<S: WaxSource>(src: &S, root: S::Op, f: &mut impl FnMut(S::Op)) {
    for r in src.regions(root) {
        for b in src.region_blocks(r) {
            for op in src.block_ops(b) {
                f(op);
                for_each_op(src, op, f);
            }
        }
    }
}

/// Every block under `root`, including the root's own.
pub(crate) fn for_each_block<S: WaxSource>(src: &S, root: S::Op, f: &mut impl FnMut(S::Block)) {
    for r in src.regions(root) {
        for b in src.region_blocks(r) {
            f(b);
            for op in src.block_ops(b) {
                for_each_block(src, op, f);
            }
        }
    }
}
