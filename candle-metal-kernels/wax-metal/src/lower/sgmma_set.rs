//! Which values are SGMMA fragment registers.
//!
//! This used to be inferred from SHAPE alone: any 2-D f32-ish tile whose extents were
//! both multiples of 8 was assumed to be a simdgroup-matrix fragment. That is not a
//! property of the shape - `[32, 32]` is equally the shape of a threadgroup-spanning
//! tile with one element per thread - so the inference claimed tiles that were never
//! near an MMA and made them impossible to load.
//!
//! A fragment is what an MMA touches, so that is what this asks. Seed from the MMA and
//! `sgmma_*` ops, then close over ops that PRESERVE the tile shape: a loop carry ties
//! the accumulator's `constant` init, the block argument, the yielded value and the
//! loop result into one value, and an elementwise op ties its operands to its result.
//! Everything in a connected component shares one register representation, so the
//! component is a fragment if any member is.
use std::collections::{HashMap, HashSet};

use wax_ir::{Opcode, Type};

use super::ir::{ValRef, WaxIr};

/// Ops whose operands and results are fragments by definition.
fn is_seed(opcode: Opcode) -> bool {
    matches!(opcode, Opcode::MmaF | Opcode::MmaI | Opcode::MmaFScaled)
}

/// Tile shape of `v`, if it is a tile at all.
fn tile_shape(ir: &WaxIr, v: ValRef) -> Option<Vec<i64>> {
    match ir.value_type(v) {
        Type::Tile(tt) => Some(tt.shape.clone()),
        _ => None,
    }
}

/// The values that must be represented as SGMMA fragment registers.
pub(crate) fn sgmma_values(ir: &WaxIr) -> HashSet<ValRef> {
    // Union-find over values, joined by shape-preserving ops.
    let mut parent: HashMap<ValRef, ValRef> = HashMap::new();
    fn find(parent: &mut HashMap<ValRef, ValRef>, v: ValRef) -> ValRef {
        let mut root = v;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        let mut cur = v;
        while let Some(&p) = parent.get(&cur) {
            if p == cur {
                break;
            }
            parent.insert(cur, root);
            cur = p;
        }
        root
    }
    fn union(parent: &mut HashMap<ValRef, ValRef>, a: ValRef, b: ValRef) {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    let mut seeds: Vec<ValRef> = Vec::new();
    for (i, op) in ir.ops.iter().enumerate() {
        let op_ref = super::ir::OpRef(i as u32);
        let _ = op_ref;
        // Every tile value this op touches, with its shape.
        let mut touched: Vec<(ValRef, Vec<i64>)> = Vec::new();
        for &o in &op.operands {
            if let Some(s) = tile_shape(ir, o) {
                touched.push((o, s));
            }
        }
        for r in op.results.iter().flatten() {
            if let Some(s) = tile_shape(ir, *r) {
                touched.push((*r, s));
            }
        }
        if is_seed(op.opcode) {
            seeds.extend(touched.iter().map(|(v, _)| *v));
        }
        // `sgmma_elem`/`sgmma_set_elem` name their fragment operand explicitly.
        if op.attributes.iter().any(|(k, _)| k == "sgmma_slot")
            && let Some(&first) = op.operands.first()
        {
            seeds.push(first);
        }
        // Join same-shaped tiles this op relates. Loads and stores are excluded: a
        // load's result comes from memory, not from another register, so joining it to
        // anything would leak the classification across a memory round trip.
        if matches!(
            op.opcode,
            Opcode::LoadViewTko | Opcode::StoreViewTko | Opcode::LoadPtrTko | Opcode::StorePtrTko
        ) {
            continue;
        }
        for i in 1..touched.len() {
            if touched[i].1 == touched[0].1 {
                union(&mut parent, touched[0].0, touched[i].0);
            }
        }

        // A loop carry is one value wearing four names: the init operand, the body's
        // block argument, the value yielded at the bottom, and the loop's result. They
        // must share a representation, so tie them POSITIONALLY - shape alone would
        // also join unrelated same-shaped tiles that happen to be carried alongside.
        //
        // `For` operands are `[start, end, step, init_0..]`; body args are
        // `[iv, carry_0..]`; results are the carries out. The body's terminator is
        // `Continue`, not `Yield` - matching only `Yield` silently never ties the
        // yielded value at all, which the same-shape rule below happens to cover here
        // but should not be relied on.
        if op.opcode == Opcode::For {
            let n_carry = op.result_types.len();
            if op.operands.len() >= 3 + n_carry
                && let Some(&body_region) = op.regions.first()
                && let Some(&body_block) = ir.region(body_region).blocks.first()
            {
                let args = &ir.block(body_block).args;
                let yielded: Option<Vec<ValRef>> = ir
                    .block(body_block)
                    .ops
                    .iter()
                    .map(|&o| ir.op(o))
                    .find(|o| matches!(o.opcode, Opcode::Continue | Opcode::Yield))
                    .map(|o| o.operands.clone());
                for i in 0..n_carry {
                    let init = op.operands[3 + i];
                    if let Some((arg, _)) = args.get(1 + i) {
                        union(&mut parent, init, *arg);
                    }
                    if let Some(res) = op.results.get(i).and_then(|r| *r) {
                        union(&mut parent, init, res);
                    }
                    if let Some(y) = yielded.as_ref().and_then(|y| y.get(i)) {
                        union(&mut parent, init, *y);
                    }
                }
            }
        }
    }

    // Block arguments are tied to the operands the enclosing op passes in, which the
    // loop's `touched` set above already covers for the carried values.
    for block in &ir.blocks {
        for (arg, ty) in &block.args {
            if let Type::Tile(_) = ty {
                parent.entry(*arg).or_insert(*arg);
            }
        }
    }

    let seed_roots: HashSet<ValRef> = seeds
        .iter()
        .map(|&v| {
            parent.entry(v).or_insert(v);
            find(&mut parent, v)
        })
        .collect();

    let all: Vec<ValRef> = parent.keys().copied().collect();
    let mut out: HashSet<ValRef> = HashSet::new();
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
