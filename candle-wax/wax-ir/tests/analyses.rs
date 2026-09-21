//! Analysis testing.
//!
//! Each test creates a minimal kernel required for the analysis it is testing.

use pliron::value::Value;
use std::collections::HashSet;
use wax_ir::builder::{OpBuilder, append_op, build_block};
use wax_ir::passes::fragment_set::fragment_values;
use wax_ir::passes::uniformity::divergent_values;
use wax_ir::{Attribute, Location, Module, Opcode, ScalarType, TileElementType, TileType, Type};

fn tile(shape: Vec<i64>) -> Type {
    Type::Tile(TileType {
        shape,
        element_type: TileElementType::Scalar(ScalarType::F32),
    })
}

/// Create single entry function with empty block.
fn shell(name: &str) -> (Module, wax_ir::OpId, wax_ir::build::BlockId) {
    let mut m = Module::new(name);
    let (entry, block, _) = m.begin_function(name, &[]);
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    (m, entry, block)
}

/// Multi-element tile load. All lanes get their own element, cannot be proven uniform.
fn load(m: &mut Module, block: wax_ir::build::BlockId, shape: Vec<i64>) -> Value {
    let (op, res) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
        .result(tile(shape))
        .build(m)
        .expect("fixture op builds");
    append_op(m, block, op);
    res[0]
}

fn finish(m: &mut Module, block: wax_ir::build::BlockId) {
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(m)
        .expect("fixture op builds");
    append_op(m, block, ret);
}

mod fragment_values {
    use super::*;

    #[test]
    fn mma_op_causes_fragment_values() {
        let (mut m, entry, block) = shell("mma");
        let t = tile(vec![8, 8]);
        let mk = |m: &mut Module| {
            let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                .result(t.clone())
                .attr("value", Attribute::i32(0))
                .build(m)
                .expect("fixture op builds");
            append_op(m, block, op);
            res[0]
        };
        let (a, b, c) = (mk(&mut m), mk(&mut m), mk(&mut m));
        let (mma, out) = OpBuilder::new(Opcode::MmaF, Location::Unknown)
            .operands([a, b, c])
            .result(t.clone())
            .build(&mut m)
            .expect("fixture op builds");
        append_op(&mut m, block, mma);
        finish(&mut m, block);

        let f = fragment_values(&m.ctx, entry);
        for (name, v) in [("a", a), ("b", b), ("c", c), ("out", out[0])] {
            assert!(f.contains(&v), "{name} should be a fragment.");
        }
    }

    #[test]
    fn no_fragment_values_without_mma() {
        for shape in [vec![8, 8], vec![16, 16], vec![32, 32]] {
            let (mut m, entry, block) = shell("elementwise");
            let t = tile(shape.clone());
            let mk = |m: &mut Module| {
                let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                    .result(t.clone())
                    .attr("value", Attribute::i32(0))
                    .build(m)
                    .expect("fixture op builds");
                append_op(m, block, op);
                res[0]
            };
            let (a, b) = (mk(&mut m), mk(&mut m));
            let (add, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operands([a, b])
                .result(t.clone())
                .build(&mut m)
                .expect("fixture op builds");
            append_op(&mut m, block, add);
            finish(&mut m, block);

            let f = fragment_values(&m.ctx, entry);
            assert!(f.is_empty(), "tile with no MMA must never be fragment");
        }
    }
}

mod uniformity {
    use super::*;

    #[test]
    fn only_single_element_loads_are_uniform() {
        let (mut m, entry, block) = shell("multi");
        let multi = load(&mut m, block, vec![32]);
        finish(&mut m, block);
        assert!(
            divergent_values(&m.ctx, entry).contains(&multi),
            "multi-element load can not be uniform"
        );

        let (mut m2, entry2, block2) = shell("single");
        let single = load(&mut m2, block2, vec![1]);
        finish(&mut m2, block2);
        assert!(
            !divergent_values(&m2.ctx, entry2).contains(&single),
            "single-element load must be uniform"
        );

        let (mut m3, entry3, block3) = shell("empty");
        finish(&mut m3, block3);
        assert!(
            divergent_values(&m3.ctx, entry3).is_empty(),
            "kernel without any kind of load must be uniform"
        );
    }

    #[test]
    fn divergence_propagates_through_operands() {
        let (mut m, entry, block) = shell("chain");
        let t = tile(vec![32]);
        let v = load(&mut m, block, vec![32]);
        let (a1, r1) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operands([v, v])
            .result(t.clone())
            .build(&mut m)
            .expect("fixture op builds");
        append_op(&mut m, block, a1);
        let (a2, r2) = OpBuilder::new(Opcode::MulF, Location::Unknown)
            .operands([r1[0], r1[0]])
            .result(t.clone())
            .build(&mut m)
            .expect("fixture op builds");
        append_op(&mut m, block, a2);
        finish(&mut m, block);

        let d = divergent_values(&m.ctx, entry);
        assert!(d.contains(&r1[0]), "must be divergent");
        assert!(d.contains(&r2[0]), "must be divergent");
    }

    #[test]
    fn reduce_over_lane_axis_converges_to_uniform() {
        let scalar = tile(vec![]);
        let row = tile(vec![4]);
        // Only the first case, a reduce over the lane axis, causes uniformity.
        for (dim, converges) in [(1i64, true), (0, false)] {
            let (mut m, entry, block) = shell("reduce");
            let i_res = vec![load(&mut m, block, vec![4, 32])];

            let (cb, cargs) = build_block(&mut m, &[scalar.clone(), scalar.clone()]);
            let (_, comb) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operands([cargs[0], cargs[1]])
                .result(scalar.clone())
                .build_and_append(&mut m, cb)
                .expect("fixture op builds");
            OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operands([comb[0]])
                .build_and_append(&mut m, cb)
                .expect("fixture op builds");

            let (red, out) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
                .operand(i_res[0])
                .result(row.clone())
                .attr("dim", Attribute::i32(dim))
                .region(vec![cb])
                .build(&mut m)
                .expect("fixture op builds");
            append_op(&mut m, block, red);
            finish(&mut m, block);

            let d: HashSet<Value> = divergent_values(&m.ctx, entry);
            assert!(d.contains(&i_res[0]), "load must be divergent (dim={dim})");
            if converges {
                assert!(
                    !d.contains(&out[0]),
                    "reduce over lane axis must be uniform"
                );
            } else {
                assert!(
                    d.contains(&out[0]),
                    "reduce not over lane axis must be divergent"
                );
            }
        }
    }
}
