use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::op::Op;
use pliron::printable::Printable;
use wax_ir::dialect::attrs::{BinaryAttr, UnaryAttr};
use wax_ir::dialect::ops::*;
use wax_ir::dialect::types::to_pliron;
use wax_ir::{ScalarType, TileElementType, TileType, Type};

fn f32_ty(ctx: &mut Context) -> pliron::r#type::TypeHandle {
    to_pliron(ctx, &Type::Scalar(ScalarType::F32)).unwrap()
}
fn i32_ty(ctx: &mut Context) -> pliron::r#type::TypeHandle {
    to_pliron(ctx, &Type::Scalar(ScalarType::I32)).unwrap()
}
fn tile_ty(ctx: &mut Context, shape: Vec<i64>) -> pliron::r#type::TypeHandle {
    to_pliron(
        ctx,
        &Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(ScalarType::F32),
        }),
    )
    .unwrap()
}

/// The uniform families map opcodes onto typed attributes exhaustively.
#[test]
fn opcode_families_map() {
    use wax_ir::Opcode;
    assert_eq!(
        BinaryAttr::from_opcode(Opcode::AddF),
        Some(BinaryAttr::AddF)
    );
    assert_eq!(
        BinaryAttr::from_opcode(Opcode::XOrI),
        Some(BinaryAttr::XOrI)
    );
    assert_eq!(
        UnaryAttr::from_opcode(Opcode::Rsqrt),
        Some(UnaryAttr::Rsqrt)
    );
    // A non-family opcode is not silently accepted
    assert_eq!(BinaryAttr::from_opcode(Opcode::LoadViewTko), None);
    // A different family's opcode is not silently accepted.
    assert_eq!(BinaryAttr::from_opcode(Opcode::Rsqrt), None);
    assert_eq!(UnaryAttr::from_opcode(Opcode::AddF), None);
}

/// A structed counted loop with a carried tile. One op, one region, the carry as a body
/// block argument.
#[test]
fn a_structured_loop_builds_and_verifies() {
    let ctx = &mut Context::new();
    let i32t = i32_ty(ctx);
    let f32t = f32_ty(ctx);
    let tile = tile_ty(ctx, vec![1, 32]);

    let fn_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "structured".try_into().unwrap(), fn_ty);
    let entry = func.get_entry_block(ctx);

    let lo = ConstantOp::new(ctx, i32t, "0");
    let hi = ConstantOp::new(ctx, i32t, "8");
    let step = ConstantOp::new(ctx, i32t, "1");
    let init = ConstantOp::new(ctx, tile, "splat 0.0");
    for o in [&lo, &hi, &step, &init] {
        o.get_operation().insert_at_back(entry, ctx);
    }
    let (lo_v, hi_v, step_v, init_v) = (
        lo.get_operation().deref(ctx).get_result(0),
        hi.get_operation().deref(ctx).get_result(0),
        step.get_operation().deref(ctx).get_result(0),
        init.get_operation().deref(ctx).get_result(0),
    );

    // The loop that carries one tile.
    let for_op = ForOp::new(
        ctx,
        vec![tile],
        Bounds {
            lo: lo_v,
            hi: hi_v,
            step: step_v,
        },
        vec![init_v],
        i32t,
        vec![tile],
    );
    for_op.get_operation().insert_at_back(entry, ctx);

    // acc = acc + acc; yield acc.
    let body = for_op.body(ctx);
    let acc = body.deref(ctx).get_argument(1);
    let add = BinaryOp::new(ctx, tile, acc, acc);
    set_binary(add.get_operation(), ctx, BinaryAttr::AddF);
    add.get_operation().insert_at_back(body, ctx);
    let sum = add.get_operation().deref(ctx).get_result(0);
    YieldOp::new(ctx, vec![sum])
        .get_operation()
        .insert_at_back(body, ctx);

    ReturnOp::new(ctx)
        .get_operation()
        .insert_at_back(entry, ctx);

    // The induction variable and the carry are body block arguments.
    assert_eq!(
        body.deref(ctx).arguments().count(),
        2,
        "body takes (induction var, carry)"
    );
    // And the loop is one op in the entry block, not a cluster of branches.
    let entry_ops: Vec<_> = {
        use pliron::linked_list::ContainsLinkedList;
        body.deref(ctx);
        entry.deref(ctx).iter(ctx).collect()
    };
    assert_eq!(
        entry_ops.len(),
        6,
        "4 constants + loop + return. No branches"
    );

    func.get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("must verify");

    // Full IR verification
    pliron::operation::verify_operation(func.get_operation(), ctx)
        .expect("a structured loop must satisfy value dominance");

    println!("{}", func.get_operation().disp(ctx));
    let _ = f32t;
}

/// A conditional keeps both arms as regions.
#[test]
fn a_structured_if_has_two_arms() {
    let ctx = &mut Context::new();
    let i32t = i32_ty(ctx);
    let tile = tile_ty(ctx, vec![1, 32]);

    let fn_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "cond".try_into().unwrap(), fn_ty);
    let entry = func.get_entry_block(ctx);

    let c = ConstantOp::new(ctx, i32t, "1");
    c.get_operation().insert_at_back(entry, ctx);
    let cv = c.get_operation().deref(ctx).get_result(0);

    let if_op = IfOp::new(ctx, vec![tile], cv, true);
    if_op.get_operation().insert_at_back(entry, ctx);

    for arm in 0..2 {
        let b = if_op.arm(ctx, arm);
        let k = ConstantOp::new(ctx, tile, if arm == 0 { "splat 1.0" } else { "splat 2.0" });
        k.get_operation().insert_at_back(b, ctx);
        let v = k.get_operation().deref(ctx).get_result(0);
        YieldOp::new(ctx, vec![v])
            .get_operation()
            .insert_at_back(b, ctx);
    }

    ReturnOp::new(ctx)
        .get_operation()
        .insert_at_back(entry, ctx);

    assert_eq!(
        if_op.get_operation().deref(ctx).num_regions(),
        2,
        "then and else are both regions"
    );
    func.get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("must verify");
    println!("{}", func.get_operation().disp(ctx));
}
