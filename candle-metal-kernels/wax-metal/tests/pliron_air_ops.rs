//! `air` dialect ops: buildable, verifiable, and printable.
//!
//! Phase A's second gate. Proves the op set can express a loop-shaped function - the case
//! that matters, because a loop is where `cfg.rs` uses phis and pliron uses BLOCK
//! ARGUMENTS instead. If that shape does not round-trip through the dialect, the bridge
//! cannot be written.


use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::{BranchOpInterface, OneRegionInterface};
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::op::Op;
use pliron::printable::Printable;
use wax_metal::air_dialect::attrs::{BinOpAttr, ICmpPredAttr};
use wax_metal::air_dialect::ops::{BinOp, BrOp, CondBrOp, ICmpOp, RetOp};
use wax_metal::air_dialect::types::to_pliron;
use wax_metal::ir_builder::LlType;

/// A counted loop: `entry -> body(i) -> body | exit`, with the induction variable as a
/// BLOCK ARGUMENT rather than a phi.
#[test]
fn a_counted_loop_builds_and_verifies() {
    let ctx = &mut Context::new();
    let i32_ty = to_pliron(ctx, &LlType::I32);
    let i1_ty = to_pliron(ctx, &LlType::I1);

    // Blocks must live in a function's region; pliron's verifier rejects an orphan block.
    let fn_ty = FunctionType::get(ctx, vec![], vec![]);
    let func = FuncOp::new(ctx, "loop_kernel".try_into().unwrap(), fn_ty);
    let region = func.get_region(ctx);
    let entry = func.get_entry_block(ctx);
    // The induction variable: one block argument, where cfg.rs would emit a phi.
    let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), vec![i32_ty]);
    let exit = BasicBlock::new(ctx, Some("exit".try_into().unwrap()), vec![]);
    body.insert_at_back(region, ctx);
    exit.insert_at_back(region, ctx);

    let i = body.deref(ctx).get_argument(0);

    // entry: seed the loop and branch into it. `i - i` stands in for a constant 0 so this
    // test needs no constant op yet.
    let zero = {
        let c = BinOp::new(ctx, i32_ty, i, i);
        c.set_op(ctx, BinOpAttr::Sub);
        c.get_operation().insert_at_back(entry, ctx);
        c.get_operation().deref(ctx).get_result(0)
    };
    let br = BrOp::new(ctx, body, vec![zero]);
    br.get_operation().insert_at_back(entry, ctx);

    // body(i): next = i + i; cond = next slt i; cond_br cond, body(next), exit
    let next = {
        let a = BinOp::new(ctx, i32_ty, i, i);
        a.set_op(ctx, BinOpAttr::Add);
        a.get_operation().insert_at_back(body, ctx);
        a.get_operation().deref(ctx).get_result(0)
    };
    let cond = {
        let c = ICmpOp::new(ctx, i1_ty, next, i);
        c.set_pred(ctx, ICmpPredAttr::Slt);
        c.get_operation().insert_at_back(body, ctx);
        c.get_operation().deref(ctx).get_result(0)
    };
    let cbr = CondBrOp::new(ctx, cond, body, vec![next], exit, vec![]);
    cbr.get_operation().insert_at_back(body, ctx);

    // exit: ret
    let ret = RetOp::new(ctx);
    ret.get_operation().insert_at_back(exit, ctx);

    // The successor operand IS the incoming value a phi would name - reachable through
    // the interface, which is what the bitcode emitter will use to rebuild phi records.
    let fwd = cbr.successor_operands(ctx, 0);
    assert_eq!(
        fwd.len(),
        1,
        "the back edge forwards the induction variable"
    );
    assert_eq!(fwd[0], next);
    assert!(
        cbr.successor_operands(ctx, 1).is_empty(),
        "the exit edge forwards nothing"
    );

    func.get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("the function must verify");
    println!("{}", func.get_operation().disp(ctx));
}

/// Operators are typed, so a wrong spelling cannot reach the emitter as a string.
#[test]
fn operator_spellings_round_trip() {
    for (text, want) in [
        ("add", BinOpAttr::Add),
        ("fmul", BinOpAttr::FMul),
        ("ashr", BinOpAttr::AShr),
        ("fdiv", BinOpAttr::FDiv),
    ] {
        assert_eq!(BinOpAttr::from_llvm(text), Some(want));
        assert_eq!(want.as_str(), text);
    }
    assert_eq!(BinOpAttr::from_llvm("not_an_operator"), None);
    assert_eq!(ICmpPredAttr::from_llvm("sle"), Some(ICmpPredAttr::Sle));
}
