//! The `cfg.rs` -> `air` bridge, on functions built the way the lowerer builds them.
//!
//! The case that matters is a LOOP: `cfg.rs` carries a carry phi, pliron carries a block
//! argument, and the branch has to forward the incoming. If phi index and block-argument
//! index ever disagree, the emitter's phi rebuild silently reorders operands - so this
//! test asserts the correspondence directly rather than trusting the translation.

#![cfg(feature = "pliron-air")]

use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::op::Op;
use pliron::printable::Printable;
use wax_metal::air_dialect::from_cfg::{BridgeErr, build};
use wax_metal::ir_builder::LlType;
use wax_metal::lower::cfg::{FuncBuilder, Operand, Terminator};

/// Straight-line arithmetic: every instruction kind the bridge maps without control flow.
#[test]
fn straight_line_function_translates() {
    let mut f = FuncBuilder::new("straight");
    let p = f.param("x", LlType::Float);
    f.append_block("entry");
    let a = f.fadd(p, p);
    let b = f.fmul(a, Operand::f32(2.0));
    let c = f.fdiv(b, a);
    let _ = f.fsub(c, b);
    f.set_terminator(Terminator::Ret);

    let ctx = &mut Context::new();
    let out = build(ctx, &f, "straight").expect("must translate");
    out.func
        .get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("must verify");
    println!("{}", out.func.get_operation().disp(ctx));
}

/// A counted loop. The carry phi must appear as block argument 0 of the loop header, and
/// BOTH edges into the header must forward exactly one value.
#[test]
fn a_loop_phi_becomes_a_block_argument() {
    let mut f = FuncBuilder::new("loop");
    let n = f.param("n", LlType::I32);
    let entry = f.append_block("entry");
    let header = f.block("header");
    let body = f.block("body");
    let exit = f.block("exit");

    // entry: br header(0)
    f.switch_to(entry);
    f.set_terminator(Terminator::Br(header));

    // header(i): cond = i slt n; cond_br cond, body, exit
    f.switch_to(header);
    let i = f.phi(LlType::I32);
    let cond = f.icmp("slt", i, n);
    f.set_terminator(Terminator::CondBr {
        cond: cond.into(),
        then_: body,
        else_: exit,
    });

    // body: next = i + 1; br header(next)
    f.switch_to(body);
    let next = f.add(i, Operand::i32(1));
    f.set_terminator(Terminator::Br(header));

    f.switch_to(exit);
    f.set_terminator(Terminator::Ret);

    // Incomings the way the lowerer wires them: 0 from entry, `next` from the back edge.
    f.phi_incoming(i, Operand::i32(0), entry);
    f.phi_incoming(i, next, body);

    let ctx = &mut Context::new();
    let out = build(ctx, &f, "loop_kernel").expect("must translate");

    // The phi is block argument 0 of the header, not an instruction.
    let hdr = out.blocks[header.0 as usize];
    assert_eq!(
        hdr.deref(ctx).arguments().count(),
        1,
        "the carry phi must be the header's single block argument"
    );

    // Both predecessors forward exactly one value into the header.
    for (bi, label) in [(entry.0 as usize, "entry"), (body.0 as usize, "body")] {
        let term = out.blocks[bi]
            .deref(ctx)
            .get_terminator(ctx)
            .expect("block must be terminated");
        let fwd = collect_forwarded(ctx, term, out.blocks[header.0 as usize]);
        assert_eq!(fwd, 1, "{label} must forward the carry value");
    }

    out.func
        .get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("must verify");
    println!("{}", out.func.get_operation().disp(ctx));
}

/// How many values a terminator forwards to `target`.
fn collect_forwarded(
    ctx: &Context,
    term: pliron::context::Ptr<pliron::operation::Operation>,
    target: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
) -> usize {
    use pliron::operation::Operation;
    use wax_metal::air_dialect::ops::{BrOp, CondBrOp};
    let succs: Vec<_> = term.deref(ctx).successors().collect();
    let idx = succs
        .iter()
        .position(|s| *s == target)
        .expect("target must be a successor");
    // `successor_operands` comes from `BranchOpInterface` either way.
    if let Some(br) = Operation::get_op::<BrOp>(term, ctx) {
        br.successor_operands(ctx, idx).len()
    } else if let Some(cbr) = Operation::get_op::<CondBrOp>(term, ctx) {
        cbr.successor_operands(ctx, idx).len()
    } else {
        panic!("terminator is not a branch")
    }
}

/// The escape hatch is refused rather than half-translated.
#[test]
fn a_raw_instruction_is_refused() {
    let mut f = FuncBuilder::new("raw");
    let p = f.param("x", LlType::I32);
    f.append_block("entry");
    f.raw(Some(LlType::I32), "add i32 $0, 7", vec![p.into()]);
    f.set_terminator(Terminator::Ret);

    let ctx = &mut Context::new();
    match build(ctx, &f, "raw") {
        Err(BridgeErr::RawInstruction(t)) => assert!(t.contains("add i32")),
        Err(e) => panic!("expected a raw-instruction refusal, got {e}"),
        Ok(_) => panic!("a raw instruction must not translate"),
    }
}

/// PHI INCOMING ORDER IS ASCENDING BY PREDECESSOR BLOCK, and the emitter depends on it.
///
/// `to_bitcode` writes a phi's incomings in the order the lowerer registered them. The
/// `air` form does not store that order - a block argument is positional and the incoming
/// lives on each predecessor's branch - so rebuilding phi records means scanning
/// predecessors in block order. That reproduces the original order only while the lowerer
/// registers incomings in that same order.
///
/// It does today: a loop wires `(preheader, latch)` and the preheader is always created
/// first, and an if/else wires `(then_end, else_end)` with the then arm translated first.
/// This pins both shapes so a future lowering change that registers them out of order
/// fails here rather than silently reordering operands in the emitted bitcode.
#[test]
fn phi_incomings_are_in_ascending_predecessor_order() {
    // Loop shape: preheader then latch.
    let mut f = FuncBuilder::new("loop_shape");
    let entry = f.append_block("entry");
    let header = f.block("header");
    let latch = f.block("latch");
    let exit = f.block("exit");
    f.switch_to(entry);
    f.set_terminator(Terminator::Br(header));
    f.switch_to(header);
    let iv = f.phi(LlType::I32);
    let cond = f.icmp("slt", iv, Operand::i32(8));
    f.set_terminator(Terminator::CondBr {
        cond: cond.into(),
        then_: latch,
        else_: exit,
    });
    f.switch_to(latch);
    let next = f.add(iv, Operand::i32(1));
    f.set_terminator(Terminator::Br(header));
    f.switch_to(exit);
    f.set_terminator(Terminator::Ret);
    f.phi_incoming(iv, Operand::i32(0), entry);
    f.phi_incoming(iv, next, latch);

    let preds: Vec<u32> = f.phi_incomings(iv).iter().map(|(_, b)| b.0).collect();
    let mut sorted = preds.clone();
    sorted.sort_unstable();
    assert_eq!(
        preds, sorted,
        "loop phi incomings must be in ascending predecessor order"
    );
}
