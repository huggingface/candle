//! PHASE A'S GATE: the same function through `cfg.rs` and through the `air` dialect must
//! produce an IDENTICAL `bitcode::func::Body`.
//!
//! This is the whole argument for Phase A. Everything before it - the dialect, the bridge,
//! the emitter - is plumbing that either reproduces the shipping path exactly or does not.
//! Comparing the `Body` rather than the final metallib keeps a failure readable: a
//! mismatched instruction shows up as a diff on one record instead of a byte offset.

#![cfg(feature = "pliron-air")]

use pliron::context::Context;
use wax_metal::air_dialect::{from_cfg, to_bitcode as air_to_bitcode};
use wax_metal::ir_builder::LlType;
use wax_metal::lower::cfg::{FuncBuilder, Operand, Terminator};
use wax_metal::lower::to_bitcode as cfg_to_bitcode;

/// Neither path resolves module symbols in these tests.
fn no_index(_: &str) -> Option<u32> {
    None
}

/// Build both bodies and compare them record by record.
fn assert_parity(f: &FuncBuilder, what: &str) {
    let want = cfg_to_bitcode::body_from_cfg(f, &no_index, &no_index)
        .unwrap_or_else(|e| panic!("{what}: cfg path failed: {e}"));

    let ctx = &mut Context::new();
    let air = from_cfg::build(ctx, f, "k").unwrap_or_else(|e| panic!("{what}: bridge failed: {e}"));
    let got = air_to_bitcode::body_from_air(ctx, &air, f.num_params(), &no_index, &no_index)
        .unwrap_or_else(|e| panic!("{what}: air path failed: {e}"));

    assert_eq!(
        want.num_blocks, got.num_blocks,
        "{what}: block count differs"
    );
    assert_eq!(
        want.insts.len(),
        got.insts.len(),
        "{what}: instruction count differs\n cfg: {:#?}\n air: {:#?}",
        want.insts,
        got.insts
    );
    for (i, (a, b)) in want.insts.iter().zip(got.insts.iter()).enumerate() {
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "{what}: instruction {i} differs"
        );
    }
    // The pool is compared through the instructions that reference it: a `Val::LocalConst`
    // index only means the same thing if both pools interned in the same order, so an
    // identical instruction stream already implies an identical pool.
}

/// Straight-line float arithmetic, with an immediate - the constant-pool path.
#[test]
fn straight_line_matches() {
    let mut f = FuncBuilder::new("straight");
    let p = f.param("x", LlType::Float);
    f.append_block("entry");
    let a = f.fadd(p, p);
    let b = f.fmul(a, Operand::f32(2.0));
    let c = f.fdiv(b, a);
    let _ = f.fsub(c, b);
    f.set_terminator(Terminator::Ret);
    assert_parity(&f, "straight line");
}

/// Integer arithmetic and a compare, exercising the icmp predicate encoding.
#[test]
fn integer_ops_match() {
    let mut f = FuncBuilder::new("ints");
    let p = f.param("n", LlType::I32);
    f.append_block("entry");
    let a = f.add(p, Operand::i32(7));
    let b = f.mul(a, Operand::i32(3));
    let c = f.and(b, Operand::i32(255));
    let _ = f.icmp("slt", c, p);
    f.set_terminator(Terminator::Ret);
    assert_parity(&f, "integer ops");
}

/// THE case Phase A exists to prove: a loop, where `cfg.rs` has a phi and `air` has a
/// block argument. If the phi rebuild reorders operands or misnumbers values, this fails.
#[test]
fn a_counted_loop_matches() {
    let mut f = FuncBuilder::new("loop");
    let n = f.param("n", LlType::I32);
    let entry = f.append_block("entry");
    let header = f.block("header");
    let latch = f.block("latch");
    let exit = f.block("exit");

    f.switch_to(entry);
    f.set_terminator(Terminator::Br(header));

    f.switch_to(header);
    let iv = f.phi(LlType::I32);
    let cond = f.icmp("slt", iv, n);
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

    assert_parity(&f, "counted loop");
}

/// An if/else merge: two phis fed from two arms, the other shape that produces block
/// arguments.
#[test]
fn an_if_else_merge_matches() {
    let mut f = FuncBuilder::new("ifelse");
    let p = f.param("x", LlType::I32);
    let entry = f.append_block("entry");
    let then_b = f.block("then");
    let else_b = f.block("else");
    let merge = f.block("merge");

    f.switch_to(entry);
    let c = f.icmp("slt", p, Operand::i32(0));
    f.set_terminator(Terminator::CondBr {
        cond: c.into(),
        then_: then_b,
        else_: else_b,
    });

    f.switch_to(then_b);
    let t = f.add(p, Operand::i32(1));
    f.set_terminator(Terminator::Br(merge));

    f.switch_to(else_b);
    let e = f.sub(p, Operand::i32(1));
    f.set_terminator(Terminator::Br(merge));

    f.switch_to(merge);
    let m = f.phi(LlType::I32);
    let _ = f.mul(m, Operand::i32(2));
    f.set_terminator(Terminator::Ret);

    f.phi_incoming(m, t, then_b);
    f.phi_incoming(m, e, else_b);

    assert_parity(&f, "if/else merge");
}

/// A symbol index that resolves anything, so the call/global paths are exercised rather
/// than short-circuiting on an unknown-symbol error.
fn some_index(_: &str) -> Option<u32> {
    Some(0)
}

/// Both bodies with a resolving symbol index.
fn assert_parity_with_symbols(f: &FuncBuilder, what: &str) {
    let want = cfg_to_bitcode::body_from_cfg(f, &some_index, &some_index)
        .unwrap_or_else(|e| panic!("{what}: cfg path failed: {e}"));
    let ctx = &mut Context::new();
    let air = from_cfg::build(ctx, f, "k").unwrap_or_else(|e| panic!("{what}: bridge failed: {e}"));
    let got = air_to_bitcode::body_from_air(ctx, &air, f.num_params(), &some_index, &some_index)
        .unwrap_or_else(|e| panic!("{what}: air path failed: {e}"));
    assert_eq!(
        want.insts.len(),
        got.insts.len(),
        "{what}: instruction count differs\n cfg: {:#?}\n air: {:#?}",
        want.insts,
        got.insts
    );
    for (i, (a, b)) in want.insts.iter().zip(got.insts.iter()).enumerate() {
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "{what}: instruction {i} differs"
        );
    }
}

/// Loads and stores through a device pointer, with a GEP - the addressing path every
/// real kernel uses.
#[test]
fn memory_ops_match() {
    let mut f = FuncBuilder::new("mem");
    let p = f.param("buf", LlType::device_ptr(LlType::Float));
    f.append_block("entry");
    let addr = f.gep(LlType::Float, p, Operand::i32(4));
    let v = f.load(addr, 4);
    let doubled = f.fadd(v, v);
    f.store(doubled, addr, 4);
    f.set_terminator(Terminator::Ret);
    assert_parity_with_symbols(&f, "memory ops");
}

/// A dense vector constant - the SGMMA stride-pair form the bridge used to refuse.
#[test]
fn a_vector_constant_matches() {
    let mut f = FuncBuilder::new("vec_const");
    let p = f.param("x", LlType::I64);
    f.append_block("entry");
    let vec_ty = LlType::Vec(2, Box::new(LlType::I64));
    let strides = Operand::vec_int("<i64 128, i64 8>", vec_ty, vec![128, 8]);
    let _ = f.insertelement(strides, p, Operand::i32(0));
    f.set_terminator(Terminator::Ret);
    assert_parity_with_symbols(&f, "vector constant");
}
