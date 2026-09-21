//! Builder-module parity for the atomic ops: `AtomicRMW` and `AtomicCAS` kernels built
//! directly with `wax_ir::builder` - the same route `tests/atomic.rs` uses for the text
//! path - compared as byte-identical `.metallib`s through BOTH routes. A `DIFF` is the
//! failure: a conversion that claims success while emitting a different lowering is a
//! wrong answer, not a gap.

use wax_ir::builder::{append_op, OpBuilder};
use wax_ir::{Attribute, Location, Module, Opcode, PointerType, ScalarType, TileElementType, TileType, Type};

/// `name(ptr: *mut E, val: E)`: `old = AtomicRMW(mode, scope=Device) ptr, val; return`.
///
/// The result is discarded - the interesting part is the intrinsic call and its spelling
/// (name, operand order, the constant tail, the token result).
fn rmw_module(name: &str, elem: ScalarType, mode: i32) -> Module {
    let mut m = Module::new(name);
    let ptr_ty = Type::Pointer(PointerType { pointee: elem.clone() });
    let scalar = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(elem.clone()),
    });
    let (entry, block, args) = m.begin_function(name, &[ptr_ty, scalar.clone()]);
    m.functions.push(entry);
    m.set_op_attr(entry, "sym_name", Attribute::String(name.to_string()));

    let (rmw_id, _res) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
        .operands([args[0], args[1]])
        .result(scalar.clone())
        .result(Type::Token)
        .attr("mode", Attribute::i32(mode as i64))
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m);
    append_op(&mut m, block, rmw_id);

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
    append_op(&mut m, block, ret);
    m
}

/// `name(ptr: *mut E, cmp: E, new: E)`: `ok = AtomicCAS(scope=Device) ptr, cmp, new; return`.
fn cas_module(name: &str, elem: ScalarType) -> Module {
    let mut m = Module::new(name);
    let ptr_ty = Type::Pointer(PointerType { pointee: elem.clone() });
    let scalar = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(elem.clone()),
    });
    let (entry, block, args) = m.begin_function(name, &[ptr_ty, scalar.clone(), scalar.clone()]);
    m.functions.push(entry);
    m.set_op_attr(entry, "sym_name", Attribute::String(name.to_string()));

    let (cas_id, _res) = OpBuilder::new(Opcode::AtomicCAS, Location::Unknown)
        .operands([args[0], args[1], args[2]])
        .result(scalar.clone())
        .result(Type::Token)
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m);
    append_op(&mut m, block, cas_id);

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
    append_op(&mut m, block, ret);
    m
}

/// Both routes, compared as bytes. Panics with the first differing offset on a `DIFF`.
fn both_routes_agree(m: &Module) {
    let profile = wax_metal::profile::TargetProfile::host();
    let cfg = wax_metal::lower_to_metallib_for(m, &profile)
        .unwrap_or_else(|e| panic!("CFG route failed: {e}"));
    // The same kernel-argument metadata the CFG route emits: user params plus the
    // implicit `_tp` / `_tg` builtins, in the order the bitcode writer expects.
    let func = *m.functions.first().expect("one entry function");
    let args = wax_metal::lower::kernel_args_for_module(&m, func);
    let conv = wax_metal::wax_to_air::lower_to_metallib_via_conversion(m, &args, &[], &profile)
        .unwrap_or_else(|e| panic!("conversion refused the kernel: {e}"));
    let first_diff = cfg
        .iter()
        .zip(&conv)
        .position(|(a, b)| a != b)
        .unwrap_or(cfg.len().min(conv.len()));
    assert_eq!(
        cfg, conv,
        "DIFF: conversion {} bytes vs CFG {}, first difference at {}",
        conv.len(),
        cfg.len(),
        first_diff
    );
}

/// mode 3: `@air.atomic.global.add.s.i32`.
#[test]
fn rmw_i32_add_matches_cfg() {
    both_routes_agree(&rmw_module("rmw_i32_add", ScalarType::I32, 3));
}

/// mode 2: `@air.atomic.global.xor.s.i32` - a non-add op part.
#[test]
fn rmw_i32_xor_matches_cfg() {
    both_routes_agree(&rmw_module("rmw_i32_xor", ScalarType::I32, 2));
}

/// mode 4: `@air.atomic.global.add.f32` - the float op part, no sign.
#[test]
fn rmw_f32_add_matches_cfg() {
    both_routes_agree(&rmw_module("rmw_f32_add", ScalarType::F32, 4));
}

/// mode 7: `@air.atomic.global.max.u.f32` - the unsigned op part on a float.
#[test]
fn rmw_f32_maxu_matches_cfg() {
    both_routes_agree(&rmw_module("rmw_f32_maxu", ScalarType::F32, 7));
}

/// f16: the `f16` air-type suffix and the half element type.
#[test]
fn rmw_f16_add_matches_cfg() {
    both_routes_agree(&rmw_module("rmw_f16_add", ScalarType::F16, 4));
}

/// `@air.atomic.global.cmpxchg.weak.i32` - the expected value's stack slot and the
/// seven-argument call.
#[test]
fn cas_i32_matches_cfg() {
    both_routes_agree(&cas_module("cas_i32", ScalarType::I32));
}

/// The float CAS: half-sized element, same stack slot and call shape.
#[test]
fn cas_f32_matches_cfg() {
    both_routes_agree(&cas_module("cas_f32", ScalarType::F32));
}