//! `air` dialect types: every [`LlType`] must map, and print the way AIR spells it.
//!
//! Cheap gate for Phase A's foundation - if a type cannot be represented the emitter
//! built on it cannot be either, and this fails long before the bitcode differential.

#![cfg(feature = "pliron-air")]

use pliron::context::Context;
use pliron::printable::Printable;
use pliron::r#type::TypedHandle;
use wax_metal::air_dialect::types::{AirPtrType, AirVecType, to_pliron};
use wax_metal::ir_builder::LlType;

/// Every variant maps, including nested vectors and pointers.
#[test]
fn every_lltype_maps() {
    let ctx = &mut Context::new();
    let cases = [
        LlType::I1,
        LlType::I8,
        LlType::I16,
        LlType::I32,
        LlType::I64,
        LlType::Half,
        LlType::BFloat,
        LlType::Float,
        LlType::Double,
        LlType::Void,
        LlType::Vec(4, Box::new(LlType::Float)),
        LlType::Vec(64, Box::new(LlType::Float)),
        LlType::Ptr {
            pointee: Box::new(LlType::Float),
            addrspace: 1,
        },
        LlType::Ptr {
            pointee: Box::new(LlType::Vec(4, Box::new(LlType::Half))),
            addrspace: 3,
        },
    ];
    for ty in &cases {
        let h = to_pliron(ctx, ty);
        println!("{:<28} -> {}", ty.ll_str(), h.disp(ctx));
    }
}

/// Types are UNIQUED: the same `LlType` must map to the same handle, or the emitter
/// would emit duplicate type-table entries and every relative value number would shift.
#[test]
fn identical_types_are_uniqued() {
    let ctx = &mut Context::new();
    let a = to_pliron(ctx, &LlType::Vec(64, Box::new(LlType::Float)));
    let b = to_pliron(ctx, &LlType::Vec(64, Box::new(LlType::Float)));
    assert_eq!(a, b);

    let p1 = to_pliron(
        ctx,
        &LlType::Ptr {
            pointee: Box::new(LlType::Float),
            addrspace: 1,
        },
    );
    let p2 = to_pliron(
        ctx,
        &LlType::Ptr {
            pointee: Box::new(LlType::Float),
            addrspace: 2,
        },
    );
    assert_ne!(p1, p2, "address space must be part of pointer identity");
}

/// THE reason this dialect exists: a pointer keeps its pointee. pliron's `llvm.ptr` is
/// opaque and would lose it, but Apple's pre-opaque-pointer AIR reader needs
/// `POINTER(pointee, addrspace)`.
#[test]
fn pointers_keep_their_pointee_and_addrspace() {
    let ctx = &mut Context::new();
    let h = to_pliron(
        ctx,
        &LlType::Ptr {
            pointee: Box::new(LlType::Half),
            addrspace: 3,
        },
    );
    let p = TypedHandle::<AirPtrType>::from_handle(h, ctx).expect("must be an air.ptr");
    let p = p.deref(ctx);
    assert_eq!(p.addrspace(), 3);
    let pointee = p.pointee();
    assert_eq!(pointee.disp(ctx).to_string().trim(), "builtin.fp16");
}

/// A vector keeps its length and element type.
#[test]
fn vectors_keep_length_and_element() {
    let ctx = &mut Context::new();
    let h = to_pliron(ctx, &LlType::Vec(64, Box::new(LlType::Float)));
    let v = TypedHandle::<AirVecType>::from_handle(h, ctx).expect("must be an air.vec");
    let v = v.deref(ctx);
    assert_eq!(v.n(), 64);
    assert_eq!(v.elem().disp(ctx).to_string().trim(), "builtin.fp32");
}
