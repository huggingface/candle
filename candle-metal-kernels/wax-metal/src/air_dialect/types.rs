//! Types of the `air` dialect, and the helper constructors the lowering spells them with.
//!
//! Most types are simply re-exports of pliron's llvm types. The only type we define ourselves
//! is the typed pointer `air.ptr`, because pliron's `llvm.ptr` is an opaque pointer (>=LLVM 22),
//! and AIR's intrinsics require typed pointers.

use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{
    BF16Type, FP16Type, FP32Type, FP64Type, IntegerType, Signedness, UnitType,
};
use pliron::context::Context;
use pliron::derive::pliron_type;
use pliron::r#type::TypeHandle;

/// A pointer that knows its pointee and address space (1 device, 2 constant, 3 threadgroup).
#[pliron_type(
    name = "air.ptr",
    generate_get = true,
    format = "`<` $pointee ` addrspace(` $addrspace `)>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AirPtrType {
    pointee: TypeHandle,
    addrspace: u32,
}

impl AirPtrType {
    pub fn pointee(&self) -> TypeHandle {
        self.pointee
    }
    pub fn addrspace(&self) -> u32 {
        self.addrspace
    }
}

/// Named constructors for the `air` scalar types.
///
/// Here simply because `IntegerType::get(ctx, 32, Signedness::Signless)` is incredibly verbose.
pub fn i1_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 1, Signedness::Signless).into()
}
pub fn i8_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 8, Signedness::Signless).into()
}
pub fn i16_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 16, Signedness::Signless).into()
}
pub fn i32_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}
pub fn i64_ty(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
}
pub fn f16_ty(ctx: &Context) -> TypeHandle {
    FP16Type::get(ctx).into()
}
pub fn bf16_ty(ctx: &Context) -> TypeHandle {
    BF16Type::get(ctx).into()
}
pub fn f32_ty(ctx: &Context) -> TypeHandle {
    FP32Type::get(ctx).into()
}
pub fn f64_ty(ctx: &Context) -> TypeHandle {
    FP64Type::get(ctx).into()
}
pub fn void_ty(ctx: &Context) -> TypeHandle {
    UnitType::get(ctx).into()
}
pub fn vec_ty(ctx: &Context, n: u32, elem: TypeHandle) -> TypeHandle {
    pliron_llvm::types::VectorType::get(ctx, elem, n, pliron_llvm::types::VectorTypeKind::Fixed)
        .into()
}
pub fn ptr_ty(ctx: &Context, pointee: TypeHandle, addrspace: u32) -> TypeHandle {
    AirPtrType::get(ctx, pointee, addrspace).into()
}
/// Address space 1.
pub fn device_ptr_ty(ctx: &Context, pointee: TypeHandle) -> TypeHandle {
    ptr_ty(ctx, pointee, 1)
}
/// Address space 2, where a constant-space buffer parameter binds.
pub fn const_ptr_ty(ctx: &Context, pointee: TypeHandle) -> TypeHandle {
    ptr_ty(ctx, pointee, 2)
}

/// Check if an `air` type is a float
pub fn is_float(ctx: &Context, h: TypeHandle) -> bool {
    pliron::r#type::type_impls::<dyn FloatTypeInterface>(&*h.deref(ctx))
}
