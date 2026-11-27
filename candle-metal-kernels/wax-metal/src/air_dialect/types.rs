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

pub const TENSOR_T: &str = "struct._tensor_t";
pub const MATMUL2D_DESC: &str = "struct.mpp::tensor_ops::matmul2d_descriptor";

/// MPP tensor handle. `%struct._tensor_t = type opaque`.
#[pliron_type(name = "air.tensor_t", format, generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AirTensorHandleType;

/// MPP `matmul2d`'s descriptor struct.
#[pliron_type(
    name = "air.matmul2d_desc",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AirMatmul2dDescType;

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

#[cfg(test)]
mod tests {
    use crate::air_dialect::types::{self as t, AirPtrType};
    use pliron::context::Context;
    use pliron::printable::Printable;
    use pliron::r#type::TypedHandle;

    #[test]
    fn every_air_type_prints() {
        let ctx = &mut Context::new();
        let cases = [
            t::i1_ty(ctx),
            t::i8_ty(ctx),
            t::i16_ty(ctx),
            t::i32_ty(ctx),
            t::i64_ty(ctx),
            t::f16_ty(ctx),
            t::bf16_ty(ctx),
            t::f32_ty(ctx),
            t::f64_ty(ctx),
            t::void_ty(ctx),
            t::vec_ty(ctx, 4, t::f32_ty(ctx)),
            t::vec_ty(ctx, 64, t::f32_ty(ctx)),
            t::device_ptr_ty(ctx, t::f32_ty(ctx)),
            t::ptr_ty(ctx, t::vec_ty(ctx, 4, t::f16_ty(ctx)), 3),
        ];
        for h in &cases {
            let text = h.disp(ctx).to_string();
            assert!(!text.trim().is_empty(), "a type printed as nothing");
        }
    }

    #[test]
    fn identical_types_are_uniqued() {
        let ctx = &mut Context::new();
        let a = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        let b = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        assert_eq!(a, b);

        let p1 = t::ptr_ty(ctx, t::f32_ty(ctx), 1);
        let p2 = t::ptr_ty(ctx, t::f32_ty(ctx), 2);
        assert_ne!(p1, p2, "address space must be part of pointer identity");
    }

    #[test]
    fn pointers_keep_their_pointee_and_addrspace() {
        let ctx = &mut Context::new();
        let h = t::ptr_ty(ctx, t::f16_ty(ctx), 3);
        let p = TypedHandle::<AirPtrType>::from_handle(h, ctx).expect("must be an air.ptr");
        let p = p.deref(ctx);
        assert_eq!(p.addrspace(), 3);
        let pointee = p.pointee();
        assert_eq!(pointee.disp(ctx).to_string().trim(), "builtin.fp16");
    }

    #[test]
    fn vectors_keep_length_and_element() {
        let ctx = &mut Context::new();
        let h = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        let v = TypedHandle::<pliron_llvm::types::VectorType>::from_handle(h, ctx)
            .expect("must be an llvm.vector");
        let (elem, n) = {
            let v = v.deref(ctx);
            (v.elem_type(), v.num_elements())
        };
        assert_eq!(n, 64);
        assert_eq!(elem.disp(ctx).to_string().trim(), "builtin.fp32");
    }
}
