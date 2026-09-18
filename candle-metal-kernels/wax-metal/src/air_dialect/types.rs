//! Types of the `air` dialect: what [`crate::ir_builder::LlType`] models, as pliron types.
//!
//! Three are defined here because pliron has no equivalent:
//!   `air.ptr`   TYPED pointer with an address space. pliron's `llvm.ptr` is
//!               opaque-pointer only (LLVM 22), but Apple's AIR reader is
//!               pre-opaque-pointer and needs `POINTER(pointee, addrspace)` records.
//!   `air.vec`   `<N x T>`; pliron has no vector type.
//!   `air.bf16`  pliron's builtin floats stop at fp16/fp32/fp64.
//! Integers and fp16/fp32/fp64 reuse `builtin`.

use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness, UnitType};
use pliron::context::Context;
use pliron::derive::{pliron_type, type_interface_impl};
use pliron::r#type::TypeHandle;
use pliron::utils::apfloat::{self, GetSemantics, Semantics};

use crate::ir_builder::LlType;

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

/// `<N x T>`.
#[pliron_type(
    name = "air.vec",
    generate_get = true,
    format = "`<` $n ` x ` $elem `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AirVecType {
    n: u32,
    elem: TypeHandle,
}

impl AirVecType {
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn elem(&self) -> TypeHandle {
        self.elem
    }
}

/// bfloat16. Same exponent range as f32, 8 fewer mantissa bits.
#[pliron_type(name = "air.bf16", format, generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct AirBF16Type;

#[type_interface_impl]
impl FloatTypeInterface for AirBF16Type {
    fn get_semantics(&self) -> Semantics {
        apfloat::BFloat::get_semantics()
    }
}

/// Translate an [`LlType`] into the `air` dialect.
///
/// One direction only: the emitter reads pliron types back through the accessors above,
/// so a round trip would just be re-deriving what the ops already carry.
pub fn to_pliron(ctx: &mut Context, ty: &LlType) -> TypeHandle {
    match ty {
        LlType::I1 => IntegerType::get(ctx, 1, Signedness::Signless).into(),
        LlType::I8 => IntegerType::get(ctx, 8, Signedness::Signless).into(),
        LlType::I16 => IntegerType::get(ctx, 16, Signedness::Signless).into(),
        LlType::I32 => IntegerType::get(ctx, 32, Signedness::Signless).into(),
        LlType::I64 => IntegerType::get(ctx, 64, Signedness::Signless).into(),
        LlType::Half => FP16Type::get(ctx).into(),
        LlType::BFloat => AirBF16Type::get(ctx).into(),
        LlType::Float => FP32Type::get(ctx).into(),
        LlType::Double => FP64Type::get(ctx).into(),
        LlType::Void => UnitType::get(ctx).into(),
        LlType::Vec(n, e) => {
            let elem = to_pliron(ctx, e);
            AirVecType::get(ctx, *n, elem).into()
        }
        LlType::Ptr { pointee, addrspace } => {
            let p = to_pliron(ctx, pointee);
            AirPtrType::get(ctx, p, *addrspace).into()
        }
    }
}

/// Translate an `air` type back to [`LlType`].
///
/// The bitcode emitter needs this direction: an op carries pliron type handles, but the
/// bitcode type table is built from `LlType`. (An earlier note here claimed one direction
/// sufficed - it does not, once the emitter reads types off ops rather than off `cfg.rs`.)
pub fn from_pliron(ctx: &Context, h: TypeHandle) -> Option<LlType> {
    use pliron::r#type::TypedHandle;
    if let Ok(p) = TypedHandle::<AirPtrType>::from_handle(h, ctx) {
        let p = p.deref(ctx);
        let pointee = from_pliron(ctx, p.pointee())?;
        return Some(LlType::Ptr {
            pointee: Box::new(pointee),
            addrspace: p.addrspace(),
        });
    }
    if let Ok(v) = TypedHandle::<AirVecType>::from_handle(h, ctx) {
        let v = v.deref(ctx);
        let elem = from_pliron(ctx, v.elem())?;
        return Some(LlType::Vec(v.n(), Box::new(elem)));
    }
    if TypedHandle::<AirBF16Type>::from_handle(h, ctx).is_ok() {
        return Some(LlType::BFloat);
    }
    if TypedHandle::<FP16Type>::from_handle(h, ctx).is_ok() {
        return Some(LlType::Half);
    }
    if TypedHandle::<FP32Type>::from_handle(h, ctx).is_ok() {
        return Some(LlType::Float);
    }
    if TypedHandle::<FP64Type>::from_handle(h, ctx).is_ok() {
        return Some(LlType::Double);
    }
    if TypedHandle::<UnitType>::from_handle(h, ctx).is_ok() {
        return Some(LlType::Void);
    }
    if let Ok(i) = TypedHandle::<IntegerType>::from_handle(h, ctx) {
        return match i.deref(ctx).width() {
            1 => Some(LlType::I1),
            8 => Some(LlType::I8),
            16 => Some(LlType::I16),
            32 => Some(LlType::I32),
            64 => Some(LlType::I64),
            _ => None,
        };
    }
    None
}

/// Whether an `air` type is a floating-point type, which decides the constant form the
/// bitcode emitter uses (`fconst` vs `iconst`).
pub fn is_float(ctx: &Context, h: TypeHandle) -> bool {
    matches!(
        from_pliron(ctx, h),
        Some(LlType::Half | LlType::BFloat | LlType::Float | LlType::Double)
    )
}
