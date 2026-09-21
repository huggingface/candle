//! Type interfaces of the `air` dialect.

use crate::bitcode::Ty;
use pliron::builtin::types::{BF16Type, FP16Type, FP32Type, FP64Type, IntegerType, UnitType};
use pliron::context::Context;
use pliron::derive::{type_interface, type_interface_impl};
use pliron::result::Result;
use pliron::r#type::{Type, TypeHandle, type_cast};

/// A [`Type`] that the bitcode writer can serialise.
#[type_interface]
pub trait ToBitcodeTy {
    /// This type in the bitcode type model.
    fn to_bitcode_ty(&self, ctx: &Context) -> Ty;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// The bitcode type of a handle
// TODO: Proper error handling over panic.
pub fn bitcode_ty(ctx: &Context, h: TypeHandle) -> Ty {
    let obj = h.deref(ctx);
    match type_cast::<dyn ToBitcodeTy>(&*obj) {
        Some(t) => t.to_bitcode_ty(ctx),
        None => panic!("no bitcode serialisation for type `{}`", obj.get_type_id()),
    }
}

#[type_interface_impl]
impl ToBitcodeTy for IntegerType {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::Int(self.width())
    }
}

#[type_interface_impl]
impl ToBitcodeTy for FP16Type {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::Half
    }
}

#[type_interface_impl]
impl ToBitcodeTy for FP32Type {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::Float
    }
}

#[type_interface_impl]
impl ToBitcodeTy for FP64Type {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::Double
    }
}

/// `UnitType` is the air spelling of `void`, including the result of a void
/// `llvm.call_intrinsic`.
#[type_interface_impl]
impl ToBitcodeTy for UnitType {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::Void
    }
}

#[type_interface_impl]
impl ToBitcodeTy for pliron_llvm::types::VectorType {
    fn to_bitcode_ty(&self, ctx: &Context) -> Ty {
        Ty::vec(self.num_elements(), bitcode_ty(ctx, self.elem_type()))
    }
}

#[type_interface_impl]
impl ToBitcodeTy for BF16Type {
    fn to_bitcode_ty(&self, _ctx: &Context) -> Ty {
        Ty::BFloat
    }
}

/// Typed pointer `air.ptr`, as opposed to the now standard opaque pointer `llvm.ptr`.
#[type_interface_impl]
impl ToBitcodeTy for super::types::AirPtrType {
    fn to_bitcode_ty(&self, ctx: &Context) -> Ty {
        Ty::ptr(bitcode_ty(ctx, self.pointee()), self.addrspace())
    }
}

/// Scalar type special values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Special {
    Zero,
    NegZero,
    Nan,
    PosInf,
    NegInf,
    One,
}

/// Scalar type definitions `air` needs.
#[type_interface]
pub trait AirScalar {
    /// Bit width.
    // TODO: const WIDTH: usize ?
    fn bits(&self) -> u32;
    /// Special value as raw bits, or `None` if this type can not represent the special.
    fn special_bits(&self, v: Special) -> Option<u64>;
    /// Float format rank. `None` for non-float.
    fn float_rank(&self) -> Option<u32>;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[type_interface_impl]
impl AirScalar for IntegerType {
    fn bits(&self) -> u32 {
        self.width()
    }
    fn special_bits(&self, v: Special) -> Option<u64> {
        (v == Special::Zero).then_some(0)
    }
    fn float_rank(&self) -> Option<u32> {
        None
    }
}

#[type_interface_impl]
impl AirScalar for FP16Type {
    fn bits(&self) -> u32 {
        16
    }
    fn special_bits(&self, v: Special) -> Option<u64> {
        match v {
            Special::Zero => 0,
            Special::NegZero => 0x8000,
            Special::Nan => 0x7E00,
            Special::PosInf => 0x7C00,
            Special::NegInf => 0xFC00,
            Special::One => 0x3C00,
        }
        .into()
    }
    fn float_rank(&self) -> Option<u32> {
        Some(1)
    }
}

#[type_interface_impl]
impl AirScalar for BF16Type {
    fn bits(&self) -> u32 {
        16
    }
    fn special_bits(&self, v: Special) -> Option<u64> {
        match v {
            Special::Zero => 0,
            Special::NegZero => 0x8000,
            Special::Nan => 0x7FC0,
            Special::PosInf => 0x7F80,
            Special::NegInf => 0xFF80,
            Special::One => 0x3F80,
        }
        .into()
    }
    fn float_rank(&self) -> Option<u32> {
        Some(1)
    }
}

#[type_interface_impl]
impl AirScalar for FP32Type {
    fn bits(&self) -> u32 {
        32
    }
    fn special_bits(&self, v: Special) -> Option<u64> {
        match v {
            Special::Zero => 0,
            Special::NegZero => 0x8000_0000,
            Special::Nan => 0x7FC0_0000,
            Special::PosInf => 0x7F80_0000,
            Special::NegInf => 0xFF80_0000,
            Special::One => f32::to_bits(1.0) as u64,
        }
        .into()
    }
    fn float_rank(&self) -> Option<u32> {
        Some(2)
    }
}

#[type_interface_impl]
impl AirScalar for FP64Type {
    fn bits(&self) -> u32 {
        64
    }
    fn special_bits(&self, v: Special) -> Option<u64> {
        match v {
            Special::Zero => 0,
            Special::NegZero => 0x8000_0000_0000_0000,
            Special::Nan => 0x7FF8_0000_0000_0000,
            Special::PosInf => 0x7FF0_0000_0000_0000,
            Special::NegInf => 0xFFF0_0000_0000_0000,
            Special::One => 0x3FF0_0000_0000_0000,
        }
        .into()
    }
    fn float_rank(&self) -> Option<u32> {
        Some(3)
    }
}

/// Bit width of a scalar air type.
pub fn scalar_bits(ctx: &Context, h: TypeHandle) -> Option<u32> {
    let obj = h.deref(ctx);
    type_cast::<dyn AirScalar>(&*obj).map(|t| t.bits())
}

/// Natural memory alignment of an air scalar in bytes.
pub fn align_bytes(ctx: &Context, h: TypeHandle) -> Option<u32> {
    scalar_bits(ctx, h).map(|b| (b / 8).max(1))
}

/// Special value as raw bits of provided air type.
pub fn special_bits(ctx: &Context, h: TypeHandle, v: Special) -> Option<u64> {
    let obj = h.deref(ctx);
    type_cast::<dyn AirScalar>(&*obj).and_then(|t| t.special_bits(v))
}

/// `Special::One` as raw bits of provided air type.
pub fn float_one_bits(ctx: &Context, h: TypeHandle) -> Option<u64> {
    special_bits(ctx, h, Special::One)
}

pub fn float_rank(ctx: &Context, h: TypeHandle) -> Option<u32> {
    let obj = h.deref(ctx);
    type_cast::<dyn AirScalar>(&*obj).and_then(|t| t.float_rank())
}

/// Returns the wider of the two float types, or `None` if either is not a float.
pub fn wider_float(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Option<TypeHandle> {
    let (ra, rb) = (float_rank(ctx, a)?, float_rank(ctx, b)?);
    Some(if ra >= rb { a } else { b })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air_dialect::types as t;

    #[test]
    fn all_types_have_expected_width() {
        let ctx = &mut Context::new();
        for (h, want) in [
            (t::i1_ty(ctx), 1),
            (t::i8_ty(ctx), 8),
            (t::i16_ty(ctx), 16),
            (t::i32_ty(ctx), 32),
            (t::i64_ty(ctx), 64),
            (t::f16_ty(ctx), 16),
            (t::bf16_ty(ctx), 16),
            (t::f32_ty(ctx), 32),
            (t::f64_ty(ctx), 64),
        ] {
            assert_eq!(scalar_bits(ctx, h), Some(want));
        }
        assert_eq!(align_bytes(ctx, t::i1_ty(ctx)), Some(1));
        assert_eq!(align_bytes(ctx, t::f64_ty(ctx)), Some(8));

        let ptr = t::device_ptr_ty(ctx, t::f32_ty(ctx));
        let vec = t::vec_ty(ctx, 4, t::f32_ty(ctx));
        for h in [ptr, vec, t::void_ty(ctx)] {
            assert_eq!(scalar_bits(ctx, h), None);
            assert_eq!(align_bytes(ctx, h), None);
        }
    }

    #[test]
    fn all_integers_support_zero_and_not_other_specials() {
        let ctx = &mut Context::new();
        for h in [
            t::i1_ty(ctx),
            t::i8_ty(ctx),
            t::i16_ty(ctx),
            t::i32_ty(ctx),
            t::i64_ty(ctx),
        ] {
            assert_eq!(special_bits(ctx, h, Special::Zero), Some(0));
            for v in [
                Special::NegZero,
                Special::Nan,
                Special::PosInf,
                Special::NegInf,
                Special::One,
            ] {
                assert_eq!(special_bits(ctx, h, v), None);
            }
        }
    }

    #[test]
    fn floats_support_all_specials() {
        let ctx = &mut Context::new();
        let cases = [
            (t::f16_ty(ctx), 0u64, 0x8000, 0x7E00, 0x7C00, 0xFC00, 0x3C00),
            (t::bf16_ty(ctx), 0, 0x8000, 0x7FC0, 0x7F80, 0xFF80, 0x3F80),
            (
                t::f32_ty(ctx),
                0,
                0x8000_0000,
                0x7FC0_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
            ),
            (
                t::f64_ty(ctx),
                0,
                0x8000_0000_0000_0000,
                0x7FF8_0000_0000_0000,
                0x7FF0_0000_0000_0000,
                0xFFF0_0000_0000_0000,
                0x3FF0_0000_0000_0000,
            ),
        ];
        for (h, zero, negzero, nan, posinf, neginf, one) in cases {
            let w = scalar_bits(ctx, h).unwrap();
            for (s, expected) in [
                (Special::Zero, zero),
                (Special::NegZero, negzero),
                (Special::Nan, nan),
                (Special::PosInf, posinf),
                (Special::NegInf, neginf),
                (Special::One, one),
            ] {
                let actual = special_bits(ctx, h, s).unwrap();
                assert_eq!(actual, expected);
                assert!(
                    w >= 64 || actual < (1u64 << w),
                    "{s:?} does not fit in {w} bits"
                );
            }
        }
    }

    #[test]
    fn floats_are_ranked_correctly() {
        let ctx = &mut Context::new();
        let (half, bfloat, float, double) = (
            t::f16_ty(ctx),
            t::bf16_ty(ctx),
            t::f32_ty(ctx),
            t::f64_ty(ctx),
        );
        assert_eq!(float_rank(ctx, half), float_rank(ctx, bfloat));
        assert!(float_rank(ctx, half) < float_rank(ctx, float));
        assert!(float_rank(ctx, float) < float_rank(ctx, double));
        assert_eq!(float_rank(ctx, t::i32_ty(ctx)), None);
        assert_eq!(wider_float(ctx, half, float), Some(float));
        assert_eq!(wider_float(ctx, double, float), Some(double));
        assert_eq!(wider_float(ctx, float, t::i32_ty(ctx)), None);
    }
}
