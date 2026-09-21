//! Ops of the `air` dialect.

use super::attrs::{BinKind, CastOpAttr};
use super::op_interfaces::{AirArrayLen, AirElemType};
use super::types::AirPtrType;
use pliron::builtin::attr_interfaces::TypedAttrInterface;
use pliron::builtin::attributes::{IntegerAttr, StringAttr};
use pliron::builtin::op_interfaces::{
    NResultsInterface, OneOpdInterface, OneResultInterface, OptionalOpdInterface,
};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::derive::pliron_op;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::verify_err;
use pliron_llvm::attributes::{FCmpPredicateAttr, ICmpPredicateAttr};
use pliron_llvm::op_interfaces::{AlignableOpInterface, CastOpInterface};
// Re-export ops that work as intended.
pub use pliron_llvm::ops::{
    BrOp, CallIntrinsicOp, CondBrOp, ConstantOp, ReturnOp, UndefOp, UnreachableOp, ZeroOp,
};

/// A conversion. Target type is the result type.
#[pliron_op(
    name = "air.cast",
    interfaces = [OneResultInterface, OneOpdInterface, CastOpInterface],
    operands = (src),
    attributes = (air_op: CastOpAttr),
    format
)]
pub struct CastOp;

#[derive(thiserror::Error, Debug)]
pub enum CastOpVerifyErr {
    #[error("Missing or incorrect operator attribute")]
    MissingOp,
    #[error("air.cast is only for pointer casts")]
    NotAPointerCast,
}

impl Verify for CastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some(kind) = self.get_attr_air_op(ctx).map(|a| *a) else {
            return verify_err!(self.loc(ctx), CastOpVerifyErr::MissingOp);
        };
        if !matches!(kind, CastOpAttr::IntToPtr | CastOpAttr::PtrToInt) {
            return verify_err!(self.loc(ctx), CastOpVerifyErr::NotAPointerCast);
        }
        Ok(())
    }
}

impl CastOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, src: Value, kind: CastOpAttr) -> Self {
        let op = CastOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![src],
                vec![],
                0,
            ),
        };
        op.set_attr_air_op(ctx, kind);
        op
    }
}

/// Load op. Requires alignment to be provided in bytes. 0 implies unaligned.
#[pliron_op(
    name = "air.load",
    interfaces = [OneResultInterface, OneOpdInterface, AlignableOpInterface],
    operands = (address: AirPtrType),
    format,
    verifier = "succ"
)]
pub struct LoadOp;

impl LoadOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, address: Value, align: u32) -> Self {
        let op = LoadOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![address],
                vec![],
                0,
            ),
        };
        op.set_alignment(ctx, align);
        op
    }
}

/// Store op. Requires alignment to be provided in bytes. 0 implies unaligned.
#[pliron_op(
    name = "air.store",
    interfaces = [NResultsInterface<0>, AlignableOpInterface],
    operands = (value, address: AirPtrType),
    format,
    verifier = "succ"
)]
pub struct StoreOp;

impl StoreOp {
    pub fn new(ctx: &mut Context, value: Value, address: Value, align: u32) -> Self {
        let op = StoreOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value, address],
                vec![],
                0,
            ),
        };
        op.set_alignment(ctx, align);
        op
    }
}

/// `getelementptr elem, ptr, idx`.
#[pliron_op(
    name = "air.gep",
    interfaces = [OneResultInterface, AirElemType],
    operands = (address: AirPtrType, idx),
    results = (_: AirPtrType),
    format,
    verifier = "succ"
)]
pub struct GepOp;

impl GepOp {
    pub fn new(
        ctx: &mut Context,
        res_ty: TypeHandle,
        address: Value,
        idx: Value,
        elem: TypeHandle,
    ) -> Self {
        let op = GepOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![address, idx],
                vec![],
                0,
            ),
        };
        op.set_elem_type(ctx, elem);
        op
    }
}

/// Alloca op.
/// `dyn_count` for runtime length, `count` for static array length, both `None` for scalar.
// TODO: Split `new` into `dynamic`, `static`, and `scalar`?
#[pliron_op(
    name = "air.alloca",
    interfaces = [
        OneResultInterface,
        OptionalOpdInterface,
        AirElemType,
        AirArrayLen,
        AlignableOpInterface,
    ],
    results = (_: AirPtrType),
    format,
    verifier = "succ"
)]
pub struct AllocaOp;

impl AllocaOp {
    pub fn new(
        ctx: &mut Context,
        res_ty: TypeHandle,
        dyn_count: Option<Value>,
        elem: TypeHandle,
        count: Option<u32>,
        align: u32,
    ) -> Self {
        let op = AllocaOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                dyn_count.into_iter().collect(),
                vec![],
                0,
            ),
        };
        op.set_elem_type(ctx, elem);
        if let Some(n) = count {
            let a = uint_attr(ctx, n as u64, 32);
            op.set_array_len(ctx, a);
        }
        op.set_alignment(ctx, align);
        op
    }
}

/// Reference to a global symbol, e.g. threadgroup scratch.
#[pliron_op(
    name = "air.global_ref",
    interfaces = [OneResultInterface],
    results = (_: AirPtrType),
    attributes = (air_sym: StringAttr),
    format,
    verifier = "succ"
)]
pub struct GlobalRefOp;

impl GlobalRefOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, sym: &str) -> Self {
        let op = GlobalRefOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        };
        op.set_attr_air_sym(ctx, StringAttr::new(sym.to_string()));
        op
    }
}

/// Indexing an array.
#[pliron_op(
    name = "air.gep_array",
    interfaces = [OneResultInterface, AirElemType, AirArrayLen],
    operands = (address: AirPtrType, idx),
    results = (_: AirPtrType),
    format,
    verifier = "succ"
)]
pub struct GepArrayOp;

impl GepArrayOp {
    pub fn new(
        ctx: &mut Context,
        res_ty: TypeHandle,
        address: Value,
        idx: Value,
        elem: TypeHandle,
        count: u32,
    ) -> Self {
        let op = GepArrayOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![address, idx],
                vec![],
                0,
            ),
        };
        op.set_elem_type(ctx, elem);
        let a = uint_attr(ctx, count as u64, 32);
        op.set_array_len(ctx, a);
        op
    }
}

/// `llvm.fneg`. Requires fastmath attribute.
pub fn fneg(ctx: &mut Context, res_ty: TypeHandle, src: pliron::value::Value) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let o = pliron_llvm::ops::FNegOp::new_with_fast_math_flags(
        ctx,
        src,
        pliron_llvm::attributes::FastmathFlagsAttr::default(),
    )
    .get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "fneg result type differs from the one the caller asked for"
    );
    o
}

/// `llvm.extractelement`.
pub fn extract_elem(
    ctx: &mut Context,
    res_ty: TypeHandle,
    vector: pliron::value::Value,
    index: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let o = pliron_llvm::ops::ExtractElementOp::new(ctx, vector, index).get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "extractelement result type differs from the one the caller asked for"
    );
    o
}

/// `llvm.insertelement`. Result is the vector's type.
pub fn insert_elem(
    ctx: &mut Context,
    res_ty: TypeHandle,
    vector: pliron::value::Value,
    element: pliron::value::Value,
    index: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let o = pliron_llvm::ops::InsertElementOp::new(ctx, vector, element, index).get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "insertelement result type differs from the one the caller asked for"
    );
    o
}

/// Build binary op.
pub fn bin(
    ctx: &mut Context,
    kind: BinKind,
    res_ty: TypeHandle,
    lhs: pliron::value::Value,
    rhs: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    use pliron_llvm::op_interfaces::{BinArithOp, IntBinArithOpWithOverflowFlag};
    use pliron_llvm::ops as llvm;
    let nof = pliron_llvm::attributes::IntegerOverflowFlagsAttr::default();
    let o = match kind {
        BinKind::Add => llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, nof).get_operation(),
        BinKind::Sub => llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, nof).get_operation(),
        BinKind::Mul => llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, nof).get_operation(),
        BinKind::Shl => llvm::ShlOp::new_with_overflow_flag(ctx, lhs, rhs, nof).get_operation(),
        BinKind::UDiv => llvm::UDivOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::SDiv => llvm::SDivOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::URem => llvm::URemOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::SRem => llvm::SRemOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::And => llvm::AndOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::Or => llvm::OrOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::Xor => llvm::XorOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::LShr => llvm::LShrOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::AShr => llvm::AShrOp::new(ctx, lhs, rhs).get_operation(),
        BinKind::FAdd => {
            let o = llvm::FAddOp::new(ctx, lhs, rhs);
            set_empty_fastmath(ctx, &o);
            o.get_operation()
        }
        BinKind::FSub => {
            let o = llvm::FSubOp::new(ctx, lhs, rhs);
            set_empty_fastmath(ctx, &o);
            o.get_operation()
        }
        BinKind::FMul => {
            let o = llvm::FMulOp::new(ctx, lhs, rhs);
            set_empty_fastmath(ctx, &o);
            o.get_operation()
        }
        BinKind::FDiv => {
            let o = llvm::FDivOp::new(ctx, lhs, rhs);
            set_empty_fastmath(ctx, &o);
            o.get_operation()
        }
        BinKind::FRem => {
            let o = llvm::FRemOp::new(ctx, lhs, rhs);
            set_empty_fastmath(ctx, &o);
            o.get_operation()
        }
    };
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "result type differs from the one provided by the caller"
    );
    o
}

/// `llvm.call_intrinsic`. `@air.*` / `@llvm.*` / `@__tensorops_*`.
///
/// `pliron-llvm` models `call_intrinsic` as `OneResultInterface`. We return `UnitType` for calls
/// that return no result.
pub fn call(
    ctx: &mut Context,
    callee: &str,
    res_ty: Option<TypeHandle>,
    args: Vec<pliron::value::Value>,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let arg_tys: Vec<TypeHandle> = args.iter().map(|a| a.get_type(ctx)).collect();
    let res = res_ty.unwrap_or_else(|| pliron::builtin::types::UnitType::get(ctx).into());
    let fty = pliron_llvm::types::FuncType::get(ctx, res, arg_tys, false);
    let c = pliron_llvm::ops::CallIntrinsicOp::new(
        ctx,
        pliron::builtin::attributes::StringAttr::new(callee.to_string()),
        fty,
        args,
    );
    // Note: separate concept from the `FastMathFlags` interface.
    c.set_attr_llvm_intrinsic_fastmath_flags(
        ctx,
        pliron_llvm::attributes::FastmathFlagsAttr::default(),
    );
    c.get_operation()
}

/// `llvm.icmp`. Constructor derives scalar or vector `i1` result type.
pub fn icmp(
    ctx: &mut Context,
    pred: ICmpPredicateAttr,
    res_ty: TypeHandle,
    lhs: pliron::value::Value,
    rhs: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let o = pliron_llvm::ops::ICmpOp::new(ctx, pred, lhs, rhs).get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "result type differs from the one provided by the caller"
    );
    o
}

/// `llvm.fcmp`. Result type derived like [`icmp`].
pub fn fcmp(
    ctx: &mut Context,
    pred: FCmpPredicateAttr,
    res_ty: TypeHandle,
    lhs: pliron::value::Value,
    rhs: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    use pliron_llvm::op_interfaces::FastMathFlags;
    let c = pliron_llvm::ops::FCmpOp::new(ctx, pred, lhs, rhs);
    c.set_fast_math_flags(ctx, pliron_llvm::attributes::FastmathFlagsAttr::default());
    let o = c.get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "result type differs from the one provided by the caller"
    );
    o
}

/// `llvm.select`.
pub fn select(
    ctx: &mut Context,
    res_ty: TypeHandle,
    cond: pliron::value::Value,
    t: pliron::value::Value,
    f: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron::r#type::Typed;
    let o = pliron_llvm::ops::SelectOp::new_with_fast_math_flags(
        ctx,
        cond,
        t,
        f,
        pliron_llvm::attributes::FastmathFlagsAttr::default(),
    )
    .get_operation();
    debug_assert_eq!(
        o.deref(ctx).get_result(0).get_type(ctx),
        res_ty,
        "result type differs from the one provided by the caller"
    );
    o
}

/// Build LLVM cast op from `kind`.
///
/// Pointer casts use `air.cast` beause pliron-llvm uses opaque pointers, while AIR requires
/// typed pointers which we provide via `AirPtrType`. See `crate::bitcode`.
pub fn cast(
    ctx: &mut Context,
    kind: CastOpAttr,
    res_ty: TypeHandle,
    src: pliron::value::Value,
) -> Ptr<Operation> {
    use pliron_llvm::op_interfaces::CastOpInterface;
    use pliron_llvm::ops as llvm;
    match kind {
        CastOpAttr::Trunc => llvm::TruncOp::new(ctx, src, res_ty).get_operation(),
        // `llvm.zext` requires an `nneg` flag, while the plain constructor leaves it unset.
        // Same issue as fast math flag for float ops.
        CastOpAttr::ZExt => {
            use pliron_llvm::op_interfaces::NNegFlag;
            let c = llvm::ZExtOp::new(ctx, src, res_ty);
            c.set_nneg(ctx, false);
            c.get_operation()
        }
        CastOpAttr::SExt => llvm::SExtOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::FPTrunc => {
            let c = llvm::FPTruncOp::new(ctx, src, res_ty);
            set_empty_fastmath(ctx, &c);
            c.get_operation()
        }
        CastOpAttr::FPExt => {
            let c = llvm::FPExtOp::new(ctx, src, res_ty);
            set_empty_fastmath(ctx, &c);
            c.get_operation()
        }
        CastOpAttr::FPToSI => llvm::FPToSIOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::FPToUI => llvm::FPToUIOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::SIToFP => llvm::SIToFPOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::UIToFP => llvm::UIToFPOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::Bitcast => llvm::BitcastOp::new(ctx, src, res_ty).get_operation(),
        CastOpAttr::IntToPtr | CastOpAttr::PtrToInt => {
            CastOp::new(ctx, res_ty, src, kind).get_operation()
        }
    }
}

/// Constant attribute of scalar `ty` from raw bits.
fn const_attr(ctx: &mut Context, ty: TypeHandle, bits: u64) -> Box<dyn TypedAttrInterface> {
    use pliron::builtin::attributes::{FPBFloatAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr};
    use pliron::builtin::types::{BF16Type, FP16Type, FP32Type, FP64Type};
    use pliron::r#type::TypedHandle;
    use pliron::utils::apfloat::{BFloat, Double, Float, Half, Single};

    if let Ok(i) = TypedHandle::<IntegerType>::from_handle(ty, ctx) {
        let w = i.deref(ctx).width() as usize;
        // The combination of how `APInt::from_u64` truncates and how negative constants arrive
        // via `v as i64 as u64` means high bits are all zero or all one. Any other value does
        // not belong at this width.
        // TODO: Triple check the negative constants claim.
        assert!(
            w >= 64 || (bits >> w) == 0 || ((bits as i64) >> w) == -1,
            "constant {bits:#x} does not fit in i{w}"
        );
        return Box::new(IntegerAttr::new(
            i,
            APInt::from_u64(bits, core::num::NonZero::new(w).unwrap()),
        ));
    }
    // Float constants are stored as raw bits. This lets us verify that the bits fit the type.
    let fits = |w: u32| {
        assert!(
            (bits >> w) == 0,
            "float constant {bits:#x} is wider than its {w}-bit type"
        );
    };
    if TypedHandle::<FP16Type>::from_handle(ty, ctx).is_ok() {
        fits(16);
        return Box::new(FPHalfAttr(Half::from_bits(bits as u128)));
    }
    if TypedHandle::<FP32Type>::from_handle(ty, ctx).is_ok() {
        fits(32);
        return Box::new(FPSingleAttr(Single::from_bits(bits as u128)));
    }
    if TypedHandle::<FP64Type>::from_handle(ty, ctx).is_ok() {
        return Box::new(FPDoubleAttr(Double::from_bits(bits as u128)));
    }
    if TypedHandle::<BF16Type>::from_handle(ty, ctx).is_ok() {
        fits(16);
        return Box::new(FPBFloatAttr(BFloat::from_bits(bits as u128)));
    }
    // TODO: Return proper Error instead
    panic!("no constant attribute for this air type");
}

/// A scalar constant as raw bits.
pub fn const_bits(ctx: &mut Context, res_ty: TypeHandle, bits: u64) -> ConstantOp {
    let value = const_attr(ctx, res_ty, bits);
    ConstantOp::new(ctx, value)
}

/// A dense vector constant, elements as raw bits. `ImmVal::VecInt` / `VecFloatBits`.
///
/// `llvm.constant` with `AggregateAttr`. How pliron-llvm represents vector constants.
pub fn const_vec(ctx: &mut Context, res_ty: TypeHandle, bits: Vec<u64>) -> ConstantOp {
    use pliron::r#type::TypedHandle;
    // TODO: Return proper Error instead
    let elem_ty = TypedHandle::<pliron_llvm::types::VectorType>::from_handle(res_ty, ctx)
        .expect("a vector constant has a vector type")
        .deref(ctx)
        .elem_type();
    let elems: Vec<Box<dyn TypedAttrInterface>> = bits
        .into_iter()
        .map(|b| const_attr(ctx, elem_ty, b))
        .collect();
    ConstantOp::new(
        ctx,
        Box::new(pliron_llvm::attributes::AggregateAttr::new(elems, res_ty)),
    )
}

/// An `IntegerAttr` of `width` bits.
fn uint_attr(ctx: &mut Context, v: u64, width: u32) -> IntegerAttr {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    IntegerAttr::new(
        ty,
        APInt::from_u64(v, core::num::NonZero::new(width as usize).unwrap()),
    )
}

/// Float ops require fastmath attribute to be set.
fn set_empty_fastmath<T: pliron_llvm::op_interfaces::FastMathFlags>(ctx: &Context, op: &T) {
    op.set_fast_math_flags(ctx, pliron_llvm::attributes::FastmathFlagsAttr::default());
}

/// Mark a call as `call fast`, which is how the `air.fast_*` intrinsics are reached.
///
/// `llvm.call_intrinsic` carries its own fastmath attribute, so the flag rides there rather
/// than in a side-channel of ours.
pub fn set_fast(op: Ptr<Operation>, ctx: &mut Context) {
    let c = pliron_llvm::ops::CallIntrinsicOp::wrap_operation(op);
    // TODO: Return proper Error instead
    let c = c
        .downcast::<pliron_llvm::ops::CallIntrinsicOp>()
        .expect("set_fast on a non-call");
    c.set_attr_llvm_intrinsic_fastmath_flags(
        ctx,
        pliron_llvm::attributes::FastmathFlags::FAST.into(),
    );
}

/// LLVM metadata symbols for unrolling.
pub const MD_LLVM_LOOP: &str = "llvm.loop";
pub const MD_LLVM_LOOP_UNROLL_ENABLE: &str = "llvm.loop.unroll.enable";
pub const MD_LLVM_LOOP_UNROLL_COUNT: &str = "llvm.loop.unroll.count";

/// Set unroll count on a loop's terminator latch.
/// Compile-time hint, not necessarily a fact.
///
/// Resulting LLVM looks similar to this:
/// ```llvm
/// br label %body, !llvm.loop !0
/// !0 = distinct !{!0, !1, !2}
/// !1 = !{!"llvm.loop.unroll.enable"}
/// !2 = !{!"llvm.loop.unroll.count", i32 8}
/// ```
pub fn set_unroll(op: Ptr<Operation>, ctx: &mut Context, count: u32) {
    use pliron_llvm::metadata::{
        MdNodeAttr, MdOperandAttr, attach_metadata, find_enclosing_module, get_metadata_table,
        set_metadata_table,
    };
    let count_attr = uint_attr(ctx, count as u64, 32);
    // TODO: Return proper Error instead
    let module_op = find_enclosing_module(ctx, op).expect("a loop latch is inside a module");
    let mut table = get_metadata_table(ctx, module_op).unwrap_or_default();
    let enable = table.push_uniqued(MdNodeAttr::new_tuple(vec![MdOperandAttr::String(
        MD_LLVM_LOOP_UNROLL_ENABLE.to_string(),
    )]));
    let count_node = table.push_uniqued(MdNodeAttr::new_tuple(vec![
        MdOperandAttr::String(MD_LLVM_LOOP_UNROLL_COUNT.to_string()),
        MdOperandAttr::Constant(Box::new(count_attr)),
    ]));
    let loop_id = table.reserve();
    table.set(
        loop_id,
        MdNodeAttr::new_distinct_tuple(vec![
            MdOperandAttr::Node(loop_id),
            MdOperandAttr::Node(enable),
            MdOperandAttr::Node(count_node),
        ]),
    );
    set_metadata_table(ctx, module_op, table);
    attach_metadata(ctx, op, MD_LLVM_LOOP, loop_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air_dialect::types as t;

    #[test]
    fn negative_integer_constant_is_valid_via_sign_extension() {
        let ctx = &mut Context::new();
        for w in [
            t::i8_ty(ctx),
            t::i16_ty(ctx),
            t::i32_ty(ctx),
            t::i64_ty(ctx),
        ] {
            const_attr(ctx, w, -1i64 as u64);
            const_attr(ctx, w, 0);
            const_attr(ctx, w, 1);
        }
    }

    #[test]
    #[should_panic(expected = "does not fit in i16")]
    fn too_wide_integer_constant_is_rejected() {
        let ctx = &mut Context::new();
        let i16_ty = t::i16_ty(ctx);
        const_attr(ctx, i16_ty, 0x1_2345);
    }

    #[test]
    #[should_panic(expected = "wider than its 16-bit type")]
    fn f16_rejects_f32_bits() {
        let ctx = &mut Context::new();
        let f16 = t::f16_ty(ctx);
        const_attr(ctx, f16, f32::to_bits(1.0) as u64);
    }

    #[test]
    fn float_const_attrs_accept_valid_bits() {
        let ctx = &mut Context::new();
        for (h, one) in [
            (t::f16_ty(ctx), 0x3C00u64),
            (t::bf16_ty(ctx), 0x3F80),
            (t::f32_ty(ctx), f32::to_bits(1.0) as u64),
            (t::f64_ty(ctx), f64::to_bits(1.0)),
        ] {
            const_attr(ctx, h, one);
        }
    }
}
