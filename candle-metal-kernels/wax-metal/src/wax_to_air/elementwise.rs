//! Elementwise. Arithmetic, atomics, casts, comparisons, float intrinsics, and tensor shape reads.

use super::*;
use crate::air_dialect::ops as air;
use crate::air_dialect::type_interfaces::{align_bytes, float_one_bits, scalar_bits, wider_float};
use crate::air_dialect::types as t;
use crate::lower::{ImmVal, dense_elements_value};
use core::num::NonZeroUsize;
use pliron::r#type::Typed;
use pliron::value::Value;
use wax_ir::dialect::ops::{BITS_1, BITS_32};

/// The partition's element type as a wax scalar.
pub(crate) fn elem_scalar_of(view_ty: &WaxType) -> Option<ScalarType> {
    match view_ty {
        WaxType::PartitionView(pv) => Some(pv.tensor_view.element_type.clone()),
        _ => None,
    }
}

/// An op's operand segments, as `(base, dyn_shape, dyn_strides)`.
///
/// Reads pliron's `OperandSegmentSizesAttr` as opposed to the source dialect's MLIR
/// `operandSegmentSizes`, which is translated on import.
pub(crate) fn wax_operand_segments(ctx: &Context, op: Ptr<Operation>) -> (usize, usize, usize) {
    use pliron::builtin::attributes::OperandSegmentSizesAttr;
    use pliron::builtin::op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES;

    let op_ref = op.deref(ctx);
    match op_ref
        .attributes
        .get::<OperandSegmentSizesAttr>(&ATTR_KEY_OPERAND_SEGMENT_SIZES)
    {
        Some(s) => {
            let get = |i: usize| s.0.get(i).copied().unwrap_or(0) as usize;
            (get(0), get(1), get(2))
        }
        // An op built without segments is one base operand and no dynamic extents.
        None => (1, 0, 0),
    }
}

impl WaxToAir {
    /// A wax constant becomes `air.const_bits` carrying the same bit pattern.
    pub(crate) fn rewrite_constant(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        // A fragment constant is a grid of `<64 x float>`, not a scalar splat. Only the
        // zero splat is supported (accumulator init).
        let res0 = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0))
        };
        if let Some(v) = res0
            && self.frags.fragments.contains(&v)
            && let Some(wt) = wax_type_of(ctx, v.get_type(ctx))
            && let Some((nr, nc)) = crate::types::sgmma_dims(&wt)
        {
            return self.rewrite_fragment_constant(ctx, rewriter, op, nr, nc);
        }
        let (res_ty, bits) = {
            let o = op.deref(ctx);
            if o.get_num_results() != 1 {
                drop(o);
                return self.refuse(Opcode::Constant, "not a single-result op");
            }
            let ty = o.get_result(0).get_type(ctx);
            // Value in IR as a dense elements attribute.
            let bits = (|| {
                let a = o.attributes.get::<wax_ir::WaxAttr>(&keys::ATTR_VALUE)?;
                let wax_ir::Attribute::DenseElements(de) = &a.0 else {
                    return None;
                };
                let wt = wax_type_of(ctx, ty)?;
                Some(match dense_elements_value(de, &wt) {
                    ImmVal::Int(i) => i as u64,
                    ImmVal::FloatBits(b) => b,
                    ImmVal::Zero => 0,
                    _ => return None,
                })
            })();
            (ty, bits)
        };
        let Some(bits) = bits else {
            return self.refuse(Opcode::Constant, "no decodable value");
        };
        let Some(res_ty) = air_ty_for(ctx, res_ty) else {
            return self.refuse(Opcode::Constant, "result type has no air type");
        };
        let Some(c) = air::const_bits(ctx, res_ty, bits) else {
            return self.refuse(Opcode::Constant, "result type unrepresentable as constant");
        };
        rewriter.insert_operation(ctx, c.get_operation());
        rewriter.replace_operation(ctx, op, c.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// Float negation. `air.unop` with `fneg` code.
    pub(crate) fn rewrite_neg_f(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1 && o.get_num_operands() == 1)
                .then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            return self.refuse(Opcode::NegF, "result type unreadable");
        };
        // NegF is a float operation. Integer negation lowers as subtraction.
        let is_float = wax_type_of(ctx, wax_res)
            .and_then(|w| crate::types::scalar_of(&w))
            .and_then(|s| crate::types::float_elem(ctx, &s))
            .is_some();
        if !is_float {
            return self.refuse(Opcode::NegF, "operand is not a float");
        }
        let Some(res_ty) = air_ty_for(ctx, wax_res) else {
            return self.refuse(Opcode::NegF, "result type has no air type");
        };
        let src = op.deref(ctx).operands().next();
        let Some(src) = src else {
            return self.refuse(Opcode::NegF, "no source operand");
        };
        let u = air::fneg(ctx, res_ty, src);
        rewriter.insert_operation(ctx, u);
        rewriter.replace_operation(ctx, op, u);
        self.converted += 1;
        Ok(())
    }
    /// `atan2(y, x)`. `call fast @air.fast_atan2.{suf}`.
    /// Widen, compute, narrow if needed.
    pub(crate) fn rewrite_atan2(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1 && o.get_num_operands() == 2)
                .then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            return self.refuse(Opcode::Atan2, "result type unreadable");
        };
        let Some((cty, suf, rty)) = fp_intr_info(ctx, wax_res) else {
            return self.refuse(Opcode::Atan2, "no float intrinsic at this result type");
        };
        let ops: Vec<_> = op.deref(ctx).operands().collect();
        // y first.
        let Some(y) = self.fpcast_if(ctx, rewriter, ops[0], rty, cty) else {
            return Ok(());
        };
        let Some(x) = self.fpcast_if(ctx, rewriter, ops[1], rty, cty) else {
            return Ok(());
        };
        let cty_h = cty;
        let c = air::call_fast(
            ctx,
            &format!("@air.fast_atan2.{suf}"),
            Some(cty_h),
            vec![y, x],
        );
        let r = emit(ctx, rewriter, c);
        let Some(r) = self.fpcast_if(ctx, rewriter, r, cty, rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// The tensor's static shape, one i32 per result.
    pub(crate) fn rewrite_get_tensor_shape(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let (src, n_res) = {
            let o = op.deref(ctx);
            (o.operands().next(), o.get_num_results())
        };
        let Some(src) = src else {
            return self.refuse(Opcode::GetTensorShape, "no source operand");
        };
        // Shape extracted from pre-conversion wax TensorView type.
        let src_wax = wax_operand_ty(ctx, operands_info, src);
        let Some(WaxType::TensorView(tv)) = src_wax else {
            return self.refuse(Opcode::GetTensorShape, "operand is not a tensor view");
        };
        if tv.shape.contains(&wax_ir::DYNAMIC) || tv.shape.len() != n_res {
            self.bail(
                Opcode::GetTensorShape,
                "tensor shape is dynamic or disagrees with the result count",
            );
            return Ok(());
        }
        let mut vals = Vec::with_capacity(n_res);
        for &d in &tv.shape {
            let c = air::const_int(ctx, BITS_32, d as u64);
            vals.push(emit_op(ctx, rewriter, c));
        }
        rewriter.replace_operation_with_values(ctx, op, vals);
        self.converted += 1;
        Ok(())
    }
    /// Atomic read-modify-write.
    /// `old = @air.atomic.{space}.{op}.{ty}(ptr, val, i32 0, i32 scope, i1 true)`.
    /// Token is currently ignored via `undef i32`.
    pub(crate) fn rewrite_atomic_rmw(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let code = Opcode::AtomicRMW;
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        let (Some(&ptr), Some(&val)) = (operands.first(), operands.get(1)) else {
            return self.refuse(code, "atomic rmw needs a pointer and a value");
        };
        let Some(elem_ll) =
            wax_operand_ty(ctx, operands_info, ptr).and_then(|ty| crate::lower::ptr_elem_ll(&ty))
        else {
            return self.refuse(code, "pointer operand type unreadable");
        };
        let Some(mode) = wax_int_attr(ctx, op, &keys::ATTR_MODE) else {
            return self.refuse(code, "atomic rmw has no mode attribute");
        };
        let Some(scope) =
            wax_int_attr(ctx, op, &keys::ATTR_MEMORY_SCOPE).and_then(crate::types::atomic_scope)
        else {
            return self.refuse(code, "atomic rmw has no usable memory scope");
        };
        let Some((intrinsic, air_ty)) = crate::types::atomic_rmw_intrinsic(mode, elem_ll) else {
            return self.refuse(code, "no AIR atomic for this operation and element type");
        };
        let Some(elem_h) = crate::types::atomic_ty(ctx, air_ty) else {
            return self.refuse(code, "no AIR atomic at this element width");
        };
        let i32_h = t::i32_ty(ctx);
        let mut const_int = |w: NonZeroUsize, bits: u64| -> Value { air_const_int(ctx, w, bits) };
        let c0 = const_int(BITS_32, 0);
        let cscope = const_int(BITS_32, scope as u64);
        let ctrue = const_int(BITS_1, 1);
        let call = air::call(
            ctx,
            &intrinsic,
            Some(elem_h),
            vec![ptr, val, c0, cscope, ctrue],
        );
        // Atomic RMW returns the contents of the location before the modification, hence "old".
        let old = emit(ctx, rewriter, call);
        let token = {
            let u = air::UndefOp::new(ctx, i32_h);
            emit_op(ctx, rewriter, u)
        };
        rewriter.replace_operation_with_values(ctx, op, vec![old, token]);
        self.converted += 1;
        Ok(())
    }
    /// Atomic compare-and-swap. `@air.atomic.{space}.cmpxchg.weak.{ty}`.
    pub(crate) fn rewrite_atomic_cas(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let code = Opcode::AtomicCAS;
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        let (Some(&ptr), Some(&cmp), Some(&new_val)) =
            (operands.first(), operands.get(1), operands.get(2))
        else {
            return self.refuse(code, "atomic cas needs pointer, expected and new value");
        };
        let Some(elem_ll) =
            wax_operand_ty(ctx, operands_info, ptr).and_then(|ty| crate::lower::ptr_elem_ll(&ty))
        else {
            return self.refuse(code, "pointer operand type unreadable");
        };
        let Some(scope) =
            wax_int_attr(ctx, op, &keys::ATTR_MEMORY_SCOPE).and_then(crate::types::atomic_scope)
        else {
            return self.refuse(code, "atomic cas has no usable memory scope");
        };
        let Some(air_ty) = crate::types::atomic_air_ty(elem_ll) else {
            return self.refuse(code, "no AIR atomic at this element width");
        };
        let Some(elem) = crate::types::atomic_ty(ctx, air_ty) else {
            return self.refuse(code, "no AIR atomic at this element width");
        };
        let i32_t = t::i32_ty(ctx);
        // The expected value's stack slot. One element, align 4, no count operand. The
        // bitcode form puts the length in the type and passes a literal 1.
        let exp = {
            let slot_ty = t::ptr_ty(ctx, elem, 0);
            let a = air::AllocaOp::new(ctx, slot_ty, None, elem, None, 4).get_operation();
            emit(ctx, rewriter, a)
        };
        let st = air::StoreOp::new(ctx, cmp, exp, 4);
        rewriter.insert_operation(ctx, st.get_operation());
        let mut const_int = |w: NonZeroUsize, bits: u64| -> Value { air_const_int(ctx, w, bits) };
        let i32_0 = const_int(BITS_32, 0);
        let cscope = const_int(BITS_32, scope as u64);
        let ctrue = const_int(BITS_1, 1);

        // `wax_ir::PointerType` is device memory by construction.
        // No threadgroup atomic to name until wax grows an address space.

        // TODO: Should not be a magic string. Perhaps we need proper structs with `to_air`.
        let intrinsic = format!("@air.atomic.global.cmpxchg.weak.{air_ty}");
        let call = air::call(
            ctx,
            &intrinsic,
            Some(elem),
            vec![ptr, exp, new_val, i32_0, i32_0, cscope, ctrue],
        );
        let old = emit(ctx, rewriter, call);
        let token = {
            let u = air::UndefOp::new(ctx, i32_t);
            emit_op(ctx, rewriter, u)
        };
        rewriter.replace_operation_with_values(ctx, op, vec![old, token]);
        self.converted += 1;
        Ok(())
    }
    /// A wax binary op becomes `air.bin` with the operator in an attribute.
    pub(crate) fn rewrite_binary(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let attr = match code {
            Opcode::AddF => BinKind::FAdd,
            Opcode::SubF => BinKind::FSub,
            Opcode::MulF => BinKind::FMul,
            Opcode::DivF => BinKind::FDiv,
            Opcode::AddI => BinKind::Add,
            Opcode::SubI => BinKind::Sub,
            Opcode::MulI => BinKind::Mul,
            // Signedness picks the instruction. Absent sign attribute means signed.
            // Sign-filling preserves the value where zero-filling would not.
            Opcode::DivI | Opcode::RemI | Opcode::ShRI => {
                let signed = wax_int_attr(ctx, op, &keys::ATTR_SIGNEDNESS) != Some(0);
                match (code, signed) {
                    (Opcode::DivI, true) => BinKind::SDiv,
                    (Opcode::DivI, false) => BinKind::UDiv,
                    (Opcode::RemI, true) => BinKind::SRem,
                    (Opcode::RemI, false) => BinKind::URem,
                    (Opcode::ShRI, true) => BinKind::AShr,
                    (Opcode::ShRI, false) => BinKind::LShr,
                    _ => {
                        return self.refuse(code, "signed/unsigned integer op with no mapping");
                    }
                }
            }
            Opcode::ShLI => BinKind::Shl,
            Opcode::AndI => BinKind::And,
            Opcode::OrI => BinKind::Or,
            Opcode::XOrI => BinKind::Xor,
            _ => {
                return self.refuse(code, "opcode has no binary kind");
            }
        };
        let parts = {
            let o = op.deref(ctx);
            let mut it = o.operands();
            let two = it.next().zip(it.next());
            let one_result = o.get_num_results() == 1;
            let ty = one_result.then(|| o.get_result(0).get_type(ctx));
            two.zip(ty)
        };
        let Some(((lhs, rhs), res_ty)) = parts else {
            return self.refuse(code, "operand count or result type unreadable");
        };
        let Some(out_ty) = air_ty_for(ctx, res_ty) else {
            return self.refuse(code, "operand type has no air type");
        };
        // `DivI` + `positive_inf` (3) rounding mode is rewritten to `ceil_div`.
        if code == Opcode::DivI && wax_int_attr(ctx, op, &keys::ATTR_ROUNDING) == Some(3) {
            return self.rewrite_ceil_div(ctx, rewriter, op, lhs, rhs, out_ty);
        }
        // Float binops with mixed precision support (widen to f32, compute, and narrow back).
        let (lhs, rhs, bin_ty, narrow) = if matches!(
            code,
            Opcode::AddF | Opcode::SubF | Opcode::MulF | Opcode::DivF
        ) {
            let (ta, tb) = (lhs.get_type(ctx), rhs.get_type(ctx));
            let cty = if ta == tb {
                ta
            } else {
                match wider_float(ctx, ta, tb) {
                    Some(c) => c,
                    None => {
                        return self.refuse(code, "operands have no common float type");
                    }
                }
            };
            let Some(lhs) = self.fpcast_if(ctx, rewriter, lhs, ta, cty) else {
                return Ok(());
            };
            let Some(rhs) = self.fpcast_if(ctx, rewriter, rhs, tb, cty) else {
                return Ok(());
            };
            let narrow = (out_ty != cty).then_some((cty, out_ty));
            (lhs, rhs, cty, narrow)
        } else {
            (lhs, rhs, out_ty, None)
        };
        let b = air::bin(ctx, attr, bin_ty, lhs, rhs);
        rewriter.insert_operation(ctx, b);
        match narrow {
            None => rewriter.replace_operation(ctx, op, b),
            Some((from, to)) => {
                let out = b.deref(ctx).get_result(0);
                let Some(final_v) = self.fpcast_if(ctx, rewriter, out, from, to) else {
                    return Ok(());
                };
                rewriter.replace_operation_with_values(ctx, op, vec![final_v]);
            }
        }
        self.converted += 1;
        Ok(())
    }
    /// `ceil_div(a, b)` with correct rounding.
    fn rewrite_ceil_div(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        a: Value,
        b: Value,
        ty: TypeHandle,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let signed = wax_int_attr(ctx, op, &keys::ATTR_SIGNEDNESS) != Some(0);
        let i1_h = t::i1_ty(ctx);
        let bin = |ctx: &mut Context,
                   rw: &mut DialectConversionRewriter,
                   t: TypeHandle,
                   x,
                   y,
                   k: BinKind| {
            let o = air::bin(ctx, k, t, x, y);
            emit(ctx, rw, o)
        };
        let cmp =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, x, y, p: ICmpPredicateAttr| {
                let o = air::icmp(ctx, p, i1_h, x, y);
                emit(ctx, rw, o)
            };
        let zero = {
            let z = air::ZeroOp::new(ctx, ty);
            emit_op(ctx, rewriter, z)
        };
        let (dv, rm) = if signed {
            (BinKind::SDiv, BinKind::SRem)
        } else {
            (BinKind::UDiv, BinKind::URem)
        };
        let q = bin(ctx, rewriter, ty, a, b, dv);
        let r = bin(ctx, rewriter, ty, a, b, rm);
        let r_nz = cmp(ctx, rewriter, r, zero, ICmpPredicateAttr::NE);
        let adjust = if signed {
            let pa = cmp(ctx, rewriter, a, zero, ICmpPredicateAttr::SGT);
            let pb = cmp(ctx, rewriter, b, zero, ICmpPredicateAttr::SGT);
            // `same sign` is the negated xor of the two predicates.
            let differ = bin(ctx, rewriter, i1_h, pa, pb, BinKind::Xor);
            let one = air_const_int(ctx, BITS_1, 1);
            let same = bin(ctx, rewriter, i1_h, differ, one, BinKind::Xor);
            bin(ctx, rewriter, i1_h, r_nz, same, BinKind::And)
        } else {
            r_nz
        };
        let Some(one) = air_const(ctx, ty, 1) else {
            return self.refuse(Opcode::DivI, "no 1 constant at the result type");
        };
        let bumped = bin(ctx, rewriter, ty, q, one, BinKind::Add);
        let sel = air::select(ctx, ty, adjust, bumped, q);
        let out = emit(ctx, rewriter, sel);
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// `get_tile_block_id` to threadgroup position. One result per axis.
    pub(crate) fn rewrite_tile_block_id(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some(tg) = self.control.tg else {
            return self.refuse(Opcode::GetTileBlockId, "no threadgroup position parameter");
        };
        let i32_ty = t::i32_ty(ctx);
        let n = op.deref(ctx).get_num_results();
        let mut replacements = Vec::with_capacity(n);
        for axis in 0..n {
            let idx = air::const_int(ctx, BITS_32, axis as u64);
            let idx_v = emit_op(ctx, rewriter, idx);
            let e = air::extract_elem(ctx, i32_ty, tg, idx_v);
            replacements.push(emit(ctx, rewriter, e));
        }
        rewriter.replace_operation_with_values(ctx, op, replacements);
        self.converted += 1;
        Ok(())
    }
    /// Integer comparison.
    pub(crate) fn rewrite_cmp_int(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let signed = wax_int_attr(ctx, op, &keys::ATTR_SIGNEDNESS) != Some(0);
        let pred = wax_int_attr(ctx, op, &keys::ATTR_COMPARISON_PREDICATE).unwrap_or(0);
        let attr = match (pred, signed) {
            (0, _) => ICmpPredicateAttr::EQ,
            (1, _) => ICmpPredicateAttr::NE,
            (2, true) => ICmpPredicateAttr::SLT,
            (2, false) => ICmpPredicateAttr::ULT,
            (3, true) => ICmpPredicateAttr::SLE,
            (3, false) => ICmpPredicateAttr::ULE,
            (4, true) => ICmpPredicateAttr::SGT,
            (4, false) => ICmpPredicateAttr::UGT,
            (5, true) => ICmpPredicateAttr::SGE,
            (5, false) => ICmpPredicateAttr::UGE,
            _ => {
                return self.refuse(Opcode::CmpI, "unsupported integer compare predicate");
            }
        };
        let Some((ops, res_ty)) = parts(ctx, op, 2) else {
            return self.refuse(Opcode::CmpI, "operand count or result type unreadable");
        };
        let c = air::icmp(ctx, attr, res_ty, ops[0], ops[1]);
        rewriter.insert_operation(ctx, c);
        rewriter.replace_operation(ctx, op, c);
        self.converted += 1;
        Ok(())
    }
    /// `select`, which is the same three operands in air.
    pub(crate) fn rewrite_select(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some((ops, res_ty)) = parts(ctx, op, 3) else {
            return self.refuse(Opcode::Select, "operand count or result type unreadable");
        };
        let s = air::select(ctx, res_ty, ops[0], ops[1], ops[2]);
        rewriter.insert_operation(ctx, s);
        rewriter.replace_operation(ctx, op, s);
        self.converted += 1;
        Ok(())
    }
    /// `fpext`/`fptrunc` between the compute and result types when they differ.
    pub(crate) fn fpcast_if(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: Value,
        from: TypeHandle,
        to: TypeHandle,
    ) -> Option<Value> {
        fpcast(ctx, rewriter, v, from, to, to).or_else(|| {
            self.refuse_same_width_fpcast();
            None
        })
    }
    /// A float intrinsic call. `@air.<name>.<suffix>`.
    pub(crate) fn rewrite_float_call(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        base: &str,
        n_args: usize,
    ) -> Result<()> {
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            return self.refuse(code, "result type unreadable");
        };
        let Some((cty, suf, rty)) = fp_intr_info(ctx, wax_res) else {
            return self.refuse(code, "no float intrinsic at this result type");
        };
        let ops: Vec<_> = op.deref(ctx).operands().collect();
        if ops.len() < n_args {
            return self.refuse(code, "too few operands for the intrinsic");
        }
        let mut args: Vec<Value> = Vec::with_capacity(n_args);
        for &a in &ops[..n_args] {
            let Some(c) = self.fpcast_if(ctx, rewriter, a, rty, cty) else {
                return Ok(());
            };
            args.push(c);
        }
        let call = air::call(ctx, &format!("{base}.{suf}"), Some(cty), args);
        let r = emit(ctx, rewriter, call);
        let Some(r) = self.fpcast_if(ctx, rewriter, r, cty, rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    pub(crate) fn rewrite_cast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        let (src_v, src_w, dst_w) = {
            let o = op.deref(ctx);
            let sv = o.operands().next();
            let src = sv.map(|v| v.get_type(ctx));
            let dst = (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx));
            (sv, src, dst)
        };
        let (Some(src_v), Some(src_w), Some(dst_w)) = (src_v, src_w, dst_w) else {
            return self.refuse(code, "no src or no dst");
        };
        // SGMMA fragment tile of `<64 x elem>` registers. Cast is applied per fragment, not elem.
        if let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, src_v) {
            let src_hist = operands_info
                .lookup_most_recent_type(src_v)
                .unwrap_or(src_w);
            let (st, dt) = (wax_type_of(ctx, src_hist), wax_type_of(ctx, dst_w));
            let (Some(WaxType::Tile(st)), Some(WaxType::Tile(dt))) = (st, dt) else {
                return self.refuse(code, "incorrect type");
            };
            let (se, de) = (st.element_type.clone(), dt.element_type.clone());
            let (Some(se), Some(de)) = (
                match &se {
                    wax_ir::types::TileElementType::Scalar(s) => Some(s),
                    _ => None,
                },
                match &de {
                    wax_ir::types::TileElementType::Scalar(s) => Some(s),
                    _ => None,
                },
            ) else {
                return self.refuse(code, "incorrect type");
            };
            let (Some((src_elem, _)), Some((dst_elem, _))) = (
                crate::types::float_elem(ctx, se),
                crate::types::float_elem(ctx, de),
            ) else {
                return self.refuse(code, "incorrect type");
            };
            let Some(grid) = self.grid_of(src_v).cloned() else {
                return self.refuse(code, "no fragment grid");
            };
            if grid.len() != nr * nc {
                return self.refuse(code, "incorrect fragment grid size");
            }
            let out: Vec<Value> = if src_elem == dst_elem {
                grid
            } else {
                let have = t::vec_ty(ctx, 64, src_elem);
                let want = t::vec_ty(ctx, 64, dst_elem);
                let mut cast = Vec::with_capacity(grid.len());
                for &f in grid.iter() {
                    let Some(c) = self.fpcast_frag(ctx, rewriter, f, have, want) else {
                        return Ok(());
                    };
                    cast.push(c);
                }
                cast
            };
            let Some(&rep) = out.first() else {
                return self.refuse(code, "invalid cast result");
            };
            let result0 = op.deref(ctx).get_result(0);
            self.frags.grids.insert(result0, out.clone());
            self.frags.grids.insert(rep, out);
            rewriter.replace_operation_with_values(ctx, op, vec![rep]);
            self.converted += 1;
            return Ok(());
        }
        // Extract types.
        let src_hist = operands_info
            .lookup_most_recent_type(src_v)
            .unwrap_or(src_w);
        let src_ll = air_ty_for(ctx, src_hist);
        let dst_ll = air_ty_for(ctx, dst_w);
        let (Some(src_ll), Some(dst_ll)) = (src_ll, dst_ll) else {
            return self.refuse(code, "incorrect cast type");
        };
        // If types are identical there is no cast to be done.
        if src_ll == dst_ll {
            let Some((ops, _)) = parts(ctx, op, 1) else {
                return self.refuse(code, "cast parts");
            };
            rewriter.replace_operation_with_values(ctx, op, vec![ops[0]]);
            self.converted += 1;
            return Ok(());
        }
        // Extract information required to decide casting behaviour.
        let signed = wax_int_attr(ctx, op, &keys::ATTR_SIGNEDNESS) != Some(0);
        let (Some(src_bits), Some(dst_bits)) = (scalar_bits(ctx, src_ll), scalar_bits(ctx, dst_ll))
        else {
            return self.refuse(code, "cast between types with no scalar width");
        };
        let kind = match code {
            Opcode::Bitcast => {
                if src_bits != dst_bits {
                    return self.refuse(code, "bitcast between types of different width");
                }
                CastOpAttr::Bitcast
            }
            Opcode::ExtI => {
                if signed {
                    CastOpAttr::SExt
                } else {
                    CastOpAttr::ZExt
                }
            }
            Opcode::TruncI => CastOpAttr::Trunc,
            Opcode::IToF => {
                if signed {
                    CastOpAttr::SIToFP
                } else {
                    CastOpAttr::UIToFP
                }
            }
            Opcode::FToI => {
                if signed {
                    CastOpAttr::FPToSI
                } else {
                    CastOpAttr::FPToUI
                }
            }
            Opcode::FToF => {
                if src_bits > dst_bits {
                    CastOpAttr::FPTrunc
                } else {
                    CastOpAttr::FPExt
                }
            }
            other => return self.refuse(other, "not a cast opcode"),
        };
        let Some((ops, res_ty)) = parts(ctx, op, 1) else {
            return self.refuse(code, "cast parts");
        };
        let c = air::cast(ctx, kind, res_ty, ops[0]);
        rewriter.insert_operation(ctx, c);
        rewriter.replace_operation(ctx, op, c);
        self.converted += 1;
        Ok(())
    }
    /// `iota`. The element's indewx index within its tile.
    pub(crate) fn rewrite_iota(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let i32_t = t::i32_ty(ctx);
        let i32_0 = air_const_int(ctx, BITS_32, 0);

        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(n) = wax_res
            .and_then(|t| wax_type_of(ctx, t))
            .as_ref()
            .and_then(crate::types::tile_elem_count)
        else {
            return self.refuse(Opcode::Iota, "result tile has no element count");
        };
        let Some(tp) = self.control.tp else {
            return self.refuse(Opcode::Iota, "no thread position parameter");
        };
        let lane = if n <= 1 {
            i32_0
        } else {
            let x = air::extract_elem(ctx, i32_t, tp, i32_0);
            let x = emit(ctx, rewriter, x);
            if n > 32 || !(n as u64).is_power_of_two() {
                x
            } else {
                let m = air_const_int(ctx, BITS_32, (n - 1) as u64);
                let a = air::bin(ctx, BinKind::And, i32_t, x, m);
                emit(ctx, rewriter, a)
            }
        };
        rewriter.replace_operation_with_values(ctx, op, vec![lane]);
        self.converted += 1;
        Ok(())
    }
    /// Float comparison.
    pub(crate) fn rewrite_cmp_float(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron_llvm::attributes::FCmpPredicateAttr;
        // Note that the default to 0 (OEQ) from a missing attribute should not occur by construction.
        let attr = match wax_int_attr(ctx, op, &keys::ATTR_COMPARISON_PREDICATE).unwrap_or(0) {
            0 => FCmpPredicateAttr::OEQ,
            1 => FCmpPredicateAttr::ONE,
            2 => FCmpPredicateAttr::OLT,
            3 => FCmpPredicateAttr::OLE,
            4 => FCmpPredicateAttr::OGT,
            5 => FCmpPredicateAttr::OGE,
            _ => {
                return self.refuse(Opcode::CmpF, "unsupported float cmp predicate code");
            }
        };
        let Some((ops, res_ty)) = parts(ctx, op, 2) else {
            return self.refuse(Opcode::CmpF, "operand count or result type unreadable");
        };
        let c = air::fcmp(ctx, attr, res_ty, ops[0], ops[1]);
        rewriter.insert_operation(ctx, c);
        rewriter.replace_operation(ctx, op, c);
        self.converted += 1;
        Ok(())
    }
    /// `min`/`max` spelled as compare and select.
    // TODO: Check if this is all metal supports.
    pub(crate) fn rewrite_minmax_select(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        use pliron_llvm::attributes::{FCmpPredicateAttr, ICmpPredicateAttr};
        let Some((ops, res_ty)) = parts(ctx, op, 2) else {
            return self.refuse(code, "operand count or result type unreadable");
        };
        let i1_t = t::i1_ty(ctx);
        let cmp = match code {
            Opcode::MinF => air::fcmp(ctx, FCmpPredicateAttr::OLT, i1_t, ops[0], ops[1]),
            Opcode::MaxI => air::icmp(ctx, ICmpPredicateAttr::SGT, i1_t, ops[0], ops[1]),
            Opcode::MinI => air::icmp(ctx, ICmpPredicateAttr::SLT, i1_t, ops[0], ops[1]),
            _ => {
                return self.refuse(code, "opcode is not a min or max");
            }
        };
        let cond = emit(ctx, rewriter, cmp);
        let s = air::select(ctx, res_ty, cond, ops[0], ops[1]);
        rewriter.insert_operation(ctx, s);
        rewriter.replace_operation(ctx, op, s);
        self.converted += 1;
        Ok(())
    }
    /// `fma(a, b, c)` as a separate multiply and add.
    /// We leave the `@llvm.fma` contraction to the AIR compiler.
    // TODO: Verify contraction.
    pub(crate) fn rewrite_fma(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let Some((ops, res_ty)) = parts(ctx, op, 3) else {
            return self.refuse(Opcode::Fma, "operand count or result type unreadable");
        };
        let m = air::bin(ctx, BinKind::FMul, res_ty, ops[0], ops[1]);
        let mv = emit(ctx, rewriter, m);
        let a = air::bin(ctx, BinKind::FAdd, res_ty, mv, ops[2]);
        rewriter.insert_operation(ctx, a);
        rewriter.replace_operation(ctx, op, a);
        self.converted += 1;
        Ok(())
    }
    /// Integer negation. `0 - x`.
    pub(crate) fn rewrite_neg_int(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let Some((ops, res_ty)) = parts(ctx, op, 1) else {
            return self.refuse(Opcode::NegI, "operand count or result type unreadable");
        };
        let Some(z) = air::const_bits(ctx, res_ty, 0) else {
            return self.refuse(Opcode::NegI, "result type has no constant spelling");
        };
        let zv = emit_op(ctx, rewriter, z);
        let s = air::bin(ctx, BinKind::Sub, res_ty, zv, ops[0]);
        rewriter.insert_operation(ctx, s);
        rewriter.replace_operation(ctx, op, s);
        self.converted += 1;
        Ok(())
    }
    /// `rsqrt` as `1 / sqrt(x)`. `air.rsqrt` exists but only in f32. One shape covers all
    /// three element types.
    pub(crate) fn rewrite_rsqrt(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            return self.refuse(Opcode::Rsqrt, "result type unreadable");
        };
        let Some((cty, suf, rty)) = fp_intr_info(ctx, wax_res) else {
            return self.refuse(Opcode::Rsqrt, "no float intrinsic at this result type");
        };
        let Some(&a) = op.deref(ctx).operands().collect::<Vec<_>>().first() else {
            return self.refuse(Opcode::Rsqrt, "no source operand");
        };
        let cty_h = cty;
        let Some(a) = self.fpcast_if(ctx, rewriter, a, rty, cty) else {
            return Ok(());
        };
        let sq = air::call(ctx, &format!("@air.sqrt.{suf}"), Some(cty_h), vec![a]);
        let sqv = emit(ctx, rewriter, sq);
        let Some(one_bits) = float_one_bits(ctx, cty) else {
            return self.refuse(Opcode::Rsqrt, "no 1.0 constant at this element type");
        };
        let Some(one) = air::const_bits(ctx, cty_h, one_bits) else {
            return self.refuse(Opcode::Rsqrt, "no 1.0 constant at the compute type");
        };
        let onev = emit_op(ctx, rewriter, one);
        let d = air::bin(ctx, BinKind::FDiv, cty_h, onev, sqv);
        let r = emit(ctx, rewriter, d);
        let Some(r) = self.fpcast_if(ctx, rewriter, r, cty, rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// Rewrite pointer ops. `int_to_ptr`, `addptr`, `load_ptr`.
    pub(crate) fn rewrite_ptr_op(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        let res_tys = op_result_types(ctx, op);
        let Some(&first_res) = res_tys.first() else {
            return self.refuse(code, "no result");
        };
        let Some(wax_res) = wax_type_of(ctx, first_res) else {
            return self.refuse(code, "result type unreadable");
        };
        let Some(elem) =
            crate::types::param_elem(&wax_res).and_then(|s| crate::types::mem_elem_ty(ctx, &s))
        else {
            return self.refuse(code, "element type is not stored verbatim");
        };
        let operands: Vec<_> = op.deref(ctx).operands().collect();

        let mut replacements: Vec<Value> = Vec::new();
        match code {
            Opcode::IntToPtr => {
                let Some(&src) = operands.first() else {
                    return self.refuse(code, "no source");
                };
                let ty = t::device_ptr_ty(ctx, elem);
                let c = air::cast(ctx, CastOpAttr::IntToPtr, ty, src);
                replacements.push(emit(ctx, rewriter, c));
            }
            Opcode::Offset => {
                let (Some(&base), Some(&off)) = (operands.first(), operands.get(1)) else {
                    return self.refuse(code, "offset needs a base and an index");
                };
                let ty = t::device_ptr_ty(ctx, elem);
                let elem_h = elem;
                let g = air::GepOp::new(ctx, ty, base, off, elem_h);
                replacements.push(emit_op(ctx, rewriter, g));
            }
            Opcode::LoadPtrTko => {
                let Some(&ptr) = operands.first() else {
                    return self.refuse(code, "no pointer");
                };
                let ty = elem;
                let Some(al) = align_bytes(ctx, ty) else {
                    return self.refuse(code, "load element has no natural alignment");
                };
                let ld = air::LoadOp::new(ctx, ty, ptr, al);
                let mut loaded = emit_op(ctx, rewriter, ld);
                // Apply mask to load if present.
                if let Some(&mask) = operands.get(1) {
                    let pad = match operands.get(2) {
                        Some(&p) => p,
                        None => {
                            let z = air::ZeroOp::new(ctx, ty);
                            emit_op(ctx, rewriter, z)
                        }
                    };
                    let sel = air::select(ctx, ty, mask, loaded, pad);
                    loaded = emit(ctx, rewriter, sel);
                }
                replacements.push(loaded);
            }
            _ => {
                return self.refuse(code, "not a pointer op");
            }
        }
        self.finish_replacement(ctx, rewriter, op, &res_tys, replacements);
        Ok(())
    }
}

/// Operands and converted result type, the shape almost every arm needs.
fn parts(ctx: &Context, op: Ptr<Operation>, n_operands: usize) -> Option<(Vec<Value>, TypeHandle)> {
    let o = op.deref(ctx);
    let ops: Vec<_> = o.operands().collect();
    if ops.len() < n_operands {
        return None;
    }
    let ty = (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx));
    let res_ty = air_ty_for(ctx, ty?)?;
    Some((ops, res_ty))
}
