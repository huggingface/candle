//! The elementwise arms: constants and casts, arithmetic, comparisons, atomics,
//! float calls, and tensor-shape reads.

use super::*;

/// The partition's element type as a wax scalar, for the wide load's alignment.
pub(crate) fn elem_scalar_of(view_ty: &wax_ir::types::Type) -> ScalarType {
    use wax_ir::types::Type as WT;
    match view_ty {
        WT::PartitionView(pv) => pv.tensor_view.element_type.clone(),
        _ => ScalarType::F32,
    }
}

/// The wax opcode of an op, or `None` if it is not a wax op.
///
/// Fallible because the conversion walks ops it has itself inserted: an `air.gep` has no wax
/// opcode, and the dialect's lookup panics rather than saying so.
pub(crate) fn try_opcode_of(ctx: &Context, op: Ptr<Operation>) -> Option<Opcode> {
    let is_wax = Operation::get_op_dyn(op, ctx)
        .op_ref()
        .get_opid()
        .to_string()
        .starts_with("wax.");
    is_wax.then(|| wax_ir::dialect::source::opcode_of(ctx, op))
}

/// The wax opcode of an op known to be one.
pub(crate) fn opcode_of(ctx: &Context, op: Ptr<Operation>) -> Opcode {
    wax_ir::dialect::source::opcode_of(ctx, op)
}

/// `operandSegmentSizes`, as `(base, dyn_shape, dyn_strides)`.
///
/// A `make_tensor_view` carries its dynamic extents and dynamic strides as trailing operand
/// groups; reading past the base group without this walks into the strides.
pub(crate) fn wax_operand_segments(ctx: &Context, op: Ptr<Operation>) -> (usize, usize, usize) {
    use wax_ir::attr::Attribute as WA;
    use wax_ir::dialect::attr_mirror::WaxAttr;
    let Ok(key) = "operandSegmentSizes".try_into() else {
        return (1, 0, 0);
    };
    match op.deref(ctx).attributes.get::<WaxAttr>(&key) {
        Some(a) => match &a.0 {
            WA::DenseI32Array(arr) => {
                let get = |i: usize| arr.get(i).copied().unwrap_or(0) as usize;
                (get(0), get(1), get(2))
            }
            WA::Array(arr) => {
                let get = |i: usize| match arr.get(i) {
                    Some(WA::Integer(v, _)) => *v as usize,
                    _ => 0,
                };
                (get(0), get(1), get(2))
            }
            _ => (1, 0, 0),
        },
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
        use pliron::r#type::Typed;
        // A FRAGMENT constant is a grid of `<64 x float>`, not one scalar splat. Only the zero
        // splat is supported, which is the accumulator init; `walk.rs` refuses any other
        // multi-fragment constant too, so this is the same subset rather than a new limit.
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
                self.unconverted.note(Opcode::Constant);
                return Ok(());
            }
            let ty = o.get_result(0).get_type(ctx);
            // The value rides as a dense-elements attribute; the per-lane value of a
            // (splat) tile constant is its element, decoded the way `walk.rs` decodes it,
            // so the two routes agree on non-f32 element types too.
            let bits = o.attributes.0.iter().find_map(|(k, a)| {
                let a = a.downcast_ref::<wax_ir::dialect::attr_mirror::WaxAttr>()?;
                if k.to_string() != "value" {
                    return None;
                }
                let wax_ir::Attribute::DenseElements(de) = &a.0 else {
                    return None;
                };
                let wt = wax_type_of(ctx, ty)?;
                Some(match crate::lower::dense_elements_value(de, &wt) {
                    crate::lower::ImmVal::Int(i) => i as u64,
                    crate::lower::ImmVal::FloatBits(b) => b,
                    crate::lower::ImmVal::Zero => 0,
                    _ => return None,
                })
            });
            (ty, bits)
        };
        let Some(bits) = bits else {
            self.unconverted.note(Opcode::Constant);
            return Ok(());
        };
        let Some(res_ty) = self.air_ty(ctx, res_ty) else {
            self.unconverted.note(Opcode::Constant);
            return Ok(());
        };
        let c = air::ConstBitsOp::new(ctx, res_ty, bits);
        rewriter.insert_operation(ctx, c.get_operation());
        rewriter.replace_operation(ctx, op, c.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// Float negation: `air.unop` with the `fneg` code, preserving the operand's type -
    /// `walk.rs`'s `f.fneg`, which does NOT compute in f32 (an f16 neg stays an f16 neg).
    pub(crate) fn rewrite_neg_f(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::UnOpAttr;
        use pliron::r#type::Typed;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1 && o.get_num_operands() == 1)
                .then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            self.unconverted.note(Opcode::NegF);
            return Ok(());
        };
        // NegF is a FLOAT operation: the `fneg` unop has no integer spelling (an integer
        // negation lowers as a subtract), so a non-float element is not an input the
        // frontend emits - refuse it the way the CFG does rather than invent a lowering.
        let Some(wax) = wax_type_of(ctx, wax_res)
            .and_then(|w| crate::types::float_elem(&crate::types::scalar_of(&w)).map(|_| w))
        else {
            self.unconverted.note(Opcode::NegF);
            return Ok(());
        };
        let Some(res_ty) = self.air_ty(ctx, wax_res) else {
            let _ = wax;
            self.unconverted.note(Opcode::NegF);
            return Ok(());
        };
        let src = op.deref(ctx).operands().next();
        let Some(src) = src else {
            self.unconverted.note(Opcode::NegF);
            return Ok(());
        };
        let u = air::UnOp::new(ctx, res_ty, src);
        u.set_op(ctx, UnOpAttr::FNeg);
        rewriter.insert_operation(ctx, u.get_operation());
        rewriter.replace_operation(ctx, op, u.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `atan2(y, x)`: both operands promoted to the compute type, `call fast
    /// @air.fast_atan2.{suf}`, result truncated back - the two-arg sibling of
    /// `rewrite_float_call`, and the only place that spelling exists.
    pub(crate) fn rewrite_atan2(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron::r#type::Typed;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1 && o.get_num_operands() == 2)
                .then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            self.unconverted.note(Opcode::Atan2);
            return Ok(());
        };
        let Some((cty, suf, rty)) = self.intrin_ty(ctx, wax_res) else {
            self.unconverted.note(Opcode::Atan2);
            return Ok(());
        };
        let ops: Vec<_> = op.deref(ctx).operands().collect();
        // The operand ORDER is the ABI: y first, as `walk.rs` calls it.
        let y = self.fpcast_if(ctx, rewriter, ops[0], &rty, &cty);
        let x = self.fpcast_if(ctx, rewriter, ops[1], &rty, &cty);
        let cty_h = crate::air_dialect::types::to_pliron(ctx, &cty);
        let c = air::CallOp::new(
            ctx,
            &format!("@air.fast_atan2.{suf}"),
            Some(cty_h),
            vec![y, x],
        );
        air::set_fast(c.get_operation(), ctx);
        rewriter.insert_operation(ctx, c.get_operation());
        let r = c.get_operation().deref(ctx).get_result(0);
        let r = self.fpcast_if(ctx, rewriter, r, &cty, &rty);
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// The tensor's STATIC shape, one i32 per result - `walk.rs`'s `shape_query`.
    /// A dynamic extent keeps the kernel on the CFG route, exactly as there.
    pub(crate) fn rewrite_get_tensor_shape(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use pliron::r#type::Typed;
        let (src, n_res) = {
            let o = op.deref(ctx);
            (o.operands().next(), o.get_num_results())
        };
        let Some(src) = src else {
            self.unconverted.note(Opcode::GetTensorShape);
            return Ok(());
        };
        // The shape lives on the WAX TensorView type, and the operand may already be the
        // pointer its view folded to: read the pre-conversion history, not the live type.
        let src_wax = operands_info
            .lookup_most_recent_type(src)
            .or_else(|| Some(src.get_type(ctx)))
            .and_then(|h| wax_type_of(ctx, h));
        let Some(WaxType::TensorView(tv)) = src_wax else {
            self.unconverted.note(Opcode::GetTensorShape);
            return Ok(());
        };
        if tv.shape.contains(&wax_ir::DYNAMIC) || tv.shape.len() != n_res {
            self.unconverted.note(Opcode::GetTensorShape);
            return Ok(());
        }
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let mut vals = Vec::with_capacity(n_res);
        for &d in &tv.shape {
            let c = air::ConstBitsOp::new(ctx, i32_h, d as u64);
            rewriter.insert_operation(ctx, c.get_operation());
            vals.push(c.get_operation().deref(ctx).get_result(0));
        }
        rewriter.replace_operation_with_values(ctx, op, vals);
        self.converted += 1;
        Ok(())
    }
    /// Atomic read-modify-write: `old = @air.atomic.{space}.{op}.{ty}(ptr, val, i32 0,
    /// i32 scope, i1 true)`, the token result bound to `undef i32` - `walk.rs`'s
    /// `atomic_rmw`. The element type and memory space come from the POINTER operand's
    /// wax type, exactly as `ptr_elem_type_and_as` gives them to the CFG.
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
            self.bail(code, "atomic rmw needs a pointer and a value");
            return Ok(());
        };
        let (elem_ll, asn) = match self.wax_operand_ty(ctx, operands_info, ptr) {
            Some(ty) => crate::lower::ptr_elem_type_and_as(&ty),
            None => {
                self.bail(code, "pointer operand type unreadable");
                return Ok(());
            }
        };
        let mode = self.wax_attr_i64(ctx, op, "mode").unwrap_or(3);
        let scope = crate::types::atomic_scope(
            self.wax_attr_i64(ctx, op, "memory_scope").unwrap_or(1),
        );
        let (intrinsic, air_ty) = crate::types::atomic_rmw_intrinsic(mode, &elem_ll, asn);
        let elem_h = crate::air_dialect::types::to_pliron(ctx, &crate::types::atomic_lltype(air_ty));
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let mut const_bits =
            |ty_h: TypeHandle, bits: u64| -> pliron::value::Value {
                let c = air::ConstBitsOp::new(ctx, ty_h, bits);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
        let c0 = const_bits(i32_h, 0);
        let cscope = const_bits(i32_h, scope as u64);
        let ctrue = const_bits(i1_h, 1);
        let call =
            air::CallOp::new(ctx, &intrinsic, Some(elem_h), vec![ptr, val, c0, cscope, ctrue]);
        rewriter.insert_operation(ctx, call.get_operation());
        let old = call.get_operation().deref(ctx).get_result(0);
        let token = {
            let u = air::UndefOp::new(ctx, i32_h);
            rewriter.insert_operation(ctx, u.get_operation());
            u.get_operation().deref(ctx).get_result(0)
        };
        rewriter.replace_operation_with_values(ctx, op, vec![old, token]);
        self.converted += 1;
        Ok(())
    }
    /// Atomic compare-and-swap: `@air.atomic.{space}.cmpxchg.weak.{ty}`. The expected
    /// value lives in a stack slot the intrinsic reads/updates - `walk.rs`'s
    /// `atomic_cas`: alloca, store, then the seven-argument call.
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
            self.bail(code, "atomic cas needs pointer, expected and new value");
            return Ok(());
        };
        let (elem_ll, asn) = match self.wax_operand_ty(ctx, operands_info, ptr) {
            Some(ty) => crate::lower::ptr_elem_type_and_as(&ty),
            None => {
                self.bail(code, "pointer operand type unreadable");
                return Ok(());
            }
        };
        let scope = crate::types::atomic_scope(
            self.wax_attr_i64(ctx, op, "memory_scope").unwrap_or(1),
        );
        let mem_space = if asn == 3 { "local" } else { "global" };
        let air_ty = crate::types::atomic_air_ty(&elem_ll);
        let intrinsic = format!("@air.atomic.{mem_space}.cmpxchg.weak.{air_ty}");
        let elem = crate::types::atomic_lltype(air_ty);
        let elem_h = crate::air_dialect::types::to_pliron(ctx, &elem);
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        // The expected value's stack slot: one element, align 4, no count operand - the
        // bitcode form puts the length in the type and passes a literal 1.
        let exp = {
            let slot_ty = crate::air_dialect::types::to_pliron(
                ctx,
                &LlType::Ptr {
                    pointee: Box::new(elem.clone()),
                    addrspace: 0,
                },
            );
            let a = air::AllocaOp::new(ctx, slot_ty, None).get_operation();
            air::set_elem_ty(a, ctx, elem_h);
            air::set_align(a, ctx, 4);
            rewriter.insert_operation(ctx, a);
            a.deref(ctx).get_result(0)
        };
        let st = air::StoreOp::new(ctx, cmp, exp);
        air::set_align(st.get_operation(), ctx, 4);
        rewriter.insert_operation(ctx, st.get_operation());
        let mut const_bits =
            |ty_h: TypeHandle, bits: u64| -> pliron::value::Value {
                let c = air::ConstBitsOp::new(ctx, ty_h, bits);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
        let c0 = const_bits(i32_h, 0);
        let cscope = const_bits(i32_h, scope as u64);
        let ctrue = const_bits(i1_h, 1);
        let call = air::CallOp::new(
            ctx,
            &intrinsic,
            Some(elem_h),
            vec![ptr, exp, new_val, c0, c0, cscope, ctrue],
        );
        rewriter.insert_operation(ctx, call.get_operation());
        let old = call.get_operation().deref(ctx).get_result(0);
        let token = {
            let u = air::UndefOp::new(ctx, i32_h);
            rewriter.insert_operation(ctx, u.get_operation());
            u.get_operation().deref(ctx).get_result(0)
        };
        rewriter.replace_operation_with_values(ctx, op, vec![old, token]);
        self.converted += 1;
        Ok(())
    }
    /// The wax type of a (possibly already-converted) operand, from the history the
    /// driver keeps - the shape and element type only exist on the wax side.
    pub(crate) fn wax_operand_ty(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        v: pliron::value::Value,
    ) -> Option<WaxType> {
        use pliron::r#type::Typed;
        operands_info
            .lookup_most_recent_type(v)
            .or_else(|| Some(v.get_type(ctx)))
            .and_then(|h| wax_type_of(ctx, h))
    }
    /// A wax binary op becomes `air.bin` with the operator in an attribute.
    pub(crate) fn rewrite_binary(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        use pliron::r#type::Typed;
        let attr = match code {
            Opcode::AddF => BinOpAttr::FAdd,
            Opcode::SubF => BinOpAttr::FSub,
            Opcode::MulF => BinOpAttr::FMul,
            Opcode::DivF => BinOpAttr::FDiv,
            Opcode::AddI => BinOpAttr::Add,
            Opcode::SubI => BinOpAttr::Sub,
            Opcode::MulI => BinOpAttr::Mul,
            // Signedness picks the instruction, and an absent attribute means signed - the
            // same default `walk.rs` takes, for the same reason: sign-filling preserves the
            // value where zero-filling would not.
            Opcode::DivI | Opcode::RemI | Opcode::ShRI => {
                let signed = self.wax_attr_i64(ctx, op, "signedness") != Some(0);
                match (code, signed) {
                    (Opcode::DivI, true) => BinOpAttr::SDiv,
                    (Opcode::DivI, false) => BinOpAttr::UDiv,
                    (Opcode::RemI, true) => BinOpAttr::SRem,
                    (Opcode::RemI, false) => BinOpAttr::URem,
                    (Opcode::ShRI, true) => BinOpAttr::AShr,
                    (Opcode::ShRI, false) => BinOpAttr::LShr,
                    _ => unreachable!("outer match restricts these three"),
                }
            }
            Opcode::ShLI => BinOpAttr::Shl,
            Opcode::AndI => BinOpAttr::And,
            Opcode::OrI => BinOpAttr::Or,
            Opcode::XOrI => BinOpAttr::Xor,
            _ => {
                self.unconverted.note(code);
                return Ok(());
            }
        };
        // Read everything out before any borrow ends, so the checker sees one scope.
        let parts = {
            let o = op.deref(ctx);
            let mut it = o.operands();
            let two = it.next().zip(it.next());
            let one_result = o.get_num_results() == 1;
            let ty = one_result.then(|| o.get_result(0).get_type(ctx));
            two.zip(ty)
        };
        let Some(((lhs, rhs), res_ty)) = parts else {
            self.unconverted.note(code);
            return Ok(());
        };
        // The op's result type on the air side (the walk's `binop_result_ty`), and the
        // instruction type for integer binops, where operand and result agree.
        let out_ll = wax_type_of(ctx, res_ty).as_ref().and_then(air_type_for);
        let Some(out_ty) = out_ll.as_ref().map(|ll| crate::air_dialect::types::to_pliron(ctx, ll))
        else {
            self.unconverted.note(code);
            return Ok(());
        };
        // Float binops mirror `walk.rs::float_binop`: the frontend may hand them mixed
        // widths - an f16 loop carry plus an f32 product, result f16. Promote both
        // operands to the wider type, compute there, and narrow back to the op's result
        // type. A raw mixed-precision `air.bin` is not legal LLVM and reads back as
        // `Invalid record`.
        let (lhs, rhs, bin_ty, narrow) = if matches!(
            code,
            Opcode::AddF | Opcode::SubF | Opcode::MulF | Opcode::DivF
        ) {
            let ta = crate::air_dialect::types::from_pliron(ctx, lhs.get_type(ctx));
            let tb = crate::air_dialect::types::from_pliron(ctx, rhs.get_type(ctx));
            let (Some(ta), Some(tb)) = (ta, tb) else {
                self.unconverted.note(code);
                return Ok(());
            };
            let cty = if ta == tb {
                ta.clone()
            } else {
                match crate::types::wider_float(&ta, &tb) {
                    Some(c) => c,
                    None => {
                        self.unconverted.note(code);
                        return Ok(());
                    }
                }
            };
            let lhs = self.fpcast_if(ctx, rewriter, lhs, &ta, &cty);
            let rhs = self.fpcast_if(ctx, rewriter, rhs, &tb, &cty);
            let narrow = out_ll.filter(|r| r != &cty).map(|r| (cty.clone(), r));
            (lhs, rhs, crate::air_dialect::types::to_pliron(ctx, &cty), narrow)
        } else {
            (lhs, rhs, out_ty, None)
        };
        let b = air::BinOp::new(ctx, bin_ty, lhs, rhs);
        b.set_op(ctx, attr);
        rewriter.insert_operation(ctx, b.get_operation());
        match narrow {
            None => rewriter.replace_operation(ctx, op, b.get_operation()),
            Some((from, to)) => {
                let out = b.get_operation().deref(ctx).get_result(0);
                let final_v = self.fpcast_if(ctx, rewriter, out, &from, &to);
                rewriter.replace_operation_with_values(ctx, op, vec![final_v]);
            }
        }
        self.converted += 1;
        Ok(())
    }
    /// An integer attribute the frontend attached to a wax op.
    ///
    /// Generic wax attributes ride as `WaxAttr`, which is where `ir::from_source` reads the
    /// names `walk.rs` uses. Reading the same names here is what keeps the two paths agreeing
    /// on things the opcode alone does not say - signedness, a comparison predicate, a
    /// reduction axis.
    pub(crate) fn wax_attr_i64(&self, ctx: &Context, op: Ptr<Operation>, key: &str) -> Option<i64> {
        use wax_ir::attr::Attribute as WA;
        use wax_ir::dialect::attr_mirror::WaxAttr;
        let k = key.try_into().ok()?;
        match &op.deref(ctx).attributes.get::<WaxAttr>(&k)?.0 {
            WA::Integer(i, _) => Some(*i),
            WA::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }
    /// Operands and converted result type, the shape almost every arm needs.
    pub(crate) fn parts(
        &mut self,
        ctx: &mut Context,
        op: Ptr<Operation>,
        n_operands: usize,
    ) -> Option<(Vec<pliron::value::Value>, TypeHandle)> {
        use pliron::r#type::Typed;
        let read = {
            let o = op.deref(ctx);
            let ops: Vec<_> = o.operands().collect();
            let ty = (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx));
            (ops, ty)
        };
        let (ops, ty) = read;
        if ops.len() < n_operands {
            return None;
        }
        let res_ty = self.air_ty(ctx, ty?)?;
        Some((ops, res_ty))
    }
    /// `get_tile_block_id`: the threadgroup position, one result per axis.
    ///
    /// Three results, not one - the CFG path binds axis 0, 1 and 2 separately, each an
    /// `extractelement` from `%_tg`.
    pub(crate) fn rewrite_tile_block_id(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some(tg) = self.control.tg else {
            self.unconverted.note(Opcode::GetTileBlockId);
            return Ok(());
        };
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let n = op.deref(ctx).get_num_results();
        let mut replacements = Vec::with_capacity(n);
        for axis in 0..n {
            let idx = air::ConstBitsOp::new(ctx, i32_ty, axis as u64);
            rewriter.insert_operation(ctx, idx.get_operation());
            let idx_v = idx.get_operation().deref(ctx).get_result(0);
            let e = air::ExtractElemOp::new(ctx, i32_ty, tg, idx_v);
            rewriter.insert_operation(ctx, e.get_operation());
            replacements.push(e.get_operation().deref(ctx).get_result(0));
        }
        rewriter.replace_operation_with_values(ctx, op, replacements);
        self.converted += 1;
        Ok(())
    }
    /// Integer comparison. The predicate mapping mirrors `walk.rs`, including its default:
    /// `signedness` absent means signed, which is the conservative reading.
    pub(crate) fn rewrite_cmp_int(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::ICmpPredAttr;
        let signed = self.wax_attr_i64(ctx, op, "signedness") != Some(0);
        let pred = self
            .wax_attr_i64(ctx, op, "comparison_predicate")
            .unwrap_or(0);
        let attr = match (pred, signed) {
            (0, _) => ICmpPredAttr::Eq,
            (1, _) => ICmpPredAttr::Ne,
            (2, true) => ICmpPredAttr::Slt,
            (2, false) => ICmpPredAttr::Ult,
            (3, true) => ICmpPredAttr::Sle,
            (3, false) => ICmpPredAttr::Ule,
            (4, true) => ICmpPredAttr::Sgt,
            (4, false) => ICmpPredAttr::Ugt,
            (5, true) => ICmpPredAttr::Sge,
            (5, false) => ICmpPredAttr::Uge,
            _ => {
                self.unconverted.note(Opcode::CmpI);
                return Ok(());
            }
        };
        let Some((ops, res_ty)) = self.parts(ctx, op, 2) else {
            self.unconverted.note(Opcode::CmpI);
            return Ok(());
        };
        let c = air::ICmpOp::new(ctx, res_ty, ops[0], ops[1]);
        c.set_pred(ctx, attr);
        rewriter.insert_operation(ctx, c.get_operation());
        rewriter.replace_operation(ctx, op, c.get_operation());
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
        let Some((ops, res_ty)) = self.parts(ctx, op, 3) else {
            self.unconverted.note(Opcode::Select);
            return Ok(());
        };
        let s = air::SelectOp::new(ctx, res_ty, ops[0], ops[1], ops[2]);
        rewriter.insert_operation(ctx, s.get_operation());
        rewriter.replace_operation(ctx, op, s.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `(compute type, AIR suffix, result type)` for a float intrinsic.
    ///
    /// Mirrors `walk.rs::intrin_ty`: bf16 has no AGX intrinsics, so it computes in f32 and
    /// truncates back. Derived from the op's RESULT type rather than an operand, because by
    /// the time an arm runs its operands may already carry converted air types.
    pub(crate) fn intrin_ty(&self, ctx: &Context, ty: TypeHandle) -> Option<(LlType, &'static str, LlType)> {
        use wax_ir::types::ScalarType;
        let wax = wax_type_of(ctx, ty)?;
        let scalar = crate::types::scalar_of(&wax);
        let (rty, _) = crate::types::float_elem(&scalar)?;
        let (cty, csuf) = match scalar {
            ScalarType::BF16 => (LlType::Float, "f32"),
            other => crate::types::float_elem(&other)?,
        };
        Some((cty, csuf, rty))
    }
    /// `fpext`/`fptrunc` between the compute and result types when they differ.
    pub(crate) fn fpcast_if(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: pliron::value::Value,
        from: &LlType,
        to: &LlType,
    ) -> pliron::value::Value {
        use crate::air_dialect::attrs::CastOpAttr;
        if from == to {
            return v;
        }
        let to_ty = crate::air_dialect::types::to_pliron(ctx, to);
        let kind = if scalar_bits(to) > scalar_bits(from) {
            CastOpAttr::FPExt
        } else {
            CastOpAttr::FPTrunc
        };
        let c = air::CastOp::new(ctx, to_ty, v);
        c.set_op(ctx, kind);
        rewriter.insert_operation(ctx, c.get_operation());
        c.get_operation().deref(ctx).get_result(0)
    }
    /// A float intrinsic call: `@air.<name>.<suffix>`.
    ///
    /// `@air.*`, never `@llvm.*`. Both build and both create a pipeline, so only value tests
    /// tell them apart - see `project_air_portability`.
    pub(crate) fn rewrite_float_call(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        base: &str,
        n_args: usize,
    ) -> Result<()> {
        use pliron::r#type::Typed;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            self.unconverted.note(code);
            return Ok(());
        };
        let Some((cty, suf, rty)) = self.intrin_ty(ctx, wax_res) else {
            self.unconverted.note(code);
            return Ok(());
        };
        let ops: Vec<_> = op.deref(ctx).operands().collect();
        if ops.len() < n_args {
            self.unconverted.note(code);
            return Ok(());
        }
        let cty_h = crate::air_dialect::types::to_pliron(ctx, &cty);
        let args: Vec<_> = ops[..n_args]
            .iter()
            .map(|&a| self.fpcast_if(ctx, rewriter, a, &rty, &cty))
            .collect();
        let call = air::CallOp::new(ctx, &format!("{base}.{suf}"), Some(cty_h), args);
        rewriter.insert_operation(ctx, call.get_operation());
        let r = call.get_operation().deref(ctx).get_result(0);
        let r = self.fpcast_if(ctx, rewriter, r, &cty, &rty);
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// Numeric conversions, all one air `cast` with the right kind.
    pub(crate) fn rewrite_cast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        use pliron::r#type::Typed;
        let (src_v, src_w, dst_w) = {
            let o = op.deref(ctx);
            let sv = o.operands().next();
            let src = sv.map(|v| v.get_type(ctx));
            let dst = (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx));
            (sv, src, dst)
        };
        let (Some(src_v), Some(src_w), Some(dst_w)) = (src_v, src_w, dst_w) else {
            self.bail(code, "cast-no-src-or-dst");
            return Ok(());
        };
        // SGMMA fragment tile: the value is a GRID of `<64 x elem>` registers, not a
        // single air value - `air_type_for` cannot map it. `walk.rs::cast_op` casts each
        // fragment (one elementwise vector cast per register, or nothing when the element
        // types agree) and binds the grid to the result.
        if let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, src_v) {
            let src_hist = operands_info
                .lookup_most_recent_type(src_v)
                .unwrap_or(src_w);
            let (st, dt) = (wax_type_of(ctx, src_hist), wax_type_of(ctx, dst_w));
            let (Some(wax_ir::types::Type::Tile(st)), Some(wax_ir::types::Type::Tile(dt))) =
                (st, dt)
            else {
                self.bail(code, "fragment-cast-type");
                return Ok(());
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
                self.bail(code, "fragment-cast-type");
                return Ok(());
            };
            let (Some((src_elem, _)), Some((dst_elem, _))) =
                (crate::types::float_elem(se), crate::types::float_elem(de))
            else {
                self.bail(code, "fragment-cast-type");
                return Ok(());
            };
            let Some(grid) = self.grid_of(src_v).cloned() else {
                self.bail(code, "fragment-cast-grid");
                return Ok(());
            };
            if grid.len() != nr * nc {
                self.bail(code, "fragment-cast-grid-size");
                return Ok(());
            }
            let out: Vec<pliron::value::Value> = if src_elem == dst_elem {
                grid
            } else {
                let have = LlType::vec(64, src_elem);
                let want = LlType::vec(64, dst_elem);
                grid.iter()
                    .map(|&f| self.fpcast_frag(ctx, rewriter, f, &have, &want))
                    .collect()
            };
            let Some(&rep) = out.first() else {
                self.bail(code, "fragment-cast-empty");
                return Ok(());
            };
            let result0 = op.deref(ctx).get_result(0);
            self.frags.grids.insert(result0, out.clone());
            self.frags.grids.insert(rep, out);
            rewriter.replace_operation_with_values(ctx, op, vec![rep]);
            self.converted += 1;
            return Ok(());
        }
        // Widths decide `ext` vs `trunc`, so both sides must be readable. The SOURCE comes
        // from the operand's type HISTORY, not its current type: by the time this arm runs the
        // operand has usually been converted already, and reading the air type off it makes
        // `wax_type_of` return nothing - which silently refused every cast in the corpus.
        let src_hist = operands_info
            .lookup_most_recent_type(src_v)
            .unwrap_or(src_w);
        let src_ll = wax_type_of(ctx, src_hist).as_ref().and_then(air_type_for);
        let dst_ll = wax_type_of(ctx, dst_w).as_ref().and_then(air_type_for);
        let (Some(src_ll), Some(dst_ll)) = (src_ll, dst_ll) else {
            self.bail(code, "cast-type-unmappable");
            return Ok(());
        };
        // The instruction is chosen from the actual TYPES, not from the wax opcode, which is
        // what `cast_op` does. Two cases the opcode alone gets wrong:
        //
        // - equal source and destination: no cast at all, forward the operand. LLVM rejects a
        //   same-width `ext`/`trunc` outright - `llvm-dis: error: Invalid cast` - so emitting
        //   one produces a metallib that is not merely different but unreadable.
        // - an integer widening or narrowing is decided by the widths, and the SIGN by the
        //   `signedness` attribute; 0 means unsigned, which matters for u8 masks and u32 ids
        //   whose high bit is set.
        if src_ll == dst_ll {
            let Some((ops, _)) = self.parts(ctx, op, 1) else {
                self.bail(code, "cast-parts");
                return Ok(());
            };
            rewriter.replace_operation_with_values(ctx, op, vec![ops[0]]);
            self.converted += 1;
            return Ok(());
        }
        let signed = self.wax_attr_i64(ctx, op, "signedness") != Some(0);
        let from_f = matches!(
            src_ll,
            LlType::Half | LlType::BFloat | LlType::Float | LlType::Double
        );
        let to_f = matches!(
            dst_ll,
            LlType::Half | LlType::BFloat | LlType::Float | LlType::Double
        );
        let kind = match (from_f, to_f) {
            (true, true) => {
                if scalar_bits(&src_ll) > scalar_bits(&dst_ll) {
                    CastOpAttr::FPTrunc
                } else {
                    CastOpAttr::FPExt
                }
            }
            (true, false) => CastOpAttr::FPToSI,
            (false, true) => CastOpAttr::SIToFP,
            (false, false) => {
                if scalar_bits(&src_ll) < scalar_bits(&dst_ll) {
                    if signed {
                        CastOpAttr::SExt
                    } else {
                        CastOpAttr::ZExt
                    }
                } else {
                    CastOpAttr::Trunc
                }
            }
        };
        let _ = code;
        let Some((ops, res_ty)) = self.parts(ctx, op, 1) else {
            self.bail(code, "cast-parts");
            return Ok(());
        };
        let c = air::CastOp::new(ctx, res_ty, ops[0]);
        c.set_op(ctx, kind);
        rewriter.insert_operation(ctx, c.get_operation());
        rewriter.replace_operation(ctx, op, c.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `iota`: the element's index WITHIN its tile, which is the same extent-derived lane
    /// term a load of that tile uses. The raw `%_tp.x` would run past a tile narrower than
    /// the threadgroup and silently corrupt tail masks.
    pub(crate) fn rewrite_iota(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        use pliron::r#type::Typed;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let n = wax_res
            .and_then(|t| wax_type_of(ctx, t))
            .as_ref()
            .and_then(crate::types::tile_elem_count)
            .unwrap_or(1);
        let Some(tp) = self.control.tp else {
            self.unconverted.note(Opcode::Iota);
            return Ok(());
        };
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let lane = if n <= 1 {
            let z = air::ConstBitsOp::new(ctx, i32_ty, 0);
            rewriter.insert_operation(ctx, z.get_operation());
            z.get_operation().deref(ctx).get_result(0)
        } else {
            let zero = air::ConstBitsOp::new(ctx, i32_ty, 0);
            rewriter.insert_operation(ctx, zero.get_operation());
            let zero = zero.get_operation().deref(ctx).get_result(0);
            let x = air::ExtractElemOp::new(ctx, i32_ty, tp, zero);
            rewriter.insert_operation(ctx, x.get_operation());
            let x = x.get_operation().deref(ctx).get_result(0);
            if n > 32 || !(n as u64).is_power_of_two() {
                x
            } else {
                let m = air::ConstBitsOp::new(ctx, i32_ty, (n - 1) as u64);
                rewriter.insert_operation(ctx, m.get_operation());
                let m = m.get_operation().deref(ctx).get_result(0);
                let a = air::BinOp::new(ctx, i32_ty, x, m);
                a.set_op(ctx, BinOpAttr::And);
                rewriter.insert_operation(ctx, a.get_operation());
                a.get_operation().deref(ctx).get_result(0)
            }
        };
        rewriter.replace_operation_with_values(ctx, op, vec![lane]);
        self.converted += 1;
        Ok(())
    }
    /// Float comparison. The `comparison_predicate` attribute is an index into the ordered
    /// predicate table (0 `oeq`, 1 `one`, 2 `olt`, 3 `ole`, 4 `ogt`, 5 `oge`); the frontend
    /// always sets it, so the `oeq` default for an absent attribute is defensive only. One
    /// deliberate disagreement with the CFG path: an out-of-range index falls back to `oeq`
    /// there, but the conversion refuses it - the frontend never emits one, and a silently
    /// wrong predicate is worse than a refusal.
    pub(crate) fn rewrite_cmp_float(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::FCmpPredAttr;
        let attr = match self
            .wax_attr_i64(ctx, op, "comparison_predicate")
            .unwrap_or(0)
        {
            0 => FCmpPredAttr::Oeq,
            1 => FCmpPredAttr::One,
            2 => FCmpPredAttr::Olt,
            3 => FCmpPredAttr::Ole,
            4 => FCmpPredAttr::Ogt,
            5 => FCmpPredAttr::Oge,
            _ => {
                self.unconverted.note(Opcode::CmpF);
                return Ok(());
            }
        };
        let Some((ops, res_ty)) = self.parts(ctx, op, 2) else {
            self.unconverted.note(Opcode::CmpF);
            return Ok(());
        };
        let c = air::FCmpOp::new(ctx, res_ty, ops[0], ops[1]);
        c.set_pred(ctx, attr);
        rewriter.insert_operation(ctx, c.get_operation());
        rewriter.replace_operation(ctx, op, c.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `min`/`max` spelled as compare-and-select, which is what the CFG path emits for the
    /// integer pair and for float `min`. Float `max` is the one that takes an intrinsic.
    pub(crate) fn rewrite_minmax_select(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        use crate::air_dialect::attrs::{FCmpPredAttr, ICmpPredAttr};
        let Some((ops, res_ty)) = self.parts(ctx, op, 2) else {
            self.unconverted.note(code);
            return Ok(());
        };
        let i1 = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let cmp = match code {
            Opcode::MinF => {
                let c = air::FCmpOp::new(ctx, i1, ops[0], ops[1]);
                c.set_pred(ctx, FCmpPredAttr::Olt);
                c.get_operation()
            }
            Opcode::MaxI | Opcode::MinI => {
                let c = air::ICmpOp::new(ctx, i1, ops[0], ops[1]);
                c.set_pred(
                    ctx,
                    if code == Opcode::MaxI {
                        ICmpPredAttr::Sgt
                    } else {
                        ICmpPredAttr::Slt
                    },
                );
                c.get_operation()
            }
            _ => {
                self.unconverted.note(code);
                return Ok(());
            }
        };
        rewriter.insert_operation(ctx, cmp);
        let cond = cmp.deref(ctx).get_result(0);
        let s = air::SelectOp::new(ctx, res_ty, cond, ops[0], ops[1]);
        rewriter.insert_operation(ctx, s.get_operation());
        rewriter.replace_operation(ctx, op, s.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `fma(a, b, c)` as a separate multiply and add, matching the CFG path. Not an
    /// `@llvm.fma` call: the backend leaves the contraction to the AIR compiler.
    pub(crate) fn rewrite_fma(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        let Some((ops, res_ty)) = self.parts(ctx, op, 3) else {
            self.unconverted.note(Opcode::Fma);
            return Ok(());
        };
        let m = air::BinOp::new(ctx, res_ty, ops[0], ops[1]);
        m.set_op(ctx, BinOpAttr::FMul);
        rewriter.insert_operation(ctx, m.get_operation());
        let mv = m.get_operation().deref(ctx).get_result(0);
        let a = air::BinOp::new(ctx, res_ty, mv, ops[2]);
        a.set_op(ctx, BinOpAttr::FAdd);
        rewriter.insert_operation(ctx, a.get_operation());
        rewriter.replace_operation(ctx, op, a.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// Integer negation, which the CFG path spells `0 - x`.
    pub(crate) fn rewrite_neg_int(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        let Some((ops, res_ty)) = self.parts(ctx, op, 1) else {
            self.unconverted.note(Opcode::NegI);
            return Ok(());
        };
        let z = air::ConstBitsOp::new(ctx, res_ty, 0);
        rewriter.insert_operation(ctx, z.get_operation());
        let zv = z.get_operation().deref(ctx).get_result(0);
        let s = air::BinOp::new(ctx, res_ty, zv, ops[0]);
        s.set_op(ctx, BinOpAttr::Sub);
        rewriter.insert_operation(ctx, s.get_operation());
        rewriter.replace_operation(ctx, op, s.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// `rsqrt` as `1 / sqrt(x)`. `air.rsqrt` exists but only in f32; one shape covers all
    /// three element types, which is the trade the CFG path also makes.
    pub(crate) fn rewrite_rsqrt(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        use pliron::r#type::Typed;
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() == 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            self.unconverted.note(Opcode::Rsqrt);
            return Ok(());
        };
        let Some((cty, suf, rty)) = self.intrin_ty(ctx, wax_res) else {
            self.unconverted.note(Opcode::Rsqrt);
            return Ok(());
        };
        let Some(&a) = op.deref(ctx).operands().collect::<Vec<_>>().first() else {
            self.unconverted.note(Opcode::Rsqrt);
            return Ok(());
        };
        let cty_h = crate::air_dialect::types::to_pliron(ctx, &cty);
        let a = self.fpcast_if(ctx, rewriter, a, &rty, &cty);
        let sq = air::CallOp::new(ctx, &format!("@air.sqrt.{suf}"), Some(cty_h), vec![a]);
        rewriter.insert_operation(ctx, sq.get_operation());
        let sqv = sq.get_operation().deref(ctx).get_result(0);
        let one = air::ConstBitsOp::new(ctx, cty_h, float_one_bits(&cty));
        rewriter.insert_operation(ctx, one.get_operation());
        let onev = one.get_operation().deref(ctx).get_result(0);
        let d = air::BinOp::new(ctx, cty_h, onev, sqv);
        d.set_op(ctx, BinOpAttr::FDiv);
        rewriter.insert_operation(ctx, d.get_operation());
        let r = d.get_operation().deref(ctx).get_result(0);
        let r = self.fpcast_if(ctx, rewriter, r, &cty, &rty);
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// The pointer ops behind a gather: `int_to_ptr`, `addptr`, `load_ptr`.
    ///
    /// One address per lane, so no affine lane term - the thread-per-element model. The
    /// element type comes from the RESULT, which is where the pointee is named.
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
            self.bail(code, "no result");
            return Ok(());
        };
        let Some(wax_res) = wax_type_of(ctx, first_res) else {
            self.bail(code, "result type unreadable");
            return Ok(());
        };
        let Some(elem) = crate::types::mem_elem_lltype(&crate::types::param_elem(&wax_res)) else {
            self.bail(code, "element type is not stored verbatim");
            return Ok(());
        };
        let operands: Vec<_> = op.deref(ctx).operands().collect();

        let mut replacements: Vec<pliron::value::Value> = Vec::new();
        match code {
            Opcode::IntToPtr => {
                let Some(&src) = operands.first() else {
                    self.bail(code, "no source");
                    return Ok(());
                };
                let ty = crate::air_dialect::types::to_pliron(ctx, &LlType::device_ptr(elem));
                let c = air::CastOp::new(ctx, ty, src);
                c.set_op(ctx, CastOpAttr::IntToPtr);
                rewriter.insert_operation(ctx, c.get_operation());
                replacements.push(c.get_operation().deref(ctx).get_result(0));
            }
            Opcode::Offset => {
                let (Some(&base), Some(&off)) = (operands.first(), operands.get(1)) else {
                    self.bail(code, "offset needs a base and an index");
                    return Ok(());
                };
                let ty =
                    crate::air_dialect::types::to_pliron(ctx, &LlType::device_ptr(elem.clone()));
                let elem_h = crate::air_dialect::types::to_pliron(ctx, &elem);
                let g = air::GepOp::new(ctx, ty, base, off);
                air::set_elem_ty(g.get_operation(), ctx, elem_h);
                rewriter.insert_operation(ctx, g.get_operation());
                replacements.push(g.get_operation().deref(ctx).get_result(0));
            }
            Opcode::LoadPtrTko => {
                let Some(&ptr) = operands.first() else {
                    self.bail(code, "no pointer");
                    return Ok(());
                };
                let ty = crate::air_dialect::types::to_pliron(ctx, &elem);
                let ld = air::LoadOp::new(ctx, ty, ptr);
                air::set_align(ld.get_operation(), ctx, align_bytes(&elem));
                rewriter.insert_operation(ctx, ld.get_operation());
                replacements.push(ld.get_operation().deref(ctx).get_result(0));
            }
            _ => {
                self.bail(code, "not a pointer op");
                return Ok(());
            }
        }
        // Remaining results are ordering tokens, which AGX has nothing to order against.
        self.finish_replacement(ctx, rewriter, op, &res_tys, replacements);
        Ok(())
    }
}
