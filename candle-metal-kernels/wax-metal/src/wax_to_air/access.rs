//! Per-access lowering.
//!
//! load/store arms, element addressing, and access bounds.

use super::*;

/// The pointee of an air pointer type.
/// The address space of a pointer type, for rebuilding it around a different element.
pub(crate) fn ptr_addrspace(ctx: &mut Context, ptr: TypeHandle) -> Option<u32> {
    match crate::air_dialect::types::from_pliron(ctx, ptr)? {
        LlType::Ptr { addrspace, .. } => Some(addrspace),
        _ => None,
    }
}

pub(crate) fn element_of(ctx: &mut Context, ptr: TypeHandle) -> Option<TypeHandle> {
    let ll = crate::air_dialect::types::from_pliron(ctx, ptr)?;
    let LlType::Ptr { pointee, .. } = ll else {
        return None;
    };
    Some(crate::air_dialect::types::to_pliron(ctx, &pointee))
}

/// Bit width of a scalar air type.
pub(crate) fn scalar_bits(t: &LlType) -> u32 {
    match t {
        LlType::I1 => 1,
        LlType::I8 => 8,
        LlType::I16 | LlType::Half | LlType::BFloat => 16,
        LlType::I32 | LlType::Float => 32,
        LlType::I64 | LlType::Double => 64,
        _ => 0,
    }
}

/// Insert an immediate i32 value (`imm`) into the rewriter, returns the pliron representation of said value.
fn rw_imm(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    i32_ty: TypeHandle,
    v: i64,
) -> pliron::value::Value {
    let c = air::ConstBitsOp::new(ctx, i32_ty, v as u64);
    rw.insert_operation(ctx, c.get_operation());
    c.get_operation().deref(ctx).get_result(0)
}

/// This lane's per-dim tile coordinates, from its flat row-major index. `f` runs once per
/// dim with extent greater than 1.
///
/// The index is decomposed so each dim pays its own stride. Scaling it by the innermost
/// stride alone is only right when every outer extent is 1.
///
/// `tp` holds the flat index. It is masked only for a tile that one simdgroup wraps, and
/// is otherwise the element index directly.
///
/// `elem_offset` folds the coordinates into an offset. `elem_coords` keeps them for bounds
/// checks. They differ only in `f`.
fn lane_decompose(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    i32_ty: TypeHandle,
    tile_shape: &[i64],
    rank: usize,
    tp: pliron::value::Value,
    mut f: impl FnMut(
        usize,
        pliron::value::Value,
        &mut Context,
        &mut DialectConversionRewriter,
    ) -> Option<()>,
) -> Option<()> {
    let n_elems: i64 = tile_shape.iter().product();
    let zero = rw_imm(ctx, rw, i32_ty, 0);
    let x = air::ExtractElemOp::new(ctx, i32_ty, tp, zero);
    rw.insert_operation(ctx, x.get_operation());
    let x = x.get_operation().deref(ctx).get_result(0);
    let mut running = if n_elems > 32 || !(n_elems as u64).is_power_of_two() {
        x
    } else {
        let m = rw_imm(ctx, rw, i32_ty, n_elems - 1);
        air_bin_i32(
            ctx,
            rw,
            i32_ty,
            x,
            m,
            crate::air_dialect::attrs::BinOpAttr::And,
        )
    };
    let mut remaining: i64 = (0..rank)
        .map(|d| *tile_shape.get(d).unwrap_or(&1))
        .product();
    for d in (0..rank).rev() {
        let extent = *tile_shape.get(d).unwrap_or(&1);
        if extent <= 1 {
            continue; // c_d is always 0
        }
        // The modulo is redundant once nothing outer is left to strip.
        let coord = if remaining == extent {
            running
        } else {
            let e = rw_imm(ctx, rw, i32_ty, extent);
            air_bin_i32(
                ctx,
                rw,
                i32_ty,
                running,
                e,
                crate::air_dialect::attrs::BinOpAttr::URem,
            )
        };
        f(d, coord, ctx, rw)?;
        remaining /= extent;
        if remaining > 1 {
            let e = rw_imm(ctx, rw, i32_ty, extent);
            running = air_bin_i32(
                ctx,
                rw,
                i32_ty,
                running,
                e,
                crate::air_dialect::attrs::BinOpAttr::UDiv,
            );
        }
    }
    Some(())
}

/// Bit pattern of `1.0` at an air float type, for the `1 / sqrt(x)` spelling.
pub(crate) fn float_one_bits(t: &LlType) -> u64 {
    match t {
        LlType::Half => 0x3C00,
        LlType::BFloat => 0x3F80,
        LlType::Double => 0x3FF0_0000_0000_0000,
        _ => f32::to_bits(1.0) as u64,
    }
}

/// A per-dim bound to check an access against.
#[derive(Clone, Copy, Debug)]
pub enum Bound {
    /// A runtime extent, already promoted to a parameter and loaded.
    Dyn(pliron::value::Value),
    /// A static extent that tiles raggedly.
    Fixed(i64),
}

/// Per-access bounds, collected before conversion begins.
///
/// `fold_view` replaces a view with the pointer it was built from, so by the time an access
/// is rewritten the `make_tensor_view` carrying the runtime extents is gone. Only the view's
/// type survives, on the operand. Keyed by the access operation, stable until it is rewritten.
///
/// A dim needs a bound when its extent is dynamic and promoted, or when a static extent
/// tiles raggedly. A dynamic extent that was not promoted yields none, as
/// `extent_value_for_view` refuses one. An unpromoted parameter reads as constant 0, and a
/// bound of 0 masks away every access.
pub(crate) fn collect_access_bounds(
    ctx: &Context,
    func: Ptr<Operation>,
    promoted: &[pliron::value::Value],
) -> std::collections::HashMap<Ptr<Operation>, Vec<(usize, Bound)>> {
    use pliron::r#type::Typed;
    let mut accesses: Vec<(Ptr<Operation>, usize)> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            match try_opcode_of(ctx, op) {
                Some(Opcode::LoadViewTko) => accesses.push((op, 0)),
                Some(Opcode::StoreViewTko) => accesses.push((op, 1)),
                _ => {}
            }
        }
    });

    let mut out = std::collections::HashMap::new();
    for (op, view_at) in accesses {
        let Some(view) = op.deref(ctx).operands().nth(view_at) else {
            continue;
        };
        let Some(wax_ir::types::Type::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx))
        else {
            continue;
        };
        if pv.padding_value.is_none() {
            continue;
        }
        let shape = &pv.tensor_view.shape;
        let mut bounds: Vec<(usize, Bound)> = Vec::new();
        for d in 0..pv.tile_shape.len() {
            let Some(&ext) = shape.get(d) else { continue };
            if ext == wax_ir::DYNAMIC {
                // Note: only a promoted extent yields a bound. `extent_value_for_view` refuses an
                // unpromoted one for two reasons.
                // 1. It is constant 0. Bound of 0 masks away every access.
                // 2. It belongs a view erased by `fold_view`, causing a dangling ref.
                match extent_value_for_view(ctx, view, shape, d) {
                    Some(v) if promoted.contains(&v) => bounds.push((d, Bound::Dyn(v))),
                    Some(_) => why("bound: extent found but not promoted"),
                    None => why("bound: extent value not traceable"),
                }
                continue;
            }
            let t = i64::from(*pv.tile_shape.get(d).unwrap_or(&1));

            // Note: no `ext > t` guard. `ext` smaller than tile `t` implies ragged An extent smaller than the tile is the ragged case that
            // needs the mask most. At `M=7` with `BM=32` every block reads rows 7..31 past the
            // end of the tensor. `ext % t != 0` already covers it. `ext < t` gives
            // `ext % t == ext`. An extent equal to, or an exact multiple of, the tile yields
            // no bound.
            if ext > 0 && t > 0 && ext % t != 0 {
                bounds.push((d, Bound::Fixed(ext)));
            }
        }
        if !bounds.is_empty() {
            out.insert(op, bounds);
        }
    }
    out
}

/// The SSA value of dim `d`'s runtime extent, if it was promoted to a parameter.
///
/// Walks the view chain through strided-view / `assume` wrapper, partition, tensor view,
/// `dyn_shape` operand group indexed by how many dynamic dims
/// precede this one.
pub(crate) fn extent_value_for_view(
    ctx: &Context,
    view: pliron::value::Value,
    shape: &[i64],
    d: usize,
) -> Option<pliron::value::Value> {
    use pliron::value::DefiningEntity;

    // Ensure shape dim is dynamic
    if shape.get(d).copied() != Some(wax_ir::DYNAMIC) {
        return None;
    }

    let DefiningEntity::Op(mut cur) = view.defining_entity() else {
        return None;
    };
    // Unnest actual operand from strided view / assume wrappers
    for _ in 0..4 {
        match try_opcode_of(ctx, cur) {
            Some(Opcode::MakeStridedView) | Some(Opcode::Assume) => {
                let next = cur.deref(ctx).operands().next()?;
                let DefiningEntity::Op(n) = next.defining_entity() else {
                    return None;
                };
                cur = n;
            }
            _ => break,
        }
    }

    // Get partition
    if try_opcode_of(ctx, cur) != Some(Opcode::MakePartitionView) {
        return None;
    }
    let tv = cur.deref(ctx).operands().next()?;
    let DefiningEntity::Op(tv_op) = tv.defining_entity() else {
        return None;
    };

    // Get tensor view
    if try_opcode_of(ctx, tv_op) != Some(Opcode::MakeTensorView) {
        return None;
    }

    // Get dyn shape index
    let dyn_idx = shape[..=d]
        .iter()
        .filter(|&&e| e == wax_ir::DYNAMIC)
        .count()
        - 1;
    let (n_base, n_dyn_shape, _) = wax_operand_segments(ctx, tv_op);
    if dyn_idx >= n_dyn_shape {
        return None;
    }

    // Get value of `d`
    let operands: Vec<_> = tv_op.deref(ctx).operands().collect();
    operands
        .get(n_base + dyn_idx)
        .copied()
        .map(|v| trace_through_assume(ctx, v))
}

/// Bit pattern of a view's padding value at an element type.
///
/// The padding constant per element type. The constant is an `air.const_bits`, so only the
/// bits are needed. Integer tiles have only zero.
pub(crate) fn padding_bits(elem: &LlType, pad: wax_ir::PaddingValue) -> Option<u64> {
    use wax_ir::PaddingValue as P;
    Some(match (elem, pad) {
        (LlType::Float, P::Zero) => 0,
        (LlType::Float, P::NegZero) => 0x8000_0000,
        (LlType::Float, P::Nan) => 0x7FC0_0000,
        (LlType::Float, P::PosInf) => 0x7F80_0000,
        (LlType::Float, P::NegInf) => 0xFF80_0000,
        (LlType::Half, P::Zero) => 0,
        (LlType::Half, P::NegZero) => 0x8000,
        (LlType::Half, P::Nan) => 0x7E00,
        (LlType::Half, P::PosInf) => 0x7C00,
        (LlType::Half, P::NegInf) => 0xFC00,
        (LlType::BFloat, P::Zero) => 0,
        (LlType::BFloat, P::NegZero) => 0x8000,
        (LlType::BFloat, P::Nan) => 0x7FC0,
        (LlType::BFloat, P::PosInf) => 0x7F80,
        (LlType::BFloat, P::NegInf) => 0xFF80,
        (LlType::I32, P::Zero) => 0,
        _ => return None,
    })
}

/// Natural alignment of an air scalar, in bytes.
pub(crate) fn align_bytes(t: &LlType) -> u32 {
    (scalar_bits(t) / 8).max(1)
}

impl WaxToAir {
    pub(crate) fn rewrite_return(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let ret = air::RetOp::new(ctx);
        rewriter.insert_operation(ctx, ret.get_operation());
        rewriter.replace_operation(ctx, op, ret.get_operation());
        self.converted += 1;
        Ok(())
    }
    /// Replace the result with the operand it forwards: `assume`, the view constructors,
    /// `broadcast` and `reshape`. Correct exactly when the source element at lane `tp`,
    /// `tp & (n_src - 1)`, is the element the destination wants at `tp`; `wax.broadcast`
    /// forwards only when that actually holds.
    pub(crate) fn rewrite_broadcast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let shapes = (|| {
            use pliron::r#type::Typed;
            let base = op.deref(ctx).operands().next()?;
            let src = match self.wax_operand_ty(ctx, operands_info, base)? {
                WaxType::Tile(t) => t.shape.clone(),
                _ => return None,
            };
            let dst = match wax_type_of(ctx, op.deref(ctx).get_result(0).get_type(ctx))? {
                WaxType::Tile(t) => t.shape.clone(),
                _ => return None,
            };
            Some((src, dst))
        })();
        let Some((src, dst)) = shapes else {
            self.bail(
                Opcode::Broadcast,
                "broadcast operand or result is not a tile",
            );
            return Ok(());
        };
        let replicated = op
            .deref(ctx)
            .operands()
            .next()
            .is_some_and(|v| self.reduce.row_replicated.contains(&v));
        if !broadcast_is_forwardable(&src, &dst)
            && !(replicated && expands_replicated_row(&src, &dst))
        {
            return self.redistribute_broadcast(ctx, rewriter, op, &src, &dst);
        }
        self.fold_view(ctx, rewriter, op)
    }
    /// No lane holds the element it wants, so route the tile through threadgroup scratch:
    /// every lane writes the element it holds, then reads the one it wants.
    fn redistribute_broadcast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        src: &[i64],
        dst: &[i64],
    ) -> Result<()> {
        use crate::air_dialect::attrs::{BinOpAttr, CastOpAttr, ICmpPredAttr};
        use pliron::r#type::Typed;
        if broadcast_src_flat(src, dst, 0).is_none() {
            self.bail(Opcode::Broadcast, "shapes are not in a broadcast relation");
            return Ok(());
        }
        let Some(value) = op.deref(ctx).operands().next() else {
            self.bail(Opcode::Broadcast, "no source operand");
            return Ok(());
        };
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Broadcast, "no thread position parameter");
            return Ok(());
        };
        // The scratch is float memory. Every element a broadcast carries here is at most 32
        // bits and round-trips through it exactly. f32 as itself, i32 by bitcast, a half by
        // widening. The same convention the column reduce uses.
        let Some(elem_ll) = crate::air_dialect::types::from_pliron(ctx, value.get_type(ctx)) else {
            self.bail(Opcode::Broadcast, "source element has no air type");
            return Ok(());
        };
        let casts: Option<(CastOpAttr, CastOpAttr)> = match elem_ll {
            LlType::Float => None,
            LlType::I32 => Some((CastOpAttr::Bitcast, CastOpAttr::Bitcast)),
            LlType::Half | LlType::BFloat => Some((CastOpAttr::FPExt, CastOpAttr::FPTrunc)),
            _ => {
                self.bail(
                    Opcode::Broadcast,
                    "source element does not ride in the scratch",
                );
                return Ok(());
            }
        };
        // Right-align the shapes, the way `broadcast_src_flat` reads them.
        let src: Vec<i64> = if src.len() < dst.len() {
            std::iter::repeat_n(1, dst.len() - src.len())
                .chain(src.iter().copied())
                .collect()
        } else {
            src.to_vec()
        };
        let (n_src, n_dst): (i64, i64) = (src.iter().product(), dst.iter().product());
        // `+ 1` for the dump slot. Sized again before conversion by `tg_scratch_floats_of`,
        // and the two must agree or the gep's array type disagrees with the global's.
        if n_src + 1 > self.reduce.tg_scratch_floats as i64 {
            self.bail(Opcode::Broadcast, "source tile does not fit the scratch");
            return Ok(());
        }
        // Emits the WAR barrier and takes the scratch reference.
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let f32_h = rt.f32_h;
        let zero = rt.const_i(ctx, 0);
        let tid = {
            let e = air::ExtractElemOp::new(ctx, i32_h, tp, zero);
            rewriter.insert_operation(ctx, e.get_operation());
            e.get_operation().deref(ctx).get_result(0)
        };
        // A lane past the source tile holds no element of it and must not write over one
        // that exists, so it writes the dump slot. Selecting the slot keeps the store
        // unconditional, which keeps it uniform across the threadgroup.
        let store_idx = {
            let n_c = rt.const_i(ctx, n_src);
            let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
            let c = air::ICmpOp::new(ctx, i1_h, tid, n_c);
            c.set_pred(ctx, ICmpPredAttr::Ult);
            rewriter.insert_operation(ctx, c.get_operation());
            let keep = c.get_operation().deref(ctx).get_result(0);
            let s = air::SelectOp::new(ctx, i32_h, keep, tid, n_c);
            rewriter.insert_operation(ctx, s.get_operation());
            s.get_operation().deref(ctx).get_result(0)
        };
        let word = match casts {
            None => value,
            Some((to, _)) => {
                let c = air::CastOp::new(ctx, f32_h, value);
                c.set_op(ctx, to);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            }
        };
        let p = rt.scratch_gep(ctx, rewriter, store_idx);
        let st = air::StoreOp::new(ctx, word, p);
        rewriter.insert_operation(ctx, st.get_operation());
        rt.barrier(ctx, rewriter);

        // `src_flat = sum over kept dims of ((tid / inner_d) % dst[d]) * inner_s`. An expanded
        // dim contributes coordinate 0. Every coordinate is taken modulo its own extent, so
        // the index is in range by construction. That includes lanes past the destination
        // tile, so the index needs no clamp.
        let (mut inner_d, mut inner_s) = (n_dst, n_src);
        let mut acc: Option<pliron::value::Value> = None;
        for d in 0..dst.len() {
            inner_d /= dst[d];
            inner_s /= src[d];
            if src[d] != dst[d] {
                continue;
            }
            let mut c = tid;
            if inner_d > 1 {
                let k = rt.const_i(ctx, inner_d);
                c = rt.bin(ctx, rewriter, c, k, BinOpAttr::UDiv);
            }
            if inner_d * dst[d] != n_dst {
                let k = rt.const_i(ctx, dst[d]);
                c = rt.bin(ctx, rewriter, c, k, BinOpAttr::URem);
            }
            if inner_s > 1 {
                let k = rt.const_i(ctx, inner_s);
                c = rt.bin(ctx, rewriter, c, k, BinOpAttr::Mul);
            }
            acc = Some(match acc {
                None => c,
                Some(a) => rt.bin(ctx, rewriter, a, c, BinOpAttr::Add),
            });
        }
        let load_idx = acc.unwrap_or(zero);
        let p = rt.scratch_gep(ctx, rewriter, load_idx);
        let ld = air::LoadOp::new(ctx, f32_h, p);
        rewriter.insert_operation(ctx, ld.get_operation());
        let got = ld.get_operation().deref(ctx).get_result(0);
        let out = match casts {
            None => got,
            Some((_, back)) => {
                let elem_h = crate::air_dialect::types::to_pliron(ctx, &elem_ll);
                let c = air::CastOp::new(ctx, elem_h, got);
                c.set_op(ctx, back);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            }
        };
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    pub(crate) fn fold_view(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let base = op.deref(ctx).operands().next();
        let n_results = op.deref(ctx).get_num_results();
        let Some(base) = base else {
            self.unconverted.note(opcode_of(ctx, op));
            return Ok(());
        };
        if n_results != 1 {
            self.unconverted.note(opcode_of(ctx, op));
            return Ok(());
        }
        rewriter.replace_operation_with_values(ctx, op, vec![base]);
        self.converted += 1;
        Ok(())
    }
    /// `wax.load_view` / `wax.store_view` -> address arithmetic + `air.gep` + `air.load`/
    /// `store`. One arm, because the two differ only in operand order and which op ends the
    /// sequence. A store leads with its value where a load leads with its view, and a store
    /// mixed up would address through its own value.
    ///
    /// The view's type comes from `operands_info`. By the time this runs `fold_view` has
    /// replaced the view with its pointer, so the live type is a pointer with the shape and
    /// strides gone.
    pub(crate) fn rewrite_access(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
        is_store: bool,
    ) -> Result<()> {
        use pliron::r#type::Typed;
        let (view_at, idx_at) = if is_store { (1, 2) } else { (0, 1) };
        let code = if is_store {
            Opcode::StoreViewTko
        } else {
            Opcode::LoadViewTko
        };

        let (operands, res_tys) = {
            let o = op.deref(ctx);
            let operands: Vec<_> = o.operands().collect();
            let res_tys = op_result_types(ctx, op);
            (operands, res_tys)
        };
        let Some(&view) = operands.get(view_at) else {
            self.bail(code, "no-view-operand");
            return Ok(());
        };
        // The view type as it was before folding.
        let view_ty = operands_info
            .lookup_most_recent_type(view)
            .or_else(|| Some(view.get_type(ctx)))
            .and_then(|h| wax_ir::dialect::types::from_pliron(ctx, h));
        let Some(view_ty) = view_ty else {
            self.bail(code, "view-type-unreadable");
            return Ok(());
        };

        // Only the first `rank` operands after the view are indices. A `*_tko` access also
        // carries mask, padding and token operands, and taking those as indices walks off the
        // end of the stride list.
        // The element width is also the natural alignment to emit. Without it the access would
        // carry no alignment, and the bitcode writer's log2 encoding turns a 0 into
        // `align 4294967296`.
        let pad_value = match &view_ty {
            wax_ir::types::Type::PartitionView(pv) => pv.padding_value.and_then(|p| {
                let elem = crate::types::float_elem(&pv.tensor_view.element_type)
                    .map(|(t, _)| t)
                    .unwrap_or(LlType::I32);
                padding_bits(&elem, p)
            }),
            _ => None,
        };
        // The element the access steps by, from the view. See the gep below.
        let velem = match &view_ty {
            wax_ir::types::Type::PartitionView(pv) => {
                match crate::types::mem_elem_lltype(&pv.tensor_view.element_type) {
                    Some(t) => t,
                    None => {
                        self.bail(code, "view-element-type");
                        return Ok(());
                    }
                }
            }
            _ => {
                self.bail(code, "view-not-partition");
                return Ok(());
            }
        };
        let (rank, align) = match &view_ty {
            wax_ir::types::Type::PartitionView(pv) => (
                pv.tile_shape.len(),
                crate::types::scalar_byte_width(&pv.tensor_view.element_type),
            ),
            _ => {
                self.bail(code, "view-not-partition");
                return Ok(());
            }
        };
        let index_ops: Vec<_> = operands.iter().skip(idx_at).take(rank).copied().collect();
        if index_ops.len() != rank {
            self.bail(code, "index-count");
            return Ok(());
        }

        // A fragment tile is loaded by the simdgroup-matrix intrinsic, not as a scalar: the
        // analysis says whether this result lives in fragment registers, and the two
        // lowerings are not interchangeable.
        let res0 = {
            let o = op.deref(ctx);
            (o.get_num_results() > 0).then(|| o.get_result(0))
        };
        let frag = res0.and_then(|v| self.frag_dims(ctx, operands_info, v));
        if !is_store && let Some((nr, nc)) = frag {
            // A planned bounds list means this load is masked: the emask/clamp
            // spelling addresses in absolute coordinates and cannot reuse the base-GEP
            // geometry of the unmasked load.
            let masked = self.addr.bounds.get(&op).is_some_and(|b| !b.is_empty());
            return if masked {
                self.rewrite_fragment_load_masked(
                    ctx,
                    rewriter,
                    op,
                    FragAccess {
                        kind: "load",
                        view,
                        view_ty: &view_ty,
                        index_ops: &index_ops,
                    },
                    nr,
                    nc,
                )
            } else {
                self.rewrite_fragment_load(
                    ctx,
                    rewriter,
                    op,
                    FragAccess {
                        kind: "load",
                        view,
                        view_ty: &view_ty,
                        index_ops: &index_ops,
                    },
                    nr,
                    nc,
                )
            };
        }
        // A store is classified by its value, as `store_view` does: a fragment in the value
        // operand takes the simdgroup-matrix store, whatever the result says. The value is
        // usually a grid representative by now, so the grids table decides, not just the
        // fragment set.
        if is_store {
            let value = *operands.first().unwrap();
            if let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, value) {
                // A planned bounds list means this store is masked: the emask
                // spelling addresses in absolute coordinates and cannot reuse the
                // base-GEP geometry of the unmasked store.
                let masked = self.addr.bounds.get(&op).is_some_and(|b| !b.is_empty());
                return if masked {
                    self.rewrite_fragment_store_masked(
                        ctx,
                        rewriter,
                        op,
                        value,
                        FragAccess {
                            kind: "store",
                            view,
                            view_ty: &view_ty,
                            index_ops: &index_ops,
                        },
                        nr,
                        nc,
                    )
                } else {
                    self.rewrite_fragment_store(
                        ctx,
                        rewriter,
                        op,
                        value,
                        FragAccess {
                            kind: "store",
                            view,
                            view_ty: &view_ty,
                            index_ops: &index_ops,
                        },
                        nr,
                        nc,
                    )
                };
            }
        }

        // Lane-vector fusion: W contiguous per-lane loads become one wide transaction,
        // answered per member by an extract. The run was planned before conversion
        // (`collect_lane_runs`). The emission lands at the head's position whatever member the
        // driver reaches first, so the order matches the CFG.
        if !is_store && self.lanes.lane_runs.contains_key(&op) {
            return self.emit_lane_vector_member(ctx, rewriter, op);
        }
        // The store mirror of the same fusion: W stores over one per-lane run become one wide
        // store, emitted at the last member. That is the first point where every stored value
        // is guaranteed lowered.
        if is_store && self.lanes.lane_runs.contains_key(&op) {
            return self.emit_lane_vector_store_member(ctx, rewriter, op);
        }

        // Cross-loop staging: the consuming loop's member is served from the cache the
        // staging loop wrote. No global address, no bounds, no padding. A staging-loop member
        // runs the normal path and then writes its slot.
        let cache_member = (!is_store)
            .then(|| self.staging.cache_members.get(&op))
            .flatten()
            .copied();
        if let Some((pi, false)) = cache_member {
            return self.rewrite_cached_load(ctx, rewriter, op, pi, &res_tys);
        }

        // The mask, if this access needs one. Computed before the address, because
        // `bounds_predicate` emits the lane decomposition first.
        let pred = self.bounds_predicate(ctx, rewriter, op, &view_ty, &index_ops);

        // A masked store is guarded by a branch, not a clamp. A clamped address would write to
        // element 0 instead of skipping the write. The address arithmetic therefore belongs
        // inside the guarded block, as `store_view` puts it. So the blocks are built before the
        // offset is computed, and the insertion point moved into the arm.
        let store_guard = match (is_store, pred) {
            (true, Some(p)) => {
                use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
                use pliron::irbuild::rewriter::Rewriter;
                use pliron::linked_list::LinkedList as _;
                let Some(head) = op.deref(ctx).get_container() else {
                    self.bail(code, "store is not inside a block");
                    return Ok(());
                };
                let Some(region) = head.deref(ctx).get_container() else {
                    self.bail(code, "block is not inside a region");
                    return Ok(());
                };
                let skip =
                    rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
                // `_pad{n}_store` is created before `_pad{n}_skip` and both are appended, so
                // the region order is do-block then skip-block.
                let do_b = rewriter.create_block(
                    ctx,
                    BlockInsertionPoint::AtRegionEnd(region),
                    None,
                    vec![],
                );
                rewriter.move_block(ctx, skip, BlockInsertionPoint::AtRegionEnd(region));
                let cbr = air::CondBrOp::new(ctx, p, do_b, vec![], skip, vec![]);
                cbr.get_operation().insert_at_back(head, ctx);
                let saved = rewriter.get_insertion_point();
                rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(do_b));
                Some((do_b, skip, saved))
            }
            _ => None,
        };

        let Some(off) = self.elem_offset(ctx, rewriter, &view_ty, &index_ops) else {
            self.bail(code, "address-unsupported");
            return Ok(());
        };

        // An out-of-range lane must not form the address at all, or the load faults. Clamp it
        // to element 0. The value it reads is replaced with the padding below.
        let off = match pred {
            Some(p) if !is_store => {
                let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
                let z = air::ConstBitsOp::new(ctx, i32_ty, 0);
                rewriter.insert_operation(ctx, z.get_operation());
                let zv = z.get_operation().deref(ctx).get_result(0);
                let sel = air::SelectOp::new(ctx, i32_ty, p, off, zv);
                rewriter.insert_operation(ctx, sel.get_operation());
                sel.get_operation().deref(ctx).get_result(0)
            }
            _ => off,
        };

        // THE VIEW DECIDES THE ELEMENT, NOT THE POINTER. `fold_view` has replaced the view
        // with the base pointer it was built on, so the obvious thing here is to step by that
        // pointer's pointee - which is what this did until 2026-10-09. Upstream cuda-tile is
        // explicit that it is the other way round: a `tensor_view` carries an `elementType`
        // and its strides count elements OF THAT TYPE (`Types.td`), with the pointer only a
        // base address. The two agree for every view built on a matching pointer, and differ
        // exactly where a pass has REVIEWED a buffer as another type - `split_k` viewing an
        // f16 destination as the f32 plane scratch, say - which used to force that pass to
        // retype the parameter so the pointee would agree.
        let base_ptr_ty = view.get_type(ctx);
        let Some(addrspace) = ptr_addrspace(ctx, base_ptr_ty) else {
            self.bail(code, "element-type");
            return Ok(());
        };
        let want = LlType::Ptr {
            pointee: Box::new(velem.clone()),
            addrspace,
        };
        let ptr_ty = crate::air_dialect::types::to_pliron(ctx, &want);
        // Only when they differ, so an ordinary access emits exactly what it always did.
        let view = if ptr_ty == base_ptr_ty {
            view
        } else {
            let bc = air::CastOp::new(ctx, ptr_ty, view);
            bc.set_op(ctx, crate::air_dialect::attrs::CastOpAttr::Bitcast);
            rewriter.insert_operation(ctx, bc.get_operation());
            bc.get_operation().deref(ctx).get_result(0)
        };
        // A gep carries the element type it steps over as an attribute, not in its result type.
        // Without it the emitter has no stride for the index.
        let elem = crate::air_dialect::types::to_pliron(ctx, &velem);
        let gep = air::GepOp::new(ctx, ptr_ty, view, off);
        air::set_elem_ty(gep.get_operation(), ctx, elem);
        rewriter.insert_operation(ctx, gep.get_operation());
        let addr = gep.get_operation().deref(ctx).get_result(0);

        let mut replacements = Vec::with_capacity(res_tys.len());
        if is_store {
            let Some(&value) = operands.first() else {
                self.bail(code, "store-value-missing");
                return Ok(());
            };
            let st = air::StoreOp::new(ctx, value, addr);
            air::set_align(st.get_operation(), ctx, align);
            rewriter.insert_operation(ctx, st.get_operation());
            if let Some((do_b, skip, saved)) = store_guard {
                let br = air::BrOp::new(ctx, skip, vec![]);
                br.get_operation().insert_at_back(do_b, ctx);
                rewriter.set_insertion_point(saved);
            }
        } else {
            let Some(&res_ty) = res_tys.first() else {
                self.bail(code, "load-result-missing");
                return Ok(());
            };
            let Some(res_ty) = self.air_ty(ctx, res_ty) else {
                self.bail(code, "load-result-type");
                return Ok(());
            };
            let ld = air::LoadOp::new(ctx, res_ty, addr);
            air::set_align(ld.get_operation(), ctx, align);
            rewriter.insert_operation(ctx, ld.get_operation());
            let mut val = ld.get_operation().deref(ctx).get_result(0);
            // Out-of-range lanes read element 0 above. Give them the pad value instead.
            if let Some(p) = pred {
                let Some(pad) = pad_value else {
                    self.bail(code, "padding value has no constant at this element type");
                    return Ok(());
                };
                let pc = air::ConstBitsOp::new(ctx, res_ty, pad);
                rewriter.insert_operation(ctx, pc.get_operation());
                let pcv = pc.get_operation().deref(ctx).get_result(0);
                let sel = air::SelectOp::new(ctx, res_ty, p, val, pcv);
                rewriter.insert_operation(ctx, sel.get_operation());
                val = sel.get_operation().deref(ctx).get_result(0);
            }
            // The staging loop also writes the loaded value to its cache slot, the padding
            // select included. That is the value the slot holds.
            if let Some((pi, true)) = cache_member {
                self.emit_cache_store(ctx, rewriter, pi, val)?;
            }
            replacements.push(val);
        }
        // Remaining results are ordering tokens, which AGX has nothing to order against.
        self.finish_replacement(ctx, rewriter, op, &res_tys, replacements);
        Ok(())
    }
    /// The element offset a view access addresses, in the static-stride case.
    ///
    /// The lane's within-tile part is the row-major decomposition of its flat index, each
    /// coordinate paid its own stride (see `lane_decompose`). One term is added per index:
    /// `index[d] * (stride[d] * tile_shape[d])`.
    ///
    /// Nothing is constant-folded. Folding here would diverge from the emitted bytes.
    pub(crate) fn elem_offset(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        view_ty: &wax_ir::types::Type,
        index_ops: &[pliron::value::Value],
    ) -> Option<pliron::value::Value> {
        use crate::air_dialect::attrs::BinOpAttr;
        use wax_ir::types::Type as WT;
        let WT::PartitionView(pv) = view_ty else {
            why("address: not-a-partition-view");
            return None;
        };
        let tile_shape: Vec<i64> = pv.tile_shape.iter().map(|&x| x as i64).collect();
        let strides = &pv.tensor_view.strides;
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);

        let n_elems: i64 = tile_shape.iter().product();
        // A dynamic stride would be resolved from a kernel argument. This route
        // has no equivalent yet, so refuse rather than address with the sentinel.
        if strides.contains(&wax_ir::DYNAMIC) {
            why("address: dynamic-stride");
            return None;
        }
        // The lane is a flat row-major index into the tile, so it decomposes into per-dim
        // coordinates, each paid its own stride. Multiplying the whole lane by the innermost
        // stride is only correct when every outer extent is 1. The fold is interleaved with the
        // decomposition (inside `lane_decompose`) so the emitted adds stay in reverse-d order.
        let rank = index_ops.len();
        let mut acc = None;
        if n_elems > 1 {
            let Some(tp) = self.control.tp else {
                why("address: no-thread-position");
                return None;
            };
            lane_decompose(
                ctx,
                rewriter,
                i32_ty,
                &tile_shape,
                rank,
                tp,
                |d, coord, ctx, rw| {
                    let Some(&sd) = strides.get(d) else {
                        why("address: lane-stride-missing");
                        return None;
                    };
                    let s = rw_imm(ctx, rw, i32_ty, sd);
                    let term = air_bin_i32(ctx, rw, i32_ty, coord, s, BinOpAttr::Mul);
                    acc = Some(match acc {
                        None => term,
                        Some(a) => air_bin_i32(ctx, rw, i32_ty, a, term, BinOpAttr::Add),
                    });
                    Some(())
                },
            )?;
        }

        // The accumulator starts at the immediate 0 and lets the lane term replace it. With
        // no lane term the first index term is therefore `add i32 0, %term`, a real instruction
        // with an immediate operand. Starting from nothing here drops that add and diverges on
        // every rank-0 tile.
        if acc.is_none() {
            acc = Some(rw_imm(ctx, rewriter, i32_ty, 0));
        }

        // One term per index: `index[d] * (stride[d] * tile_shape[d])`.
        for (d, &idx) in index_ops.iter().enumerate() {
            let Some(mult) = strides
                .get(d)
                .and_then(|s| s.checked_mul(*tile_shape.get(d).unwrap_or(&1)))
            else {
                why("address: index-stride-missing");
                return None;
            };
            let m = rw_imm(ctx, rewriter, i32_ty, mult);
            let term = air_bin_i32(ctx, rewriter, i32_ty, idx, m, BinOpAttr::Mul);
            acc = Some(match acc {
                Some(a) => air_bin_i32(ctx, rewriter, i32_ty, a, term, BinOpAttr::Add),
                None => term,
            });
        }
        acc
    }
    /// Per-dim absolute coordinates of this lane's element: `index[d]*tile[d] + c_d`.
    ///
    /// Same decomposition as the address arithmetic, but keeping the coordinates instead of
    /// folding them into an offset, so each can be compared against the tensor extent.
    pub(crate) fn elem_coords(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        view_ty: &wax_ir::types::Type,
        index_ops: &[pliron::value::Value],
    ) -> Option<Vec<pliron::value::Value>> {
        use crate::air_dialect::attrs::BinOpAttr;
        use wax_ir::types::Type as WT;
        let WT::PartitionView(pv) = view_ty else {
            return None;
        };
        let tile_shape: Vec<i64> = pv.tile_shape.iter().map(|&x| x as i64).collect();
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let rank = index_ops.len();
        let n_elems: i64 = tile_shape.iter().product();
        let mut coords: Vec<Option<pliron::value::Value>> = vec![None; rank];
        if n_elems > 1 {
            let tp = self.control.tp?;
            lane_decompose(
                ctx,
                rewriter,
                i32_ty,
                &tile_shape,
                rank,
                tp,
                |d, coord, _ctx, _rw| {
                    coords[d] = Some(coord);
                    Some(())
                },
            )?;
        }
        let mut out = Vec::with_capacity(rank);
        for (d, &idx) in index_ops.iter().enumerate() {
            let tile_d = *tile_shape.get(d).unwrap_or(&1);
            let base = if tile_d == 1 {
                idx
            } else {
                let m = rw_imm(ctx, rewriter, i32_ty, tile_d);
                air_bin_i32(ctx, rewriter, i32_ty, idx, m, BinOpAttr::Mul)
            };
            out.push(match coords[d] {
                None => base,
                Some(c) => air_bin_i32(ctx, rewriter, i32_ty, base, c, BinOpAttr::Add),
            });
        }
        Some(out)
    }
    /// True for lanes whose element lies inside the tensor, or `None` when this access needs
    /// no mask.
    ///
    /// Decided from the bounds collected before conversion, and emitted only when there is at
    /// least one. `elem_coords` materialises the lane decomposition, and leaving that behind on
    /// an unmasked access is not free. Dead arithmetic in a loop body costs the unroll
    /// heuristic its straight-line body.
    pub(crate) fn bounds_predicate(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        view_ty: &wax_ir::types::Type,
        index_ops: &[pliron::value::Value],
    ) -> Option<pliron::value::Value> {
        use crate::air_dialect::attrs::{BinOpAttr, ICmpPredAttr};
        let bounds = self.addr.bounds.get(&op)?.clone();
        if bounds.is_empty() {
            return None;
        }
        let coords = self.elem_coords(ctx, rewriter, view_ty, index_ops)?;
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i1_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let mut pred: Option<pliron::value::Value> = None;
        for (d, b) in bounds {
            let ext = match b {
                Bound::Dyn(v) => v,
                Bound::Fixed(n) => {
                    let c = air::ConstBitsOp::new(ctx, i32_ty, n as u64);
                    rewriter.insert_operation(ctx, c.get_operation());
                    c.get_operation().deref(ctx).get_result(0)
                }
            };
            let coord = *coords.get(d)?;
            let cmp = air::ICmpOp::new(ctx, i1_ty, coord, ext);
            cmp.set_pred(ctx, ICmpPredAttr::Slt);
            rewriter.insert_operation(ctx, cmp.get_operation());
            let ok = cmp.get_operation().deref(ctx).get_result(0);
            pred = Some(match pred {
                None => ok,
                Some(p) => {
                    let a = air::BinOp::new(ctx, i1_ty, p, ok);
                    a.set_op(ctx, BinOpAttr::And);
                    rewriter.insert_operation(ctx, a.get_operation());
                    a.get_operation().deref(ctx).get_result(0)
                }
            });
        }
        pred
    }
}

/// The source element a broadcast's destination wants at flat index `flat`.
///
/// `wax.broadcast` keeps rank and expands dims whose source extent is 1, so the source
/// coordinate is the destination's in a kept dim and 0 in an expanded one. It may also add
/// leading dims, so the shapes line up on the right. `None` when the two are not in that
/// relation, which is a refusal rather than a guess.
pub(crate) fn broadcast_src_flat(src: &[i64], dst: &[i64], flat: i64) -> Option<i64> {
    if src.len() > dst.len() {
        return None;
    }
    let padded: Vec<i64>;
    let src = if src.len() < dst.len() {
        padded = std::iter::repeat_n(1, dst.len() - src.len())
            .chain(src.iter().copied())
            .collect();
        &padded[..]
    } else {
        src
    };
    let mut inner_d: i64 = dst.iter().product();
    let mut inner_s: i64 = src.iter().product();
    let mut out = 0i64;
    for d in 0..dst.len() {
        inner_d /= dst[d];
        inner_s /= src[d];
        if src[d] == dst[d] {
            out += ((flat / inner_d) % dst[d]) * inner_s;
        } else if src[d] != 1 {
            return None;
        }
    }
    Some(out)
}

/// Whether forwarding the operand is a correct lowering of this broadcast.
///
/// One element per lane and the source's lane term is `tp & (n_src - 1)`, so lane `tp` holds
/// source element `tp & (n_src - 1)`. Forwarding is correct exactly when every lane already
/// holds the element the destination wants there.
pub(crate) fn broadcast_is_forwardable(src: &[i64], dst: &[i64]) -> bool {
    let (src_n, dst_n): (i64, i64) = (src.iter().product(), dst.iter().product());
    if src_n <= 0 || dst_n <= 0 || src.len() > dst.len() {
        return false;
    }
    (0..dst_n).all(|f| broadcast_src_flat(src, dst, f) == Some(f & (src_n - 1)))
}

/// Whether `dst` is `src` with its last dim opened from 1 to the 32 lanes a row-replicated
/// value already occupies. Only meaningful for a value in `row_replicated`, where the lanes
/// hold the row's total rather than the single logical column.
pub(crate) fn expands_replicated_row(src: &[i64], dst: &[i64]) -> bool {
    dst.len() == 2
        && dst[1] == 32
        && match src.len() {
            1 => src[0] == dst[0],
            2 => src[0] == dst[0] && src[1] == 1,
            _ => false,
        }
}
