//! Matrix fragments. Load/store, `mma` ops, constants, reduces, fragment grid, address machinery.

use super::*;
use crate::air_dialect::attrs::BinKind;
use crate::air_dialect::ops as air;
use pliron::r#type::Typed;
use pliron::value::Value;
use std::collections::HashMap;
use wax_ir::dialect::ops::{BITS_16, BITS_32, BITS_64};

/// Information required by both masked fragment load and store.
pub(crate) struct MaskedFragPreamble {
    frag_ty: TypeHandle,
    pair_frag_ty: TypeHandle,
    pair_int_ty: TypeHandle,
    iptr_ty: TypeHandle,
    c0: Value,
    c1: Value,
    row: Value,
    col: Value,
    rs: Value,
    cs: Value,
    m3_i16: Value,
    m0_i16: Value,
    bytes_i16: Value,
}

/// Fragment load and store parameters.
#[derive(Clone, Copy)]
pub(crate) struct FragAccess<'a> {
    pub(crate) code: Opcode,
    pub(crate) view: Value,
    pub(crate) view_ty: &'a WaxType,
    pub(crate) index_ops: &'a [Value],
}

/// Checked fragment addressing information.
/// Base view pointer, base pointer type, element type and suffix, and striding.
#[derive(Clone)]
pub(crate) struct FragAddr {
    base: Value,
    elem_ty: TypeHandle,
    ptr_ty: TypeHandle,
    suffix: &'static str,
    row_stride: i64,
    col_stride: i64,
    row_strides: Value,
    col_strides: Value,
    zeros: Value,
}

/// Dynamic stride values for a fragment load/store, resolved before conversion.
pub(crate) fn collect_frag_strides(
    ctx: &Context,
    func: Ptr<Operation>,
) -> HashMap<Ptr<Operation>, Vec<Option<Value>>> {
    let mut out: HashMap<Ptr<Operation>, Vec<Option<Value>>> = HashMap::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            let view_at = match opcode_of(ctx, op) {
                Some(Opcode::LoadViewTko) => 0,
                Some(Opcode::StoreViewTko) => 1,
                _ => continue,
            };
            let Some(view) = op.deref(ctx).operands().nth(view_at) else {
                continue;
            };
            let Some(WaxType::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx)) else {
                continue;
            };
            let strides = &pv.tensor_view.strides;
            if !strides.contains(&wax_ir::DYNAMIC) {
                continue;
            }
            let Some(tv_def) = tensor_view_def(ctx, view) else {
                continue;
            };
            let (n_base, n_dyn_shape, n_dyn_stride) = wax_operand_segments(ctx, tv_def);
            let operands: Vec<_> = tv_def.deref(ctx).operands().collect();
            let start = n_base + n_dyn_shape;
            let dyn_vals: Vec<Value> = operands
                .iter()
                .skip(start)
                .take(n_dyn_stride)
                .copied()
                .map(|v| trace_through_assume(ctx, v))
                .collect();
            let mut di = 0;
            let mut per_dim: Vec<Option<Value>> = Vec::new();
            for &s in strides {
                if s == wax_ir::DYNAMIC {
                    per_dim.push(dyn_vals.get(di).copied());
                    di += 1;
                } else {
                    per_dim.push(None);
                }
            }
            out.insert(op, per_dim);
        }
    });
    out
}

/// Get emask integer `(type,suffix,width)` from float type.
fn emask_int(ctx: &Context, frag_ty: TypeHandle) -> Option<(TypeHandle, &'static str, i64)> {
    use crate::air_dialect::types as t;
    if frag_ty == t::f32_ty(ctx) {
        Some((t::i32_ty(ctx), "i32", 4))
    } else if frag_ty == t::f16_ty(ctx) || frag_ty == t::bf16_ty(ctx) {
        Some((t::i16_ty(ctx), "i16", 2))
    } else {
        None
    }
}

/// The register scalar of a fragment at this element type.
pub(crate) fn sgmma_frag_scalar(
    ctx: &Context,
    e: &wax_ir::types::ScalarType,
) -> Option<TypeHandle> {
    crate::types::float_elem(ctx, e).map(|(t, _)| t)
}

/// `(tile_h, tile_w), (row_stride, col_stride)`. Fragments are 2D by construction.
fn frag_geometry(pv: &wax_ir::types::PartitionViewType) -> Option<((i64, i64), (i64, i64))> {
    Some((
        (
            i64::from(*pv.tile_shape.first()?),
            i64::from(*pv.tile_shape.get(1)?),
        ),
        (
            pv.tensor_view.strides.first().copied()?,
            pv.tensor_view.strides.get(1).copied()?,
        ),
    ))
}

/// Intrinsic suffix for a fragment load at this element type.
pub(crate) fn sgmma_load_suffix(e: &wax_ir::types::ScalarType) -> Option<&'static str> {
    use wax_ir::types::ScalarType as S;
    Some(match e {
        S::F32 => "v64f32.p1f32",
        S::F16 => "v64f16.p1f16",
        S::BF16 => "v64bf16.p1bf16",
        _ => return None,
    })
}

/// Masked fragment grid.
/// Extents, block origina, absolute coordinates, and edge mask.
struct EdgeGrid {
    row_extent: Option<Value>,
    col_extent: Option<Value>,
    row_origin: Option<Value>,
    col_origin: Option<Value>,
    row: Value,
    col: Value,
    c0: Value,
    m3_i16: Value,
    m0_i16: Value,
    i16_ty: TypeHandle,
    i1_ty: TypeHandle,
}

impl EdgeGrid {
    fn new(
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        p: &MaskedFragPreamble,
        bounds: &[(usize, Bound)],
        index_ops: &[Value],
        tile_h: i64,
        tile_w: i64,
    ) -> Self {
        // The extents, materialized from the planned bounds: a fixed ragged extent is a
        // constant, a dynamic one the promoted parameter the prologue loaded.
        let mut ext = |dim: usize| -> Option<Value> {
            bounds
                .iter()
                .find(|(d, _)| *d == dim)
                .map(|(_, b)| match b {
                    Bound::Dyn(v) => *v,
                    Bound::Fixed(n) => air_const_i32(ctx, *n),
                })
        };
        let row_extent = ext(0);
        let col_extent = ext(1);
        // The block's own origin, added once rather than per sub-tile.
        let mut org = |b: Option<&Value>, tile: i64| {
            b.map(|&b| {
                let t = air_const_i32(ctx, tile);
                air_bin_i32(ctx, rewriter, b, t, BinKind::Mul)
            })
        };
        let row_origin = org(index_ops.first(), tile_h);
        let col_origin = org(index_ops.get(1), tile_w);
        Self {
            row_extent,
            col_extent,
            row_origin,
            col_origin,
            row: p.row,
            col: p.col,
            c0: p.c0,
            m3_i16: p.m3_i16,
            m0_i16: p.m0_i16,
            i16_ty: t::i16_ty(ctx),
            i1_ty: t::i1_ty(ctx),
        }
    }

    /// Absolute coordinates of sub-tile `(r, c)`. Block origin + sub-tile + lane.
    fn coords(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        r: usize,
        c: usize,
    ) -> (Value, Value) {
        let mut axis = |mut a: Value, step: usize, org: Option<Value>| {
            if step > 0 {
                let t = air_const_i32(ctx, step as i64 * 8);
                a = air_bin_i32(ctx, rewriter, a, t, BinKind::Add);
            }
            if let Some(o) = org {
                a = air_bin_i32(ctx, rewriter, a, o, BinKind::Add);
            }
            a
        };
        let ar = axis(self.row, r, self.row_origin);
        let ac = axis(self.col, c, self.col_origin);
        (ar, ac)
    }

    /// Apply masking as needed. `edgecheck` on column axis, select on row axis.
    fn mask(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        ar: Value,
        ac: Value,
    ) -> Value {
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let mut mask = match self.col_extent {
            Some(ec) => {
                // TODO: Prefer no magic strings.
                let m = air::call(
                    ctx,
                    "@llvm.agx3.edgecheck",
                    Some(self.i16_ty),
                    vec![ac, self.c0, ec],
                );
                let mv = emit(ctx, rewriter, m);
                let a = air::bin(ctx, BinKind::And, self.i16_ty, mv, self.m3_i16);
                emit(ctx, rewriter, a)
            }
            None => self.m3_i16,
        };
        if let Some(er) = self.row_extent {
            let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, self.i1_ty, ar, er);
            let rok = emit(ctx, rewriter, cmp);
            let s = air::select(ctx, self.i16_ty, rok, mask, self.m0_i16);
            mask = emit(ctx, rewriter, s);
        }
        mask
    }
}

fn sgmma_dims_of(ctx: &Context, operands_info: &OperandsInfo, v: Value) -> Option<(usize, usize)> {
    wax_operand_ty(ctx, operands_info, v)
        .as_ref()
        .and_then(crate::types::sgmma_dims)
}

impl WaxToAir {
    /// Get `nr x nc` fragment grid of value, if present.
    pub(crate) fn grid_of(&self, v: Value) -> Option<&Vec<Value>> {
        self.frags.grids.get(&v)
    }
    /// Fragment dimensions of a value, if present.
    pub(crate) fn frag_dims(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        v: Value,
    ) -> Option<(usize, usize)> {
        if !self.frags.fragments.contains(&v) {
            return None;
        }
        sgmma_dims_of(ctx, operands_info, v)
    }
    /// Fragment dimensions of a value, possibly registered in `grids`, if present.
    pub(crate) fn frag_dims_of_grid(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        v: Value,
    ) -> Option<(usize, usize)> {
        if !self.frags.fragments.contains(&v) && !self.frags.grids.contains_key(&v) {
            return None;
        }
        sgmma_dims_of(ctx, operands_info, v)
    }
    /// Fragment row reduce via threadgroup memory.
    /// Each lane folds the eight columns of its row, resulting in one scalar per lane.
    pub(crate) fn fragment_row_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
        rc: ReduceCtx,
    ) -> Result<()> {
        let ReduceCtx { src, intrinsic, .. } = rc;
        use crate::air_dialect::attrs::BinKind;
        type V = Value;
        let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, src) else {
            return self.refuse(Opcode::Reduce, "dims unreadable");
        };
        let Some(grid) = self.grid_of(src) else {
            return self.refuse(Opcode::Reduce, "no grid");
        };
        let i32_ty = t::i32_ty(ctx);
        let f32_ty = t::f32_ty(ctx);
        // NOTE: within lane register folding was attempted, but only even rows were present in
        // the simdgroup matrix when reading via `extract_elem`. There may be a way to read the
        // values correctly, but since we haven't found that yet we instead store to threadgroup
        // memory in the same manner as `simdgroup_store(C, scratch, 8)` (`<8,8>:<1,8>`), then
        // fold per lane.
        if nr != 1 || nc != 1 {
            return self.refuse(Opcode::Reduce, "single 8x8 fragment");
        }
        let Some(tile0) = grid.first().copied() else {
            return self.refuse(Opcode::Reduce, "empty grid");
        };
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        const REGION: i64 = 32 * crate::wax_to_air::reduce::SCRATCH_PITCH;
        if i64::from(rt.scratch_n) < REGION + 64 {
            return self.refuse(Opcode::Reduce, "threadgroup scratch too small");
        }
        let base = {
            let off = air_const_i32(ctx, REGION);
            rt.scratch_gep(ctx, rewriter, off)
        };
        // `(<64 x float>, ptr addrspace(3), <2 x i64> row, <2 x i64> col, <2 x i64> zero)`.
        let row_strides = stride_vec2(ctx, rewriter, 8, None, 8, None);
        let col_strides = stride_vec2(ctx, rewriter, 1, None, 8, None);
        let zeros = {
            let ty = t::vec_ty(ctx, 2, t::i64_ty(ctx));
            let z = air::ZeroOp::new(ctx, ty);
            emit_op(ctx, rewriter, z)
        };
        let store = air::call(
            ctx,
            "@air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            None,
            vec![tile0, base, row_strides, col_strides, zeros],
        );
        rewriter.insert_operation(ctx, store);
        air_wg_barrier(ctx, rewriter);

        // Intermediate results stored to lanes `{s, s+8, s+16, s+24}`, row index is `lane & 7`.
        let row = {
            let c7 = air_const_i32(ctx, 7);
            let a = air::bin(ctx, BinKind::And, i32_ty, rt.lane, c7);
            emit(ctx, rewriter, a)
        };
        let row_base = {
            let c8 = air_const_i32(ctx, 8);
            let m = air::bin(ctx, BinKind::Mul, i32_ty, row, c8);
            let mv = emit(ctx, rewriter, m);
            let off = air_const_i32(ctx, REGION);
            let a = air::bin(ctx, BinKind::Add, i32_ty, mv, off);
            emit(ctx, rewriter, a)
        };
        let mut res: Option<V> = None;
        for c in 0..8i64 {
            let flat = if c == 0 {
                row_base
            } else {
                let cc = air_const_i32(ctx, c);
                let a = air::bin(ctx, BinKind::Add, i32_ty, row_base, cc);
                emit(ctx, rewriter, a)
            };
            let g = rt.scratch_gep(ctx, rewriter, flat);
            let ld = air::LoadOp::new(ctx, f32_ty, g, 4);
            let v = emit_op(ctx, rewriter, ld);
            res = Some(match res {
                None => v,
                Some(acc) => emit_combine(ctx, rewriter, acc, v, intrinsic, f32_ty),
            });
        }
        let Some(res) = res else {
            return self.refuse(Opcode::Reduce, "no columns");
        };
        rewriter.replace_operation_with_values(ctx, op, vec![res]);
        self.converted += 1;
        Ok(())
    }
    /// Extract and verify fragment addressing information.
    pub(crate) fn fragment_addr<'a>(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        acc: FragAccess<'a>,
    ) -> Result<Option<FragAddr>> {
        let FragAccess {
            code,
            view,
            view_ty,
            index_ops,
        } = acc;
        let WaxType::PartitionView(pv) = view_ty else {
            return self.refuse(code, "no partition view").map(|_| None);
        };
        if let Some(bounds) = self.addr.bounds.get(&op)
            && !bounds.is_empty()
        {
            return self.refuse(code, "empty access bounds").map(|_| None);
        }
        let Some(((tile_h, tile_w), (row_stride, col_stride))) = frag_geometry(pv) else {
            return self.refuse(code, "fragment view is not 2D").map(|_| None);
        };
        // Dynamic strides, resolved before conversion.
        let dyns = self.addr.frag_strides.get(&op);
        let row_d = dyns.and_then(|v| v.first()).copied().flatten();
        let col_d = dyns.and_then(|v| v.get(1)).copied().flatten();
        if row_stride == wax_ir::DYNAMIC && row_d.is_none() {
            return self.refuse(code, "unresolved dyn row stride").map(|_| None);
        }
        if col_stride == wax_ir::DYNAMIC && col_d.is_none() {
            return self.refuse(code, "unresolved dyn col stride").map(|_| None);
        }
        let Some(elem_ty) = sgmma_frag_scalar(ctx, &pv.tensor_view.element_type) else {
            return self.refuse(code, "type is not f32/f16/bf16").map(|_| None);
        };
        let Some(suffix) = sgmma_load_suffix(&pv.tensor_view.element_type) else {
            return self.refuse(code, "type has intrinsic suffix").map(|_| None);
        };
        let ptr_ty = view.get_type(ctx);

        // Calculate base pointer from index ops.
        // Block indices scale by the full tile dim.
        // Dynamic stride multiplies by stride value, then tile scale.
        // Static stride multiplies by `stride * scale` constant.
        let base = if index_ops.is_empty() {
            view
        } else {
            let i32_ty = t::i32_ty(ctx);
            let mut off = None;
            for (d, &idx) in index_ops.iter().enumerate().take(2) {
                let (s, t, sd) = if d == 0 {
                    (row_stride, tile_h, row_d)
                } else {
                    (col_stride, tile_w, col_d)
                };
                let term = match sd {
                    Some(reg) => {
                        let m = air::bin(ctx, BinKind::Mul, i32_ty, idx, reg);
                        let mv = emit(ctx, rewriter, m);
                        if t == 1 {
                            mv
                        } else {
                            let c = air::const_int(ctx, BITS_32, t as u64);
                            let cv = emit_op(ctx, rewriter, c);
                            let m2 = air::bin(ctx, BinKind::Mul, i32_ty, mv, cv);
                            emit(ctx, rewriter, m2)
                        }
                    }
                    None => {
                        let m = air::const_int(ctx, BITS_32, (s * t) as u64);
                        let mv = emit_op(ctx, rewriter, m);
                        let mul = air::bin(ctx, BinKind::Mul, i32_ty, idx, mv);
                        emit(ctx, rewriter, mul)
                    }
                };
                off = Some(match off {
                    None => term,
                    Some(a) => {
                        let add = air::bin(ctx, BinKind::Add, i32_ty, a, term);
                        emit(ctx, rewriter, add)
                    }
                });
            }
            match off {
                None => view,
                Some(o) => {
                    let g = air::GepOp::new(ctx, ptr_ty, view, o, elem_ty);
                    emit_op(ctx, rewriter, g)
                }
            }
        };
        // The two stride vectors, built once for every sub-tile.
        let row_strides = stride_vec2(ctx, rewriter, row_stride, row_d, 8, None);
        let col_strides = stride_vec2(ctx, rewriter, col_stride, col_d, row_stride, row_d);
        let zeros = {
            let ty = t::vec_ty(ctx, 2, t::i64_ty(ctx));
            let z = air::ZeroOp::new(ctx, ty);
            emit_op(ctx, rewriter, z)
        };
        Ok(Some(FragAddr {
            base,
            ptr_ty,
            elem_ty,
            suffix,
            row_stride,
            col_stride,
            row_strides,
            col_strides,
            zeros,
        }))
    }
    /// `nr x nc` fragment loads. One `@air.simdgroup_matrix_8x8_load` per sub-tile.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rewrite_fragment_load(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let Some(f) = self.fragment_addr(ctx, rewriter, op, acc)? else {
            return Ok(());
        };
        let load_suf = f.suffix;
        let frag_ty = t::vec_ty(ctx, 64, f.elem_ty);

        // TODO: Prefer no magic strings.
        let load_name = format!("@air.simdgroup_matrix_8x8_load.{load_suf}");
        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let sub = fragment_sub_ptr(ctx, rewriter, &f, r, c);
                let call = air::call(
                    ctx,
                    &load_name,
                    Some(frag_ty),
                    vec![sub, f.row_strides, f.col_strides, f.zeros],
                );
                frags.push(emit(ctx, rewriter, call));
            }
        }
        // The result stands for the whole grid. Bind fragment 0 and record the rest.
        let res_tys = op_result_types(ctx, op);
        let Some(&rep) = frags.first() else {
            self.bail(Opcode::LoadViewTko, "fragment load produced no fragments");
            return Ok(());
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, frags.clone());
        self.finish_replacement(ctx, rewriter, op, &res_tys, vec![rep]);
        // Keyed by the representative too, so a consumer holding the replaced value finds it.
        self.frags.grids.insert(rep, frags);
        Ok(())
    }

    /// Build the shared half of the masked fragment load and store.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn masked_frag_preamble(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        int_ty: TypeHandle,
        frag_ty: TypeHandle,
        elem_bytes: i64,
        row_s: i64,
        col_s: i64,
    ) -> Option<MaskedFragPreamble> {
        use crate::air_dialect::attrs::BinKind;
        let Some(tp) = self.control.tp else {
            self.bail(code, "no thread position");
            return None;
        };
        let i32_ty = t::i32_ty(ctx);
        let pair_frag_ty = t::vec_ty(ctx, 2, frag_ty);
        let pair_int_ty = t::vec_ty(ctx, 2, int_ty);
        let iptr_ty = t::device_ptr_ty(ctx, int_ty);

        // 8x8 fragment coordinates.
        let c0 = air_const_i32(ctx, 0);
        let c1 = air_const_i32(ctx, 1);
        let c2 = air_const_i32(ctx, 2);
        let c3 = air_const_i32(ctx, 3);
        let c4 = air_const_i32(ctx, 4);
        let c7 = air_const_i32(ctx, 7);
        let lane = {
            let e = air::extract_elem(ctx, i32_ty, tp, c0);
            emit(ctx, rewriter, e)
        };
        // Calculate row and column index:
        // let x   = (lane >> 3) & 1;
        // let y   =  lane >> 4;
        // let loc =  lane & 7;
        // let row = (loc >> 1) + (y << 2);
        // let col = ((loc & 1) << 1) + (x << 2);
        let x1 = air_bin_i32(ctx, rewriter, lane, c3, BinKind::LShr);
        let x = air_bin_i32(ctx, rewriter, x1, c1, BinKind::And);
        let y = air_bin_i32(ctx, rewriter, lane, c4, BinKind::LShr);
        let loc = air_bin_i32(ctx, rewriter, lane, c7, BinKind::And);
        let rloc = air_bin_i32(ctx, rewriter, loc, c1, BinKind::LShr);
        let y4 = air_bin_i32(ctx, rewriter, y, c2, BinKind::Shl);
        let row = air_bin_i32(ctx, rewriter, rloc, y4, BinKind::Add);
        let lmod2 = air_bin_i32(ctx, rewriter, loc, c1, BinKind::And);
        let clo = air_bin_i32(ctx, rewriter, lmod2, c1, BinKind::Shl);
        let x4 = air_bin_i32(ctx, rewriter, x, c2, BinKind::Shl);
        let col = air_bin_i32(ctx, rewriter, clo, x4, BinKind::Add);

        // Static strides are immediates, so they emit no stride of their own.
        let (rs, cs) = match self
            .addr
            .frag_strides
            .get(&op)
            .and_then(|v| v.first())
            .copied()
            .flatten()
        {
            Some(reg) => (reg, air_const_i32(ctx, col_s)),
            None => (air_const_i32(ctx, row_s), air_const_i32(ctx, col_s)),
        };
        Some(MaskedFragPreamble {
            frag_ty,
            pair_frag_ty,
            pair_int_ty,
            iptr_ty,
            c0,
            c1,
            row,
            col,
            rs,
            cs,
            m3_i16: air_const_int(ctx, BITS_16, 3),
            m0_i16: air_const_int(ctx, BITS_16, 0),
            bytes_i16: air_const_int(ctx, BITS_16, elem_bytes as u64),
        })
    }
    /// Masked fragment load.
    /// When column stride is 1 we can use simple emask loads.
    /// When it is not we clamp, load, and select with padding.
    pub(crate) fn rewrite_fragment_load_masked(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        use crate::air_dialect::attrs::{BinKind, CastOpAttr};
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let FragAccess {
            view,
            view_ty,
            index_ops,
            ..
        } = acc;
        let WaxType::PartitionView(pv) = view_ty else {
            return self.refuse(
                Opcode::LoadViewTko,
                "masked fragment load needs a partition view",
            );
        };
        let Some(bounds) = self.addr.bounds.get(&op).cloned() else {
            return self.refuse(Opcode::LoadViewTko, "masked fragment load without bounds");
        };
        let Some(frag_ty) = sgmma_frag_scalar(ctx, &pv.tensor_view.element_type) else {
            return self.refuse(Opcode::LoadViewTko, "fragment type is not f32/f16/bf16");
        };
        let Some(pad_bits) = pv.padding_value.and_then(|p| padding_bits(ctx, frag_ty, p)) else {
            return self.refuse(Opcode::LoadViewTko, "masked fragment load without padding");
        };
        let Some((int_ty, int_suf, elem_bytes)) = emask_int(ctx, frag_ty) else {
            return self.refuse(Opcode::LoadViewTko, "masked fragment load element type");
        };
        let Some(((tile_h, tile_w), (row_s, col_s))) = frag_geometry(pv) else {
            return self.refuse(Opcode::LoadViewTko, "masked fragment view is not rank 2");
        };
        // Dynamic row stride is fine. The mask math only needs the stride's runtime register,
        // which `frag_strides` resolved. Dynamic col stride not supported yet.
        if col_s == wax_ir::DYNAMIC {
            return self.refuse(
                Opcode::LoadViewTko,
                "masked fragment load with dynamic col stride",
            );
        }
        let Some(p) = self.masked_frag_preamble(
            ctx,
            rewriter,
            op,
            Opcode::LoadViewTko,
            int_ty,
            frag_ty,
            elem_bytes,
            row_s,
            col_s,
        ) else {
            return Ok(());
        };
        let MaskedFragPreamble {
            frag_ty,
            pair_frag_ty,
            pair_int_ty,
            iptr_ty,
            c0,
            c1,
            rs,
            cs,
            m3_i16,
            bytes_i16,
            ..
        } = p;
        let (i32_ty, i1_ty) = (t::i32_ty(ctx), t::i1_ty(ctx));

        let frag_vec_h = t::vec_ty(ctx, 64, frag_ty);
        let predicated = col_s == 1;
        let align = elem_bytes as u32;
        let Some(pad) = air_const(ctx, frag_ty, pad_bits) else {
            return self.refuse(Opcode::LoadViewTko, "invalid constant type for padding");
        };
        let undef_frag = {
            let u = air::UndefOp::new(ctx, frag_vec_h);
            emit_op(ctx, rewriter, u)
        };

        let grid = EdgeGrid::new(ctx, rewriter, &p, &bounds, index_ops, tile_h, tile_w);
        // TODO: Prefer no magic strings
        let emask_name = format!("@llvm.agx3.load.with.emask.global.v2{int_suf}");
        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let (ar, ac0) = grid.coords(ctx, rewriter, r, c);
                let frag = if predicated {
                    let roff = air_bin_i32(ctx, rewriter, ar, rs, BinKind::Mul);
                    let coff = air_bin_i32(ctx, rewriter, ac0, cs, BinKind::Mul);
                    let off = air_bin_i32(ctx, rewriter, roff, coff, BinKind::Add);
                    let g = air::GepOp::new(ctx, view.get_type(ctx), view, off, frag_ty);
                    let gep = emit_op(ctx, rewriter, g);
                    let mask = grid.mask(ctx, rewriter, ar, ac0);
                    let iptr = {
                        let bc = air::cast(ctx, CastOpAttr::Bitcast, iptr_ty, gep);
                        emit(ctx, rewriter, bc)
                    };
                    let vi = air::call(
                        ctx,
                        &emask_name,
                        Some(pair_int_ty),
                        vec![iptr, mask, m3_i16, bytes_i16],
                    );
                    let vi = emit(ctx, rewriter, vi);
                    let vf = {
                        let bc = air::cast(ctx, CastOpAttr::Bitcast, pair_frag_ty, vi);
                        emit(ctx, rewriter, bc)
                    };
                    let v0 = {
                        let e = air::extract_elem(ctx, frag_ty, vf, c0);
                        emit(ctx, rewriter, e)
                    };
                    let f0 = {
                        let i = air::insert_elem(ctx, frag_vec_h, undef_frag, v0, c0);
                        emit(ctx, rewriter, i)
                    };
                    let v1 = {
                        let e = air::extract_elem(ctx, frag_ty, vf, c1);
                        emit(ctx, rewriter, e)
                    };
                    {
                        let i = air::insert_elem(ctx, frag_vec_h, f0, v1, c1);
                        emit(ctx, rewriter, i)
                    }
                } else {
                    // Portable spelling: clamp, load, select the pad value.
                    let rok = grid.row_extent.map(|er| {
                        let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, ar, er);
                        emit(ctx, rewriter, cmp)
                    });
                    let ar_addr = match grid.row_extent {
                        Some(er) => {
                            let last = air_bin_i32(ctx, rewriter, er, c1, BinKind::Sub);
                            let lt = {
                                let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, ar, er);
                                emit(ctx, rewriter, cmp)
                            };
                            let s = air::select(ctx, i32_ty, lt, ar, last);
                            emit(ctx, rewriter, s)
                        }
                        None => ar,
                    };
                    let roff = air_bin_i32(ctx, rewriter, ar_addr, rs, BinKind::Mul);
                    let mut f2 = undef_frag;
                    for j in 0..2i64 {
                        let ac = if j == 0 {
                            ac0
                        } else {
                            air_bin_i32(ctx, rewriter, ac0, c1, BinKind::Add)
                        };
                        let cok = grid.col_extent.map(|ec| {
                            let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, ac, ec);
                            emit(ctx, rewriter, cmp)
                        });
                        let ac_addr = match grid.col_extent {
                            Some(ec) => {
                                let last = air_bin_i32(ctx, rewriter, ec, c1, BinKind::Sub);
                                let lt = {
                                    let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, ac, ec);
                                    emit(ctx, rewriter, cmp)
                                };
                                let s = air::select(ctx, i32_ty, lt, ac, last);
                                emit(ctx, rewriter, s)
                            }
                            None => ac,
                        };
                        let coff = air_bin_i32(ctx, rewriter, ac_addr, cs, BinKind::Mul);
                        let off = air_bin_i32(ctx, rewriter, roff, coff, BinKind::Add);
                        let g = air::GepOp::new(ctx, view.get_type(ctx), view, off, frag_ty);
                        let gep = emit_op(ctx, rewriter, g);
                        let ld = air::LoadOp::new(ctx, frag_ty, gep, align);
                        let v = emit_op(ctx, rewriter, ld);
                        let v = match (rok, cok) {
                            (Some(a), Some(b)) => {
                                let p = {
                                    let a2 = air::bin(ctx, BinKind::And, i1_ty, a, b);
                                    emit(ctx, rewriter, a2)
                                };
                                let s = air::select(ctx, frag_ty, p, v, pad);
                                emit(ctx, rewriter, s)
                            }
                            (Some(a), None) => {
                                let s = air::select(ctx, frag_ty, a, v, pad);
                                emit(ctx, rewriter, s)
                            }
                            (None, Some(b)) => {
                                let s = air::select(ctx, frag_ty, b, v, pad);
                                emit(ctx, rewriter, s)
                            }
                            (None, None) => v,
                        };
                        let idx = if j == 0 { c0 } else { c1 };
                        let i = air::insert_elem(ctx, frag_vec_h, f2, v, idx);
                        f2 = emit(ctx, rewriter, i);
                    }
                    f2
                };
                frags.push(frag);
            }
        }

        // Result for the whole grid.
        let res_tys = op_result_types(ctx, op);
        let Some(&rep) = frags.first() else {
            return self.refuse(Opcode::LoadViewTko, "masked fragment load produced nothing");
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, frags.clone());
        self.finish_replacement(ctx, rewriter, op, &res_tys, vec![rep]);
        // Make grid findable from replaced value.
        self.frags.grids.insert(rep, frags);
        Ok(())
    }

    /// `nr x nc` fragment stores. One `@air.simdgroup_matrix_8x8_store` per sub-tile.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rewrite_fragment_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        value: Value,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let Some(f) = self.fragment_addr(ctx, rewriter, op, acc)? else {
            return Ok(());
        };

        let Some(frags) = self.grid_of(value).cloned() else {
            return self.refuse(Opcode::StoreViewTko, "fragment store value has no grid");
        };
        if frags.len() != nr * nc {
            return self.refuse(Opcode::StoreViewTko, "incorrect fragment store grid size");
        }
        // Accumulator is `<64 x float>`. Cast if required.
        let want = t::vec_ty(ctx, 64, f.elem_ty);
        // TODO: Prefer no magic strings.
        let store_name = format!("@air.simdgroup_matrix_8x8_store.{}", f.suffix);
        for r in 0..nr {
            for c in 0..nc {
                let sub = fragment_sub_ptr(ctx, rewriter, &f, r, c);
                let held = frags[r * nc + c];
                let have = held.get_type(ctx);
                let frag = if have != want {
                    let Some(c) = self.fpcast_frag(ctx, rewriter, held, have, want) else {
                        return Ok(());
                    };
                    c
                } else {
                    held
                };
                let call = air::call(
                    ctx,
                    &store_name,
                    None,
                    vec![frag, sub, f.row_strides, f.col_strides, f.zeros],
                );
                rewriter.insert_operation(ctx, call);
            }
        }
        // undef ordering tokens
        self.erase_to_undef(ctx, rewriter, op)
    }

    /// Masked fragment store.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rewrite_fragment_store_masked(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        value: Value,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        use crate::air_dialect::attrs::{BinKind, CastOpAttr};
        use WaxType as WT;
        let FragAccess {
            view,
            view_ty,
            index_ops,
            ..
        } = acc;
        let WT::PartitionView(pv) = view_ty else {
            return self.refuse(Opcode::StoreViewTko, "missing partition view");
        };
        let Some(bounds) = self.addr.bounds.get(&op).cloned() else {
            return self.refuse(Opcode::StoreViewTko, "masked fragment store without bounds");
        };
        let Some(frag_ty) = sgmma_frag_scalar(ctx, &pv.tensor_view.element_type) else {
            return self.refuse(Opcode::StoreViewTko, "fragment type is not f32/f16/bf16");
        };
        // Emask store is an integer op. Floats bitcast through it. Exists only for 16/32-bits.
        let Some((int_ty, int_suf, elem_bytes)) = emask_int(ctx, frag_ty) else {
            return self.refuse(Opcode::StoreViewTko, "masked fragment store element type");
        };
        let Some(frags) = self.grid_of(value).cloned() else {
            return self.refuse(Opcode::StoreViewTko, "no fragment grid");
        };
        if frags.len() != nr * nc {
            return self.refuse(Opcode::StoreViewTko, "incorrect fragment store grid size");
        }
        let Some(((tile_h, tile_w), (row_s, col_s))) = frag_geometry(pv) else {
            return self.refuse(Opcode::StoreViewTko, "masked fragment view is not rank 2");
        };
        // Dynamic row stride is fine. The mask math only needs the stride's runtime register,
        // which `frag_strides` resolved. Dynamic col stride not supported yet.
        if col_s == wax_ir::DYNAMIC {
            return self.refuse(
                Opcode::StoreViewTko,
                "masked fragment store with dynamic col stride",
            );
        }
        // TODO: Implement non-unit column stride variant.
        if col_s != 1 {
            return self.refuse(Opcode::StoreViewTko, "unsupported column stride");
        }
        let Some(p) = self.masked_frag_preamble(
            ctx,
            rewriter,
            op,
            Opcode::StoreViewTko,
            int_ty,
            frag_ty,
            elem_bytes,
            row_s,
            col_s,
        ) else {
            return Ok(());
        };
        let MaskedFragPreamble {
            frag_ty,
            pair_frag_ty,
            pair_int_ty,
            iptr_ty,
            c0,
            c1,
            rs,
            cs,
            m3_i16,
            bytes_i16,
            ..
        } = p;
        let grid = EdgeGrid::new(ctx, rewriter, &p, &bounds, index_ops, tile_h, tile_w);

        let u = air::UndefOp::new(ctx, pair_frag_ty);
        let undef_pair = emit_op(ctx, rewriter, u);

        let frag_8x8_ty = t::vec_ty(ctx, 64, frag_ty);
        // TODO: Prefer no magic strings.
        let emask_name = format!("@llvm.agx3.store.with.emask.global.v2{int_suf}");
        for (n, &frag) in frags.iter().enumerate() {
            let (r, c) = (n / nc, n % nc);
            // Accumulator is `<64 x float>`. Cast if required.
            let actual = frag.get_type(ctx);
            let frag = if actual != frag_8x8_ty {
                let Some(c) = self.fpcast_frag(ctx, rewriter, frag, actual, frag_8x8_ty) else {
                    return Ok(());
                };
                c
            } else {
                frag
            };
            let (ar, ac0) = grid.coords(ctx, rewriter, r, c);
            let mask = grid.mask(ctx, rewriter, ar, ac0);
            // Pack the two lane elements into one vector value.
            let v0 = {
                let e = air::extract_elem(ctx, frag_ty, frag, c0);
                emit(ctx, rewriter, e)
            };
            let p0 = {
                let i = air::insert_elem(ctx, pair_frag_ty, undef_pair, v0, c0);
                emit(ctx, rewriter, i)
            };
            let v1 = {
                let e = air::extract_elem(ctx, frag_ty, frag, c1);
                emit(ctx, rewriter, e)
            };
            let p1 = {
                let i = air::insert_elem(ctx, pair_frag_ty, p0, v1, c1);
                emit(ctx, rewriter, i)
            };
            let pair = {
                let bc = air::cast(ctx, CastOpAttr::Bitcast, pair_int_ty, p1);
                emit(ctx, rewriter, bc)
            };
            let roff = air_bin_i32(ctx, rewriter, ar, rs, BinKind::Mul);
            let coff = air_bin_i32(ctx, rewriter, ac0, cs, BinKind::Mul);
            let off = air_bin_i32(ctx, rewriter, roff, coff, BinKind::Add);
            let g = air::GepOp::new(ctx, view.get_type(ctx), view, off, frag_ty);
            let gep = emit_op(ctx, rewriter, g);
            let iptr = {
                let bc = air::cast(ctx, CastOpAttr::Bitcast, iptr_ty, gep);
                emit(ctx, rewriter, bc)
            };
            let call = air::call(
                ctx,
                &emask_name,
                None,
                vec![iptr, pair, mask, m3_i16, bytes_i16],
            );
            rewriter.insert_operation(ctx, call);
        }
        self.erase_to_undef(ctx, rewriter, op)
    }
    /// Cast `<64 x from>` to `<64 x to>` when both are different float vectors.
    pub(crate) fn fpcast_frag(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: Value,
        from: TypeHandle,
        to: TypeHandle,
    ) -> Option<Value> {
        use pliron::r#type::TypedHandle;
        let elem = |h| {
            TypedHandle::<pliron_llvm::types::VectorType>::from_handle(h, ctx)
                .ok()
                .map(|v| v.deref(ctx).elem_type())
        };
        let (Some(from_elem), Some(to_elem)) = (elem(from), elem(to)) else {
            return Some(v);
        };
        if !t::is_float(ctx, from_elem) || !t::is_float(ctx, to_elem) {
            return Some(v);
        }
        fpcast(ctx, rewriter, v, from_elem, to_elem, to).or_else(|| {
            self.refuse_same_width_fpcast();
            None
        })
    }
    /// Whether this `mma_f` is the per-lane shape. `[1,1] x [1,N] -> [1,N]`.
    pub(crate) fn mma_is_per_lane(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        a: Value,
        b: Value,
        c: Value,
    ) -> bool {
        if [a, b, c]
            .iter()
            .any(|v| self.frags.fragments.contains(v) || self.frags.grids.contains_key(v))
        {
            return false;
        }
        let shape = |v: Value| -> Option<Vec<i64>> {
            let Some(WaxType::Tile(t)) = wax_operand_ty(ctx, operands_info, v) else {
                return None;
            };
            Some(t.shape.clone())
        };
        match (shape(a), shape(b), shape(c)) {
            (Some(sa), Some(sb), Some(sc)) => {
                sa == vec![1, 1] && sb.len() == 2 && sb[0] == 1 && sb == sc
            }
            _ => false,
        }
    }
    /// Widen a float value to f32, if needed.
    fn widen_to_f32(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: Value,
    ) -> Option<Value> {
        let from = v.get_type(ctx);
        let to = t::f32_ty(ctx);
        fpcast(ctx, rewriter, v, from, to, to).or_else(|| {
            self.refuse_same_width_fpcast();
            None
        })
    }
    /// `wax.mma_f` at `[1,1] x [1,N] -> [1,N]`. `C[0,lane] += A[0,0] * B[0,lane]`.
    ///
    /// f32 accumulator.
    pub(crate) fn rewrite_mma_f_per_lane(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        a: Value,
        b: Value,
        c: Value,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let f32_ty = t::f32_ty(ctx);
        let Some(a32) = self.widen_to_f32(ctx, rewriter, a) else {
            return Ok(());
        };
        let Some(b32) = self.widen_to_f32(ctx, rewriter, b) else {
            return Ok(());
        };
        let prod = air::bin(ctx, BinKind::FMul, f32_ty, a32, b32);
        let p = emit(ctx, rewriter, prod);
        let sum = air::bin(ctx, BinKind::FAdd, f32_ty, p, c);
        rewriter.insert_operation(ctx, sum);
        rewriter.replace_operation(ctx, op, sum);
        self.converted += 1;
        Ok(())
    }

    /// `wax.mma_f` -> one `@air.simdgroup_matrix_8x8_multiply_accumulate` per output
    /// fragment, accumulated over sub-tiles.
    pub(crate) fn rewrite_mma_f(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let code = Opcode::MmaF;
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        let (Some(&a), Some(&b), Some(&c)) = (operands.first(), operands.get(1), operands.get(2))
        else {
            return self.refuse(code, "mma needs three operands");
        };
        // The GEMV case,
        if self.mma_is_per_lane(ctx, operands_info, a, b, c) {
            return self.rewrite_mma_f_per_lane(ctx, rewriter, op, a, b, c);
        }
        let (Some((nr, ks)), Some((_, nc))) = (
            self.frag_dims_of_grid(ctx, operands_info, a),
            self.frag_dims_of_grid(ctx, operands_info, b),
        ) else {
            return self.refuse(code, "mma operands are not fragment tiles");
        };
        // f32 accumulator. A and B must share precision (f32/f16/bf16).
        let scalar_of = |v: Value| -> Option<ScalarType> {
            wax_operand_ty(ctx, operands_info, v)
                .as_ref()
                .and_then(crate::types::scalar_of)
        };
        let (Some(a_ty), Some(b_ty), Some(c_ty)) = (scalar_of(a), scalar_of(b), scalar_of(c))
        else {
            return self.refuse(code, "mma operands are not scalar tiles");
        };
        let mac_suf = match (a_ty, b_ty, c_ty) {
            (ScalarType::F32, ScalarType::F32, ScalarType::F32) => "v64f32.v64f32.v64f32.v64f32",
            (ScalarType::F16, ScalarType::F16, ScalarType::F32) => "v64f32.v64f16.v64f16.v64f32",
            (ScalarType::BF16, ScalarType::BF16, ScalarType::F32) => {
                "v64f32.v64bf16.v64bf16.v64f32"
            }
            _ => {
                return self.refuse(code, "requires f32 accumulator and A/B shared precision");
            }
        };
        let (Some(ga), Some(gb), Some(gc)) = (self.grid_of(a), self.grid_of(b), self.grid_of(c))
        else {
            return self.refuse(code, "mma operand has no fragment grid");
        };
        if ga.len() != nr * ks || gb.len() != ks * nc || gc.len() != nr * nc {
            return self.refuse(code, "fragment grid sizes disagree with tile dims");
        }
        let frag_ty = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        let mut out = Vec::with_capacity(nr * nc);
        // TODO: Prefer no magic strings.
        let mac_name = format!("@air.simdgroup_matrix_8x8_multiply_accumulate.{mac_suf}");
        for r in 0..nr {
            for col in 0..nc {
                let mut acc = gc[r * nc + col];
                for k in 0..ks {
                    let af = ga[r * ks + k];
                    let bf = gb[k * nc + col];
                    let call = air::call(ctx, &mac_name, Some(frag_ty), vec![af, bf, acc]);
                    acc = emit(ctx, rewriter, call);
                }
                out.push(acc);
            }
        }
        let Some(&rep) = out.first() else {
            return self.refuse(code, "mma produced no fragments");
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, out.clone());
        rewriter.replace_operation_with_values(ctx, op, vec![rep]);
        self.frags.grids.insert(rep, out);
        self.converted += 1;
        Ok(())
    }
    /// Fragment constant. `nr x nc` zero-initialised `<64 x float>` registers.
    // TODO: Support splat.
    // TODO: Support non-uniform constant via `masked_frag_preamble` row/col calculation.
    pub(crate) fn rewrite_fragment_constant(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        use wax_ir::dialect::attr_mirror::WaxAttr;
        let is_zero = matches!(
            op.deref(ctx).attributes.get::<WaxAttr>(&keys::ATTR_VALUE).map(|a| &a.0),
            Some(wax_ir::Attribute::DenseElements(de)) if de.data.iter().all(|&b| b == 0)
        );
        if !is_zero {
            return self.refuse(Opcode::Constant, "non-zero fragment constant");
        }
        let frag_ty = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        let mut frags = Vec::with_capacity(nr * nc);
        for _ in 0..nr * nc {
            let z = air::ZeroOp::new(ctx, frag_ty);
            frags.push(emit_op(ctx, rewriter, z));
        }
        let Some(&rep) = frags.first() else {
            return self.refuse(Opcode::Constant, "empty fragment grid");
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, frags.clone());
        rewriter.replace_operation_with_values(ctx, op, vec![rep]);
        self.frags.grids.insert(rep, frags);
        self.converted += 1;
        Ok(())
    }
}

/// Calculate `<2 x i64>` stride vector.
fn stride_vec2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    a: i64,
    da: Option<Value>,
    b: i64,
    db: Option<Value>,
) -> Value {
    use crate::air_dialect::attrs::CastOpAttr;
    let ty = t::vec_ty(ctx, 2, t::i64_ty(ctx));
    // All static path.
    if da.is_none() && db.is_none() {
        let c = air::const_vec_int(ctx, BITS_64, vec![a as u64, b as u64]);
        return emit_op(ctx, rewriter, c);
    }

    let undef = air::UndefOp::new(ctx, ty);
    let u = emit_op(ctx, rewriter, undef);

    let mut one = |v: Option<Value>, s: i64| -> Value {
        match v {
            Some(reg) => {
                let i64_ty = t::i64_ty(ctx);
                let z = air::cast(ctx, CastOpAttr::ZExt, i64_ty, reg);
                emit(ctx, rewriter, z)
            }
            None => {
                let c = air::const_int(ctx, BITS_64, s as u64);
                emit_op(ctx, rewriter, c)
            }
        }
    };
    // zext or const as needed.
    let e0 = one(da, a);
    let e1 = one(db, b);
    let zero = air_const_i32(ctx, 0);
    let i0 = air::insert_elem(ctx, ty, u, e0, zero);
    let t0 = emit(ctx, rewriter, i0);
    let one = air_const_i32(ctx, 1);
    let i1 = air::insert_elem(ctx, ty, t0, e1, one);
    emit(ctx, rewriter, i1)
}

/// Get element pointer of sub-tile `(r, c)`, `(r*row + c*col) * 8` elements into the base.
/// No offset means the sub-tile originates from the base itself with no gep.
fn fragment_sub_ptr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    f: &FragAddr,
    r: usize,
    c: usize,
) -> Value {
    let offset = (r as i64 * f.row_stride + c as i64 * f.col_stride) * 8;
    if offset == 0 {
        return f.base;
    }
    let offset_c = air::const_int(ctx, BITS_32, offset as u64);
    let offset_v = emit_op(ctx, rewriter, offset_c);
    let gep = air::GepOp::new(ctx, f.ptr_ty, f.base, offset_v, f.elem_ty);
    emit_op(ctx, rewriter, gep)
}
