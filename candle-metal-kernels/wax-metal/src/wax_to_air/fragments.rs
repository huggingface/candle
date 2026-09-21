//! The fragment arms: the load/store arms (plain and masked), `mmaf`, fragment
//! constants and row reduces, and the fragment grid and addressing machinery.

use super::*;

/// The shared prologue of the masked fragment load and store, emitted in exactly the
/// order `walk.rs`'s two separate lowers emit it: the thread-position check, the int/frag
/// type handles, the lane's 8x8 fragment coordinates, the stride pair, and the i16 mask
/// constants. The rest of the two arms diverges (the load has a portable path, the store
/// fpcasts its value), so this is where the sharing stops - the divergence mirrors the
/// CFG, whose `sgmma_load_masked` / `sgmma_store_masked` are also two functions.
pub(crate) struct MaskedFragPreamble {
    i32_ty: TypeHandle,
    i16_ty: TypeHandle,
    i1_ty: TypeHandle,
    frag_h: TypeHandle,
    pair_frag_h: TypeHandle,
    pair_int_h: TypeHandle,
    iptr_ty: TypeHandle,
    c0: pliron::value::Value,
    c1: pliron::value::Value,
    row: pliron::value::Value,
    col: pliron::value::Value,
    rs: pliron::value::Value,
    cs: pliron::value::Value,
    m3_i16: pliron::value::Value,
    m0_i16: pliron::value::Value,
    bytes_i16: pliron::value::Value,
}


/// A per-lane run `walk.rs` fuses into one wide load, planned before conversion: the head
/// op (the run's element 0, where the wide load is emitted), the members in slot order, and
/// the type the head's view carried before `fold_view` erased it.
/// The fragment access both arms share, built by `fragment_addr`: the view's element type,
/// the base pointer the block indices address (scaled by the FULL tile dim), and the two
/// stride vectors plus the zero vector the intrinsics take.
/// The inputs every fragment load and store arm addresses: the access kind (for the
/// diagnostics and the opcode it records), the view it goes through, the view's type
/// before `fold_view` erased it, and the index ops the element coordinates read from.
/// The four arms share it so the arm signatures stay under the clippy argument limit
/// and the two masked arms stay parallel to their plain counterparts in `walk.rs`.
#[derive(Clone, Copy)]
pub(crate) struct FragAccess<'a> {
    pub(crate) kind: &'a str,
    pub(crate) view: pliron::value::Value,
    pub(crate) view_ty: &'a wax_ir::types::Type,
    pub(crate) index_ops: &'a [pliron::value::Value],
}

#[derive(Clone)]
pub(crate) struct FragAddr<'a> {
    elem: &'a wax_ir::types::ScalarType,
    elem_scalar: LlType,
    base: pliron::value::Value,
    vec0: pliron::value::Value,
    vec1: pliron::value::Value,
    zeros: pliron::value::Value,
    ptr_ty: TypeHandle,
    elem_h: TypeHandle,
    row_s: i64,
    col_s: i64,
}

/// A fragment access's dynamic stride values, resolved BEFORE conversion: the view's
/// `MakeTensorView` is folded away by the time the access rewrites, and its stride operand -
/// after promotion - is the loaded value, which the pre-conversion walk can still read.
///
/// The pliron mirror of `walk.rs::resolve_stride` / `find_stride_val_for_pv`: see through
/// any `MakeStridedView` / `assume` wrapper to the partition view, then to its tensor view,
/// then into the DYN-STRIDE operand group indexed by how many dynamic strides precede the
/// dim. A dim whose stride is not a promoted block argument yields nothing, exactly as the
/// CFG's `find_stride_val_for_pv` refuses to hand one back.
pub(crate) fn collect_frag_strides(
    ctx: &Context,
    func: Ptr<Operation>,
) -> std::collections::HashMap<Ptr<Operation>, Vec<Option<pliron::value::Value>>> {
    use pliron::r#type::Typed;
    use wax_ir::types::Type as WT;
    let mut out: std::collections::HashMap<Ptr<Operation>, Vec<Option<pliron::value::Value>>> =
        std::collections::HashMap::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            let view_at = match try_opcode_of(ctx, op) {
                Some(Opcode::LoadViewTko) => 0,
                Some(Opcode::StoreViewTko) => 1,
                _ => continue,
            };
            let Some(view) = op.deref(ctx).operands().nth(view_at) else {
                continue;
            };
            let Some(WT::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx)) else {
                continue;
            };
            let strides = pv.tensor_view.strides.clone();
            if !strides.contains(&wax_ir::DYNAMIC) {
                continue;
            }
            let mut cur = match view.defining_entity() {
                pliron::value::DefiningEntity::Op(o) => o,
                _ => continue,
            };
            for _ in 0..4 {
                match try_opcode_of(ctx, cur) {
                    Some(Opcode::MakeStridedView) | Some(Opcode::Assume) => {
                        let Some(next) = cur.deref(ctx).operands().next() else {
                            break;
                        };
                        let pliron::value::DefiningEntity::Op(n) = next.defining_entity()
                        else {
                            break;
                        };
                        cur = n;
                    }
                    _ => break,
                }
            }
            if try_opcode_of(ctx, cur) != Some(Opcode::MakePartitionView) {
                continue;
            }
            let Some(tv) = cur.deref(ctx).operands().next() else {
                continue;
            };
            let pliron::value::DefiningEntity::Op(tv_def) = tv.defining_entity() else {
                continue;
            };
            if try_opcode_of(ctx, tv_def) != Some(Opcode::MakeTensorView) {
                continue;
            }
            let (n_base, n_dyn_shape, n_dyn_stride) = wax_operand_segments(ctx, tv_def);
            let operands: Vec<_> = tv_def.deref(ctx).operands().collect();
            let start = n_base + n_dyn_shape;
            let dyn_vals: Vec<pliron::value::Value> = operands
                .iter()
                .skip(start)
                .take(n_dyn_stride.min(operands.len().saturating_sub(start)))
                .copied()
                .map(|v| trace_through_assume(ctx, v))
                .collect();
            let mut di = 0;
            let mut per_dim: Vec<Option<pliron::value::Value>> = Vec::new();
            for &s in &strides {
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

/// The register scalar of a fragment at this element type.
pub(crate) fn sgmma_frag_scalar(e: &wax_ir::types::ScalarType) -> Option<LlType> {
    use wax_ir::types::ScalarType as S;
    Some(match e {
        S::F32 => LlType::Float,
        S::F16 => LlType::Half,
        S::BF16 => LlType::BFloat,
        _ => return None,
    })
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

impl WaxToAir {
    /// The `nr x nc` fragment grid a value stands for, if it is a fragment tile.
    ///
    /// A fragment tile is not one register: an `[16, 16]` f32 tile is a 2x2 grid of
    /// `<64 x float>` simdgroup matrices. pliron's `replace_operation_with_values` maps a
    /// result to exactly ONE value, so the grid lives in a side table and the result is bound
    /// to fragment 0 as a representative. That is the same shape `walk.rs` uses -
    /// `bind_grid`/`resolve_grid` - and the representative is never read as a whole: an op
    /// that consumes a grid value without going through this is refused, so a silent
    /// "operated on fragment 0 only" cannot happen.
    pub(crate) fn grid_of(&self, v: pliron::value::Value) -> Option<&Vec<pliron::value::Value>> {
        self.frags.grids.get(&v)
    }
    /// Fragment dimensions of a value the analysis says is a fragment.
    pub(crate) fn frag_dims(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        v: pliron::value::Value,
    ) -> Option<(usize, usize)> {
        use pliron::r#type::Typed;
        if !self.frags.fragments.contains(&v) {
            return None;
        }
        let ty = operands_info
            .lookup_most_recent_type(v)
            .unwrap_or_else(|| v.get_type(ctx));
        crate::types::sgmma_dims(&wax_type_of(ctx, ty)?)
    }
    /// Fragment dimensions of a value that stands for a fragment grid, however it got there:
    /// classified by the shared analysis (a value from the pre-conversion module), or
    /// registered in `grids` (a representative standing in for a grid - an mma result, a
    /// fragment load, a loop result or a body block argument).
    ///
    /// The grids table is the authoritative check for a CONSUMER: by the time an arm runs,
    /// its operands are usually the representatives the defining op registered, and those
    /// are new air values the fragment set does not know.
    pub(crate) fn frag_dims_of_grid(
        &self,
        ctx: &Context,
        operands_info: &OperandsInfo,
        v: pliron::value::Value,
    ) -> Option<(usize, usize)> {
        use pliron::r#type::Typed;
        if !self.frags.fragments.contains(&v) && !self.frags.grids.contains_key(&v) {
            return None;
        }
        let ty = operands_info
            .lookup_most_recent_type(v)
            .unwrap_or_else(|| v.get_type(ctx));
        crate::types::sgmma_dims(&wax_type_of(ctx, ty)?)
    }
    /// Per-row reduction of an SGMMA fragment: `walk.rs`'s `sgmma_row_reduce`. Each lane
    /// owns two elements of its row - register elements 0 and 1 of every row-0 sub-tile -
    /// so the walk folds those within the lane, then across the four lanes of the row
    /// group with `shuffle_xor` masks 1 and 8. No scratch, no barriers. The result is one
    /// scalar per lane - the reduce of the lane's row - which is exactly the per-row
    /// shape the `[rows]` result tile wants.
    pub(crate) fn fragment_row_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
        rc: ReduceCtx,
    ) -> Result<()> {
        let ReduceCtx { src, intrinsic, .. } = rc;
        let Intrinsic { base, suf } = intrinsic;
        use crate::air_dialect::attrs::BinOpAttr;
        use crate::air_dialect::attrs::FCmpPredAttr;
        type V = pliron::value::Value;
        let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, src) else {
            self.bail(Opcode::Reduce, "fragment reduce: dims unreadable");
            return Ok(());
        };
        let Some(grid) = self.grid_of(src) else {
            self.bail(Opcode::Reduce, "fragment reduce: no grid");
            return Ok(());
        };
        let f32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::Float);
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);

        // The fold is the COMBINE op, spelled the way `walk.rs`'s `scan_binop` spells it:
        // fadd / fmul, `call @air.fast_fmax` for max, `fcmp olt` + `select` for min.
        let fold = |ctx: &mut Context, rw: &mut DialectConversionRewriter, a: V, b: V| -> V {
            let o = match base {
                "@air.simd_product" => {
                    let m = air::BinOp::new(ctx, f32_h, a, b);
                    m.set_op(ctx, BinOpAttr::Mul);
                    m.get_operation()
                }
                "@air.simd_max" => {
                    let c = air::CallOp::new(
                        ctx,
                        &format!("@air.fast_fmax.{suf}"),
                        Some(f32_h),
                        vec![a, b],
                    );
                    air::set_fast(c.get_operation(), ctx);
                    c.get_operation()
                }
                "@air.simd_min" => {
                    let c = air::FCmpOp::new(ctx, i1_h, a, b);
                    c.set_pred(ctx, FCmpPredAttr::Olt);
                    rw.insert_operation(ctx, c.get_operation());
                    let cond = c.get_operation().deref(ctx).get_result(0);
                    let s = air::SelectOp::new(ctx, f32_h, cond, a, b);
                    s.get_operation()
                }
                _ => {
                    let d = air::BinOp::new(ctx, f32_h, a, b);
                    d.set_op(ctx, BinOpAttr::Add);
                    d.get_operation()
                }
            };
            rw.insert_operation(ctx, o);
            o.deref(ctx).get_result(0)
        };

        // NOTE: no within-lane register fold here. Reading the accumulator with
        // `extract_elem` is what the register walk did and it loses the odd rows (see below),
        // so nothing is read out of the fragment at all - it goes to memory whole.

        // The fragment is spilled to threadgroup scratch and reduced there, rather than
        // folded across lanes in registers.
        //
        // The register walk (extract elements 0 and 1, then `shuffle_xor` 1 and 8) is what
        // `walk.rs` did, and it does not work on this route: measured on device, elements 0
        // and 1 of the mma accumulator read ZERO for every lane with bit 2 set, so only the
        // even rows are present and the odd rows are lost. `gemm_8x8` proves the fragment
        // itself holds all 64 values - a matrix store reproduces them exactly - so the fault
        // is in reading a simdgroup matrix with `extract_elem`, which is not the sanctioned
        // accessor for one.
        //
        // So the value is written out by the intrinsic that is known to read it correctly,
        // and the reduce becomes an ordinary one over memory: store the 8x8 contiguously
        // (row stride 8, column stride 1 - the same `<8,8>`/`<1,8>` pair Apple's own
        // `simdgroup_store(C, scratch, 8)` emits), barrier, then every lane folds the eight
        // columns of the row its store slot wants. No lane-layout assumption survives.
        if nr != 1 || nc != 1 {
            self.bail(Opcode::Reduce, "fragment row reduce: only a single 8x8 fragment");
            return Ok(());
        }
        let tile0 = grid.first().copied().expect("grid checked non-empty");
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        const REGION: i64 = 32 * crate::wax_to_air::reduce::SCRATCH_PITCH;
        if i64::from(rt.scratch_n) < REGION + 64 {
            self.bail(Opcode::Reduce, "fragment row reduce: threadgroup scratch too small");
            return Ok(());
        }
        let base = {
            let off = rt.const_i(ctx, REGION);
            rt.scratch_gep(ctx, rewriter, off)
        };
        // `(<64 x float>, ptr addrspace(3), <2 x i64> row, <2 x i64> col, <2 x i64> zero)`.
        let v_row = self.stride_vec2(ctx, rewriter, 8, None, 8, None);
        let v_col = self.stride_vec2(ctx, rewriter, 1, None, 8, None);
        let v_zero = {
            let ty = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(2, LlType::I64));
            let z = air::ZeroOp::new(ctx, ty);
            rewriter.insert_operation(ctx, z.get_operation());
            z.get_operation().deref(ctx).get_result(0)
        };
        let st = air::CallOp::new(
            ctx,
            "@air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            None,
            vec![tile0, base, v_row, v_col, v_zero],
        );
        rewriter.insert_operation(ctx, st.get_operation());
        crate::wax_to_air::helpers::air_wg_barrier(ctx, rewriter, rt.i32_ty);

        // The `[8]` result hands element `s` to lanes `{s, s+8, s+16, s+24}`, so the row this
        // lane must produce is `lane & 7`. Four lanes compute the same row; they only read.
        let row = {
            let c7 = rt.const_i(ctx, 7);
            let a = air::BinOp::new(ctx, rt.i32_ty, rt.lane, c7);
            a.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::And);
            rewriter.insert_operation(ctx, a.get_operation());
            a.get_operation().deref(ctx).get_result(0)
        };
        let row_base = {
            let c8 = rt.const_i(ctx, 8);
            let m = air::BinOp::new(ctx, rt.i32_ty, row, c8);
            m.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Mul);
            rewriter.insert_operation(ctx, m.get_operation());
            let mv = m.get_operation().deref(ctx).get_result(0);
            let off = rt.const_i(ctx, REGION);
            let a = air::BinOp::new(ctx, rt.i32_ty, mv, off);
            a.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Add);
            rewriter.insert_operation(ctx, a.get_operation());
            a.get_operation().deref(ctx).get_result(0)
        };
        let mut res: Option<V> = None;
        for c in 0..8i64 {
            let flat = if c == 0 {
                row_base
            } else {
                let cc = rt.const_i(ctx, c);
                let a = air::BinOp::new(ctx, rt.i32_ty, row_base, cc);
                a.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Add);
                rewriter.insert_operation(ctx, a.get_operation());
                a.get_operation().deref(ctx).get_result(0)
            };
            let g = rt.scratch_gep(ctx, rewriter, flat);
            let ld = air::LoadOp::new(ctx, f32_h, g);
            air::set_align(ld.get_operation(), ctx, 4);
            rewriter.insert_operation(ctx, ld.get_operation());
            let v = ld.get_operation().deref(ctx).get_result(0);
            res = Some(match res {
                None => v,
                Some(acc) => fold(ctx, rewriter, acc, v),
            });
        }
        let Some(res) = res else {
            self.bail(Opcode::Reduce, "fragment row reduce: no columns");
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![res]);
        self.converted += 1;
        Ok(())
    }
    /// A `<2 x i64>` stride vector. Static strides only - a dynamic one is built with
    /// `insertelement` on the CFG path and this route refuses it rather than guess.
    pub(crate) fn stride_vec2(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        a: i64,
        da: Option<pliron::value::Value>,
        b: i64,
        db: Option<pliron::value::Value>,
    ) -> pliron::value::Value {
        use crate::air_dialect::attrs::CastOpAttr;
        let ty = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(2, LlType::I64));
        // `walk.rs::build_i64x2`: all static is one constant vector; a runtime element is a
        // zext inserted into an `undef`, element by element.
        if da.is_none() && db.is_none() {
            let c = air::ConstVecOp::new(ctx, ty, vec![a as u64, b as u64]);
            rewriter.insert_operation(ctx, c.get_operation());
            return c.get_operation().deref(ctx).get_result(0);
        }
        let one = |ctx: &mut Context,
                   rw: &mut DialectConversionRewriter,
                   v: Option<pliron::value::Value>,
                   s: i64| -> pliron::value::Value {
            match v {
                Some(reg) => {
                    let i64_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I64);
                    let z = air::CastOp::new(ctx, i64_ty, reg);
                    z.set_op(ctx, CastOpAttr::ZExt);
                    rw.insert_operation(ctx, z.get_operation());
                    z.get_operation().deref(ctx).get_result(0)
                }
                None => {
                    let i64_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I64);
                    let c = air::ConstBitsOp::new(ctx, i64_ty, s as u64);
                    rw.insert_operation(ctx, c.get_operation());
                    c.get_operation().deref(ctx).get_result(0)
                }
            }
        };
        let undef = air::UndefOp::new(ctx, ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        let u = undef.get_operation().deref(ctx).get_result(0);
        // Both elements FIRST, exactly as `build_i64x2` does - a zext precedes its insert.
        let e0 = one(ctx, rewriter, da, a);
        let e1 = one(ctx, rewriter, db, b);
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let zero = air_const_i32(ctx, i32_ty, 0);
        let i0 = air::InsertElemOp::new(ctx, ty, u, e0, zero);
        rewriter.insert_operation(ctx, i0.get_operation());
        let t0 = i0.get_operation().deref(ctx).get_result(0);
        let one = air_const_i32(ctx, i32_ty, 1);
        let i1 = air::InsertElemOp::new(ctx, ty, t0, e1, one);
        rewriter.insert_operation(ctx, i1.get_operation());
        i1.get_operation().deref(ctx).get_result(0)
    }
    /// The half of a fragment access the load and store arms share: the view checks, the
    /// base pointer the block indices address, and the two stride vectors plus the zero
    /// vector the intrinsics take.
    ///
    /// Emission order is the CFG's - base terms, then vec0, vec1, zeros - and both arms
    /// call this at the same spot, so the two stay in step. `kind` ("load" / "store") only
    /// feeds the bail messages: the checks are the same on both arms because the lane
    /// distribution is, and a disagreement between the arms would be a DIFF.
    pub(crate) fn fragment_addr<'a>(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        acc: FragAccess<'a>,
    ) -> Result<Option<FragAddr<'a>>> {
        let FragAccess { kind, view, view_ty, index_ops } = acc;
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        let code = if kind == "load" {
            Opcode::LoadViewTko
        } else {
            Opcode::StoreViewTko
        };
        let WT::PartitionView(pv) = view_ty else {
            self.bail(code, &format!("fragment {kind} needs a partition view"));
            return Ok(None);
        };
        // A padded fragment access is assembled elementwise on the CFG path; refuse when a
        // bound is actually collected - declared padding alone is not enough, because
        // `Tensor::partition` declares `padding::Zero` by default and nearly every view is
        // nominally padded without needing a mask.
        if let Some(bounds) = self.addr.bounds.get(&op)
            && !bounds.is_empty()
        {
            self.bail(code, &format!("masked fragment {kind} is not implemented"));
            return Ok(None);
        }
        let strides = &pv.tensor_view.strides;
        let (row_s, col_s) = (
            strides.first().copied().unwrap_or(8),
            strides.get(1).copied().unwrap_or(1),
        );
        // Dynamic strides, resolved before conversion: the runtime register a DYNAMIC
        // stride slot comes from. `walk.rs::resolve_stride` answers the same question.
        let dyns = self.addr.frag_strides.get(&op);
        let row_d = dyns.and_then(|v| v.first()).copied().flatten();
        let col_d = dyns.and_then(|v| v.get(1)).copied().flatten();
        if row_s == wax_ir::DYNAMIC && row_d.is_none() {
            self.bail(code, &format!("fragment {kind} with an unresolved dynamic row stride"));
            return Ok(None);
        }
        if col_s == wax_ir::DYNAMIC && col_d.is_none() {
            self.bail(code, &format!("fragment {kind} with an unresolved dynamic col stride"));
            return Ok(None);
        }
        let (tile_h, tile_w) = (
            i64::from(*pv.tile_shape.first().unwrap_or(&8)),
            i64::from(*pv.tile_shape.get(1).unwrap_or(&8)),
        );
        let Some(elem_scalar) = sgmma_frag_scalar(&pv.tensor_view.element_type) else {
            self.bail(code, "fragment element type is not f32/f16/bf16");
            return Ok(None);
        };
        let elem_h = crate::air_dialect::types::to_pliron(ctx, &elem_scalar);
        let ptr_ty = view.get_type(ctx);

        // Base pointer: the block indices scale by the FULL tile dim. `walk.rs`
        // `mul_stride`: a dynamic stride multiplies by the runtime register (then the
        // tile scale), a static one by the folded `stride * scale` constant.
        let base = if index_ops.is_empty() {
            view
        } else {
            let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
            let mut off = None;
            for (d, &idx) in index_ops.iter().enumerate().take(2) {
                let (s, t, sd) = if d == 0 {
                    (row_s, tile_h, row_d)
                } else {
                    (col_s, tile_w, col_d)
                };
                let term = match sd {
                    Some(reg) => {
                        let m = air::BinOp::new(ctx, i32_ty, idx, reg);
                        m.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Mul);
                        rewriter.insert_operation(ctx, m.get_operation());
                        let mv = m.get_operation().deref(ctx).get_result(0);
                        if t == 1 {
                            mv
                        } else {
                            let c = air::ConstBitsOp::new(ctx, i32_ty, t as u64);
                            rewriter.insert_operation(ctx, c.get_operation());
                            let cv = c.get_operation().deref(ctx).get_result(0);
                            let m2 = air::BinOp::new(ctx, i32_ty, mv, cv);
                            m2.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Mul);
                            rewriter.insert_operation(ctx, m2.get_operation());
                            m2.get_operation().deref(ctx).get_result(0)
                        }
                    }
                    None => {
                        let m = air::ConstBitsOp::new(ctx, i32_ty, (s * t) as u64);
                        rewriter.insert_operation(ctx, m.get_operation());
                        let mv = m.get_operation().deref(ctx).get_result(0);
                        let mul = air::BinOp::new(ctx, i32_ty, idx, mv);
                        mul.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Mul);
                        rewriter.insert_operation(ctx, mul.get_operation());
                        mul.get_operation().deref(ctx).get_result(0)
                    }
                };
                off = Some(match off {
                    None => term,
                    Some(a) => {
                        let add = air::BinOp::new(ctx, i32_ty, a, term);
                        add.set_op(ctx, crate::air_dialect::attrs::BinOpAttr::Add);
                        rewriter.insert_operation(ctx, add.get_operation());
                        add.get_operation().deref(ctx).get_result(0)
                    }
                });
            }
            match off {
                None => view,
                Some(o) => {
                    let g = air::GepOp::new(ctx, ptr_ty, view, o);
                    air::set_elem_ty(g.get_operation(), ctx, elem_h);
                    rewriter.insert_operation(ctx, g.get_operation());
                    g.get_operation().deref(ctx).get_result(0)
                }
            }
        };

        // The two stride vectors, built once for every sub-tile.
        let vec0 = self.stride_vec2(ctx, rewriter, row_s, row_d, 8, None);
        let vec1 = self.stride_vec2(ctx, rewriter, col_s, col_d, row_s, row_d);
        let zeros = {
            let ty = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(2, LlType::I64));
            let z = air::ZeroOp::new(ctx, ty);
            rewriter.insert_operation(ctx, z.get_operation());
            z.get_operation().deref(ctx).get_result(0)
        };

        Ok(Some(FragAddr {
            elem: &pv.tensor_view.element_type,
            elem_scalar,
            base,
            vec0,
            vec1,
            zeros,
            ptr_ty,
            elem_h,
            row_s,
            col_s,
        }))
    }
    /// The address of sub-tile `(r, c)`: `(r*row + c*col) * 8` elements into the base, the
    /// `(0, 0)` sub-tile being the base itself with no gep - the CFG's spelling.
    pub(crate) fn fragment_sub_ptr(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        f: &FragAddr,
        r: usize,
        c: usize,
    ) -> pliron::value::Value {
        let off = (r as i64 * f.row_s + c as i64 * f.col_s) * 8;
        if off == 0 {
            return f.base;
        }
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let o = air::ConstBitsOp::new(ctx, i32_ty, off as u64);
        rewriter.insert_operation(ctx, o.get_operation());
        let ov = o.get_operation().deref(ctx).get_result(0);
        let g = air::GepOp::new(ctx, f.ptr_ty, f.base, ov);
        air::set_elem_ty(g.get_operation(), ctx, f.elem_h);
        rewriter.insert_operation(ctx, g.get_operation());
        g.get_operation().deref(ctx).get_result(0)
    }

    /// `nr x nc` fragment loads off one base pointer, row-major.
    ///
    /// Mirrors `sgmma_load_single` / `sgmma_load_multi`: one
    /// `@air.simdgroup_matrix_8x8_load` per sub-tile, with the two stride vectors built once
    /// and shared, and sub-tile `(r, c)` at `(r*row + c*col)*8` elements into the base. The
    /// shared half of that - view checks, base pointer, stride vectors - is `fragment_addr`,
    /// which the store arm calls too, because the lane distribution is the same on both
    /// sides and a disagreement between the arms would be a DIFF.
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
        let Some(f) = self.fragment_addr(ctx, rewriter, op, acc)?
        else {
            return Ok(());
        };
        let load_suf = sgmma_load_suffix(f.elem).unwrap_or("v64f32.p1f32");
        let frag_ty =
            crate::air_dialect::types::to_pliron(ctx, &LlType::vec(64, f.elem_scalar.clone()));

        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let sub = self.fragment_sub_ptr(ctx, rewriter, &f, r, c);
                let call = air::CallOp::new(
                    ctx,
                    &format!("@air.simdgroup_matrix_8x8_load.{load_suf}"),
                    Some(frag_ty),
                    vec![sub, f.vec0, f.vec1, f.zeros],
                );
                rewriter.insert_operation(ctx, call.get_operation());
                frags.push(call.get_operation().deref(ctx).get_result(0));
            }
        }

        // The result stands for the whole grid; bind fragment 0 and record the rest.
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

    /// Bounds-masked fragment load, the `walk.rs::sgmma_load_masked` spelling: absolute
    /// per-element addressing off the raw pointer, with the block origin added to the
    /// lane coordinate. Two sub-spellings, decided exactly as the walk decides them:
    ///
    /// - PREDICATED (unit column stride, static): one emask vector load per sub-tile -
    ///   masked-off lanes read ZERO and their access is suppressed, so no clamp and no
    ///   select.
    /// - PORTABLE (anything else): clamp the address so an out-of-range lane cannot
    ///   fault, load, then select the pad value.
    ///
    /// Each lane owns the two elements at (row, col) and (row, col+1); the 64-register
    /// fragment is otherwise `undef`, which is what the unmasked intrinsic also yields
    /// for the elements the lane does not own.
    #[allow(clippy::too_many_arguments)]
    /// [`MaskedFragPreamble`]: the half of the masked fragment load and store the two
    /// lowers share, in `walk.rs`'s emission order.
    pub(crate) fn masked_frag_preamble(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        int_ty: LlType,
        frag_scalar: LlType,
        elem_bytes: i64,
        row_s: i64,
        col_s: i64,
    ) -> Option<MaskedFragPreamble> {
        use crate::air_dialect::attrs::BinOpAttr;
        let tp = match self.control.tp {
            Some(tp) => tp,
            None => {
                self.bail(code, "no thread position parameter");
                return None;
            }
        };
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i16_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I16);
        let i1_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let frag_h = crate::air_dialect::types::to_pliron(ctx, &frag_scalar);
        let pair_frag_h =
            crate::air_dialect::types::to_pliron(ctx, &LlType::vec(2, frag_scalar));
        let pair_int_h = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(2, int_ty.clone()));
        let iptr_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::device_ptr(int_ty));

        // The lane's 8x8 fragment coordinates, exactly in the walk's order.
        let c0 = air_const_i32(ctx, i32_ty, 0);
        let c1 = air_const_i32(ctx, i32_ty, 1);
        let c2 = air_const_i32(ctx, i32_ty, 2);
        let c3 = air_const_i32(ctx, i32_ty, 3);
        let c4 = air_const_i32(ctx, i32_ty, 4);
        let c7 = air_const_i32(ctx, i32_ty, 7);
        let lane = {
            let e = air::ExtractElemOp::new(ctx, i32_ty, tp, c0);
            rewriter.insert_operation(ctx, e.get_operation());
            e.get_operation().deref(ctx).get_result(0)
        };
        let x1 = air_bin_i32(ctx, rewriter, i32_ty, lane, c3, BinOpAttr::LShr);
        let x = air_bin_i32(ctx, rewriter, i32_ty, x1, c1, BinOpAttr::And);
        let y = air_bin_i32(ctx, rewriter, i32_ty, lane, c4, BinOpAttr::LShr);
        let loc = air_bin_i32(ctx, rewriter, i32_ty, lane, c7, BinOpAttr::And);
        let rloc = air_bin_i32(ctx, rewriter, i32_ty, loc, c1, BinOpAttr::LShr);
        let y4 = air_bin_i32(ctx, rewriter, i32_ty, y, c2, BinOpAttr::Shl);
        let row = air_bin_i32(ctx, rewriter, i32_ty, rloc, y4, BinOpAttr::Add);
        let lmod2 = air_bin_i32(ctx, rewriter, i32_ty, loc, c1, BinOpAttr::And);
        let clo = air_bin_i32(ctx, rewriter, i32_ty, lmod2, c1, BinOpAttr::Shl);
        let x4 = air_bin_i32(ctx, rewriter, i32_ty, x, c2, BinOpAttr::Shl);
        let col = air_bin_i32(ctx, rewriter, i32_ty, clo, x4, BinOpAttr::Add);

        // Static strides are immediates; the walk's `stride_i32` emits nothing for them.
        let (rs, cs) =
            match self.addr.frag_strides.get(&op).and_then(|v| v.first()).copied().flatten() {
                Some(reg) => (reg, air_const_i32(ctx, i32_ty, col_s)),
                None => (
                    air_const_i32(ctx, i32_ty, row_s),
                    air_const_i32(ctx, i32_ty, col_s),
                ),
            };
        let m3_i16 = {
            let c = air::ConstBitsOp::new(ctx, i16_ty, 3);
            c.get_operation().deref(ctx).get_result(0)
        };
        let m0_i16 = {
            let c = air::ConstBitsOp::new(ctx, i16_ty, 0);
            c.get_operation().deref(ctx).get_result(0)
        };
        let bytes_i16 = {
            let c = air::ConstBitsOp::new(ctx, i16_ty, elem_bytes as u64);
            c.get_operation().deref(ctx).get_result(0)
        };
        Some(MaskedFragPreamble {
            i32_ty,
            i16_ty,
            i1_ty,
            frag_h,
            pair_frag_h,
            pair_int_h,
            iptr_ty,
            c0,
            c1,
            row,
            col,
            rs,
            cs,
            m3_i16,
            m0_i16,
            bytes_i16,
        })
    }
    pub(crate) fn rewrite_fragment_load_masked(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let FragAccess { view, view_ty, index_ops, .. } = acc;
        use crate::air_dialect::attrs::{BinOpAttr, CastOpAttr, ICmpPredAttr};
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        let WT::PartitionView(pv) = view_ty else {
            self.bail(Opcode::LoadViewTko, "masked fragment load needs a partition view");
            return Ok(());
        };
        let Some(bounds) = self.addr.bounds.get(&op).cloned() else {
            self.bail(Opcode::LoadViewTko, "masked fragment load without bounds");
            return Ok(());
        };
        let Some(frag_scalar) = sgmma_frag_scalar(&pv.tensor_view.element_type) else {
            self.bail(Opcode::LoadViewTko, "fragment element type is not f32/f16/bf16");
            return Ok(());
        };
        let Some(pad_bits) = pv
            .padding_value
            .and_then(|p| padding_bits(&frag_scalar, p))
        else {
            self.bail(Opcode::LoadViewTko, "masked fragment load without a padding value");
            return Ok(());
        };
        let (int_ty, int_suf, elem_bytes) = match frag_scalar {
            LlType::Float => (LlType::I32, "i32", 4i64),
            LlType::Half | LlType::BFloat => (LlType::I16, "i16", 2i64),
            _ => {
                self.bail(Opcode::LoadViewTko, "masked fragment load element type");
                return Ok(());
            }
        };
        let (tile_h, tile_w) = (
            i64::from(*pv.tile_shape.first().unwrap_or(&8)),
            i64::from(*pv.tile_shape.get(1).unwrap_or(&8)),
        );
        let strides = &pv.tensor_view.strides;
        let (row_s, col_s) = (
            strides.first().copied().unwrap_or(8),
            strides.get(1).copied().unwrap_or(1),
        );
        // A dynamic ROW stride is fine: the mask math only needs the stride's runtime
        // register, which `frag_strides` resolved. A dynamic col stride needs the portable
        // path (not in the corpus yet).
        if col_s == wax_ir::DYNAMIC {
            self.bail(
                Opcode::LoadViewTko,
                "masked fragment load with a dynamic col stride",
            );
            return Ok(());
        }
        let Some(p) = self.masked_frag_preamble(
            ctx,
            rewriter,
            op,
            Opcode::LoadViewTko,
            int_ty,
            frag_scalar.clone(),
            elem_bytes,
            row_s,
            col_s,
        ) else {
            return Ok(());
        };
        let MaskedFragPreamble {
            i32_ty,
            i16_ty,
            i1_ty,
            frag_h,
            pair_frag_h,
            pair_int_h,
            iptr_ty,
            c0,
            c1,
            row,
            col,
            rs,
            cs,
            m3_i16,
            m0_i16,
            bytes_i16,
        } = p;

        let frag_vec_h =
            crate::air_dialect::types::to_pliron(ctx, &LlType::vec(64, frag_scalar.clone()));
        let predicated = col_s == 1;
        let align = match frag_scalar {
            LlType::Half | LlType::BFloat => 2,
            _ => 4,
        };
        let pad = {
            let c = air::ConstBitsOp::new(ctx, frag_h, pad_bits);
            c.get_operation().deref(ctx).get_result(0)
        };
        let undef_frag = {
            let u = air::UndefOp::new(ctx, frag_vec_h);
            rewriter.insert_operation(ctx, u.get_operation());
            u.get_operation().deref(ctx).get_result(0)
        };

        // The extents, materialized from the planned bounds.
        let bound_val = |ctx: &mut Context, i32_ty: TypeHandle, b: &Bound| -> Option<pliron::value::Value> {
            match b {
                Bound::Dyn(v) => Some(*v),
                Bound::Fixed(n) => Some(air_const_i32(ctx, i32_ty, *n)),
            }
        };
        let ext_r = bounds
            .iter()
            .find(|(d, _)| *d == 0)
            .and_then(|(_, b)| bound_val(ctx, i32_ty, b));
        let ext_c = bounds
            .iter()
            .find(|(d, _)| *d == 1)
            .and_then(|(_, b)| bound_val(ctx, i32_ty, b));

        // The block's own origin, added once rather than per sub-tile.
        let org_r = index_ops.first().map(|&b| {
            let t = air_const_i32(ctx, i32_ty, tile_h);
            air_bin_i32(ctx, rewriter, i32_ty, b, t, BinOpAttr::Mul)
        });
        let org_c = index_ops.get(1).map(|&b| {
            let t = air_const_i32(ctx, i32_ty, tile_w);
            air_bin_i32(ctx, rewriter, i32_ty, b, t, BinOpAttr::Mul)
        });

        let emask_name = format!("@llvm.agx3.load.with.emask.global.v2{int_suf}");
        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                // Absolute row of this lane's elements: block origin + sub-tile + lane.
                let mut ar = row;
                if r > 0 {
                    let t = air_const_i32(ctx, i32_ty, r as i64 * 8);
                    ar = air_bin_i32(ctx, rewriter, i32_ty, ar, t, BinOpAttr::Add);
                }
                if let Some(o) = org_r {
                    ar = air_bin_i32(ctx, rewriter, i32_ty, ar, o, BinOpAttr::Add);
                }
                let mut ac0 = col;
                if c > 0 {
                    let t = air_const_i32(ctx, i32_ty, c as i64 * 8);
                    ac0 = air_bin_i32(ctx, rewriter, i32_ty, ac0, t, BinOpAttr::Add);
                }
                if let Some(o) = org_c {
                    ac0 = air_bin_i32(ctx, rewriter, i32_ty, ac0, o, BinOpAttr::Add);
                }
                let frag = if predicated {
                    let roff = air_bin_i32(ctx, rewriter, i32_ty, ar, rs, BinOpAttr::Mul);
                    let coff = air_bin_i32(ctx, rewriter, i32_ty, ac0, cs, BinOpAttr::Mul);
                    let off = air_bin_i32(ctx, rewriter, i32_ty, roff, coff, BinOpAttr::Add);
                    let g = air::GepOp::new(ctx, view.get_type(ctx), view, off);
                    air::set_elem_ty(g.get_operation(), ctx, frag_h);
                    rewriter.insert_operation(ctx, g.get_operation());
                    let gep = g.get_operation().deref(ctx).get_result(0);
                    // `edgecheck` bounds the column axis (the pair's two halves); the row
                    // axis has no such instruction, so it selects the mask to 0.
                    let mut mask = match ext_c {
                        Some(ec) => {
                            let m = air::CallOp::new(
                                ctx,
                                "@llvm.agx3.edgecheck",
                                Some(i16_ty),
                                vec![ac0, c0, ec],
                            );
                            rewriter.insert_operation(ctx, m.get_operation());
                            let mv = m.get_operation().deref(ctx).get_result(0);
                            let a = air::BinOp::new(ctx, i16_ty, mv, m3_i16);
                            a.set_op(ctx, BinOpAttr::And);
                            rewriter.insert_operation(ctx, a.get_operation());
                            a.get_operation().deref(ctx).get_result(0)
                        }
                        None => m3_i16,
                    };
                    if let Some(er) = ext_r {
                        let cmp = air::ICmpOp::new(ctx, i1_ty, ar, er);
                        cmp.set_pred(ctx, ICmpPredAttr::Slt);
                        rewriter.insert_operation(ctx, cmp.get_operation());
                        let rok = cmp.get_operation().deref(ctx).get_result(0);
                        let s = air::SelectOp::new(ctx, i16_ty, rok, mask, m0_i16);
                        rewriter.insert_operation(ctx, s.get_operation());
                        mask = s.get_operation().deref(ctx).get_result(0);
                    }
                    let iptr = {
                        let bc = air::CastOp::new(ctx, iptr_ty, gep);
                        bc.set_op(ctx, CastOpAttr::Bitcast);
                        rewriter.insert_operation(ctx, bc.get_operation());
                        bc.get_operation().deref(ctx).get_result(0)
                    };
                    let vi = air::CallOp::new(
                        ctx,
                        &emask_name,
                        Some(pair_int_h),
                        vec![iptr, mask, m3_i16, bytes_i16],
                    );
                    rewriter.insert_operation(ctx, vi.get_operation());
                    let vi = vi.get_operation().deref(ctx).get_result(0);
                    let vf = {
                        let bc = air::CastOp::new(ctx, pair_frag_h, vi);
                        bc.set_op(ctx, CastOpAttr::Bitcast);
                        rewriter.insert_operation(ctx, bc.get_operation());
                        bc.get_operation().deref(ctx).get_result(0)
                    };
                    let v0 = {
                        let e = air::ExtractElemOp::new(ctx, frag_h, vf, c0);
                        rewriter.insert_operation(ctx, e.get_operation());
                        e.get_operation().deref(ctx).get_result(0)
                    };
                    let f0 = {
                        let i = air::InsertElemOp::new(ctx, frag_vec_h, undef_frag, v0, c0);
                        rewriter.insert_operation(ctx, i.get_operation());
                        i.get_operation().deref(ctx).get_result(0)
                    };
                    let v1 = {
                        let e = air::ExtractElemOp::new(ctx, frag_h, vf, c1);
                        rewriter.insert_operation(ctx, e.get_operation());
                        e.get_operation().deref(ctx).get_result(0)
                    };
                    {
                        let i = air::InsertElemOp::new(ctx, frag_vec_h, f0, v1, c1);
                        rewriter.insert_operation(ctx, i.get_operation());
                        i.get_operation().deref(ctx).get_result(0)
                    }
                } else {
                    // Portable spelling: clamp, load, select the pad value.
                    let rok = ext_r.map(|er| {
                        let cmp = air::ICmpOp::new(ctx, i1_ty, ar, er);
                        cmp.set_pred(ctx, ICmpPredAttr::Slt);
                        rewriter.insert_operation(ctx, cmp.get_operation());
                        cmp.get_operation().deref(ctx).get_result(0)
                    });
                    let ar_addr = match ext_r {
                        Some(er) => {
                            let last = air_bin_i32(ctx, rewriter, i32_ty, er, c1, BinOpAttr::Sub);
                            let lt = {
                                let cmp = air::ICmpOp::new(ctx, i1_ty, ar, er);
                                cmp.set_pred(ctx, ICmpPredAttr::Slt);
                                rewriter.insert_operation(ctx, cmp.get_operation());
                                cmp.get_operation().deref(ctx).get_result(0)
                            };
                            let s = air::SelectOp::new(ctx, i32_ty, lt, ar, last);
                            rewriter.insert_operation(ctx, s.get_operation());
                            s.get_operation().deref(ctx).get_result(0)
                        }
                        None => ar,
                    };
                    let roff = air_bin_i32(ctx, rewriter, i32_ty, ar_addr, rs, BinOpAttr::Mul);
                    let mut f2 = undef_frag;
                    for j in 0..2i64 {
                        let ac = if j == 0 {
                            ac0
                        } else {
                            air_bin_i32(ctx, rewriter, i32_ty, ac0, c1, BinOpAttr::Add)
                        };
                        let cok = ext_c.map(|ec| {
                            let cmp = air::ICmpOp::new(ctx, i1_ty, ac, ec);
                            cmp.set_pred(ctx, ICmpPredAttr::Slt);
                            rewriter.insert_operation(ctx, cmp.get_operation());
                            cmp.get_operation().deref(ctx).get_result(0)
                        });
                        let ac_addr = match ext_c {
                            Some(ec) => {
                                let last = air_bin_i32(ctx, rewriter, i32_ty, ec, c1, BinOpAttr::Sub);
                                let lt = {
                                    let cmp = air::ICmpOp::new(ctx, i1_ty, ac, ec);
                                    cmp.set_pred(ctx, ICmpPredAttr::Slt);
                                    rewriter.insert_operation(ctx, cmp.get_operation());
                                    cmp.get_operation().deref(ctx).get_result(0)
                                };
                                let s = air::SelectOp::new(ctx, i32_ty, lt, ac, last);
                                rewriter.insert_operation(ctx, s.get_operation());
                                s.get_operation().deref(ctx).get_result(0)
                            }
                            None => ac,
                        };
                        let coff = air_bin_i32(ctx, rewriter, i32_ty, ac_addr, cs, BinOpAttr::Mul);
                        let off = air_bin_i32(ctx, rewriter, i32_ty, roff, coff, BinOpAttr::Add);
                        let g = air::GepOp::new(ctx, view.get_type(ctx), view, off);
                        air::set_elem_ty(g.get_operation(), ctx, frag_h);
                        rewriter.insert_operation(ctx, g.get_operation());
                        let gep = g.get_operation().deref(ctx).get_result(0);
                        let ld = air::LoadOp::new(ctx, frag_h, gep);
                        air::set_align(ld.get_operation(), ctx, align);
                        rewriter.insert_operation(ctx, ld.get_operation());
                        let v = ld.get_operation().deref(ctx).get_result(0);
                        let v = match (rok, cok) {
                            (Some(a), Some(b)) => {
                                let p = {
                                    let a2 = air::BinOp::new(ctx, i1_ty, a, b);
                                    a2.set_op(ctx, BinOpAttr::And);
                                    rewriter.insert_operation(ctx, a2.get_operation());
                                    a2.get_operation().deref(ctx).get_result(0)
                                };
                                let s = air::SelectOp::new(ctx, frag_h, p, v, pad);
                                rewriter.insert_operation(ctx, s.get_operation());
                                s.get_operation().deref(ctx).get_result(0)
                            }
                            (Some(a), None) => {
                                let s = air::SelectOp::new(ctx, frag_h, a, v, pad);
                                rewriter.insert_operation(ctx, s.get_operation());
                                s.get_operation().deref(ctx).get_result(0)
                            }
                            (None, Some(b)) => {
                                let s = air::SelectOp::new(ctx, frag_h, b, v, pad);
                                rewriter.insert_operation(ctx, s.get_operation());
                                s.get_operation().deref(ctx).get_result(0)
                            }
                            (None, None) => v,
                        };
                        let idx = if j == 0 { c0 } else { c1 };
                        let i = air::InsertElemOp::new(ctx, frag_vec_h, f2, v, idx);
                        rewriter.insert_operation(ctx, i.get_operation());
                        f2 = i.get_operation().deref(ctx).get_result(0);
                    }
                    f2
                };
                frags.push(frag);
            }
        }

        // The result stands for the whole grid; bind fragment 0 and record the rest,
        // exactly as the unmasked load does.
        let res_tys = op_result_types(ctx, op);
        let Some(&rep) = frags.first() else {
            self.bail(Opcode::LoadViewTko, "masked fragment load produced no fragments");
            return Ok(());
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, frags.clone());
        self.finish_replacement(ctx, rewriter, op, &res_tys, vec![rep]);
        // Keyed by the representative too, so a consumer holding the replaced value finds it.
        self.frags.grids.insert(rep, frags);
        Ok(())
    }

    /// `nr x nc` fragment stores off one base pointer, row-major.
    ///
    /// Mirrors `sgmma_store_single` / `sgmma_store_multi`: one
    /// `@air.simdgroup_matrix_8x8_store` per sub-tile, with the two stride vectors built once
    /// and shared, and sub-tile `(r, c)` at `(r*row + c*col)*8` elements into the base. The
    /// fragment is cast to the destination element first: the mma accumulator is always
    /// `<64 x float>` but the store intrinsic is typed by the buffer element
    /// (`sgmma_frag_to`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rewrite_fragment_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        value: pliron::value::Value,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let Some(f) = self.fragment_addr(ctx, rewriter, op, acc)?
        else {
            return Ok(());
        };
        let store_suf = sgmma_load_suffix(f.elem).unwrap_or("v64f32.p1f32");

        let Some(frags) = self.grid_of(value).cloned() else {
            self.bail(Opcode::StoreViewTko, "fragment store value has no fragment grid");
            return Ok(());
        };
        if frags.len() != nr * nc {
            self.bail(Opcode::StoreViewTko, "fragment store grid size disagrees with the tile dims");
            return Ok(());
        }
        use pliron::r#type::Typed;
        for r in 0..nr {
            for c in 0..nc {
                let sub = self.fragment_sub_ptr(ctx, rewriter, &f, r, c);
                // The accumulator is `<64 x float>`; the store is typed by the buffer
                // element, so cast when they differ (`sgmma_frag_to`).
                let want = LlType::vec(64, f.elem_scalar.clone());
                let have = crate::air_dialect::types::from_pliron(
                    ctx,
                    frags[r * nc + c].get_type(ctx),
                );
                let frag = match have {
                    Some(h) if h != want => {
                        self.fpcast_frag(ctx, rewriter, frags[r * nc + c], &h, &want)
                    }
                    _ => frags[r * nc + c],
                };                let call = air::CallOp::new(
                    ctx,
                    &format!("@air.simdgroup_matrix_8x8_store.{store_suf}"),
                    None,
                    vec![frag, sub, f.vec0, f.vec1, f.zeros],
                );
                rewriter.insert_operation(ctx, call.get_operation());
            }
        }

        // The op's results are ordering tokens; AGX has nothing to order against.
        self.erase_to_undef(ctx, rewriter, op)
    }

    /// Masked multi-fragment store: the AGX emask spelling of
    /// `walk.rs::sgmma_store_masked`. One masked vector store per fragment per owned
    /// element pair, addressed in ABSOLUTE coordinates off the raw pointer - the block
    /// origin is added to the lane coordinate rather than folded into a base GEP.
    ///
    /// The 8x8 fragment's lane distribution, the same one the walk derives: x =
    /// (lane>>3)&1, y = lane>>4, loc = lane&7; each lane owns the two elements at
    /// (row, col) and (row, col+1), held at vector indices 0 and 1.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rewrite_fragment_store_masked(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        value: pliron::value::Value,
        acc: FragAccess,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let FragAccess { view, view_ty, index_ops, .. } = acc;
        use crate::air_dialect::attrs::{BinOpAttr, CastOpAttr, ICmpPredAttr};
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        let WT::PartitionView(pv) = view_ty else {
            self.bail(Opcode::StoreViewTko, "masked fragment store needs a partition view");
            return Ok(());
        };
        let Some(bounds) = self.addr.bounds.get(&op).cloned() else {
            self.bail(Opcode::StoreViewTko, "masked fragment store without bounds");
            return Ok(());
        };
        let Some(frag_scalar) = sgmma_frag_scalar(&pv.tensor_view.element_type) else {
            self.bail(Opcode::StoreViewTko, "fragment element type is not f32/f16/bf16");
            return Ok(());
        };
        // The emask store is an INTEGER op - floats bitcast through it - and it
        // exists only for 32-bit and 16-bit elements.
        let (int_ty, int_suf, elem_bytes) = match frag_scalar {
            LlType::Float => (LlType::I32, "i32", 4i64),
            LlType::Half | LlType::BFloat => (LlType::I16, "i16", 2i64),
            _ => {
                self.bail(Opcode::StoreViewTko, "masked fragment store element type");
                return Ok(());
            }
        };
        let Some(frags) = self.grid_of(value).cloned() else {
            self.bail(Opcode::StoreViewTko, "fragment store value has no fragment grid");
            return Ok(());
        };
        if frags.len() != nr * nc {
            self.bail(
                Opcode::StoreViewTko,
                "fragment store grid size disagrees with the tile dims",
            );
            return Ok(());
        }
        let (tile_h, tile_w) = (
            i64::from(*pv.tile_shape.first().unwrap_or(&8)),
            i64::from(*pv.tile_shape.get(1).unwrap_or(&8)),
        );
        let strides = &pv.tensor_view.strides;
        let (row_s, col_s) = (
            strides.first().copied().unwrap_or(8),
            strides.get(1).copied().unwrap_or(1),
        );
        // A dynamic ROW stride is fine: the mask math only needs the stride's runtime
        // register, which `frag_strides` resolved. A dynamic col stride needs the portable
        // path (not in the corpus yet).
        if col_s == wax_ir::DYNAMIC {
            self.bail(
                Opcode::StoreViewTko,
                "masked fragment store with a dynamic col stride",
            );
            return Ok(());
        }
        // Only the predicated spelling is implemented: a non-unit column stride falls to
        // the walk's portable per-element CondBr pairs.
        if col_s != 1 {
            self.bail(
                Opcode::StoreViewTko,
                "masked fragment store with a non-unit column stride",
            );
            return Ok(());
        }
        let Some(p) = self.masked_frag_preamble(
            ctx,
            rewriter,
            op,
            Opcode::StoreViewTko,
            int_ty,
            frag_scalar.clone(),
            elem_bytes,
            row_s,
            col_s,
        ) else {
            return Ok(());
        };
        let MaskedFragPreamble {
            i32_ty,
            i16_ty,
            i1_ty,
            frag_h,
            pair_frag_h,
            pair_int_h,
            iptr_ty,
            c0,
            c1,
            row,
            col,
            rs,
            cs,
            m3_i16,
            m0_i16,
            bytes_i16,
        } = p;

        // The extents, materialized from the planned bounds (a fixed ragged extent is a
        // constant, a dynamic one the promoted parameter the prologue loaded).
        let ext_r = bounds
            .iter()
            .find(|(d, _)| *d == 0)
            .map(|(_, b)| match b {
                Bound::Dyn(v) => *v,
                Bound::Fixed(n) => air_const_i32(ctx, i32_ty, *n),
            });
        let ext_c = bounds
            .iter()
            .find(|(d, _)| *d == 1)
            .map(|(_, b)| match b {
                Bound::Dyn(v) => *v,
                Bound::Fixed(n) => air_const_i32(ctx, i32_ty, *n),
            });

        // The block's own origin, added once rather than per sub-tile.
        let org_r = index_ops.first().map(|&b| {
            let t = air_const_i32(ctx, i32_ty, tile_h);
            air_bin_i32(ctx, rewriter, i32_ty, b, t, BinOpAttr::Mul)
        });
        let org_c = index_ops.get(1).map(|&b| {
            let t = air_const_i32(ctx, i32_ty, tile_w);
            air_bin_i32(ctx, rewriter, i32_ty, b, t, BinOpAttr::Mul)
        });

        // The pair's `undef` base: a constant, so its stream position is irrelevant.
        let undef_pair = {
            let u = air::UndefOp::new(ctx, pair_frag_h);
            rewriter.insert_operation(ctx, u.get_operation());
            u.get_operation().deref(ctx).get_result(0)
        };

        let want_frag = LlType::vec(64, frag_scalar.clone());
        let emask_name = format!("@llvm.agx3.store.with.emask.global.v2{int_suf}");
        for (n, &frag) in frags.iter().enumerate() {
            let (r, c) = (n / nc, n % nc);
            // The accumulator is `<64 x float>`; the destination element may be narrower
            // (`sgmma_frag_to`), cast per fragment before anything else.
            let have = crate::air_dialect::types::from_pliron(ctx, frag.get_type(ctx));
            let frag = match have {
                Some(h) if h != want_frag => self.fpcast_frag(ctx, rewriter, frag, &h, &want_frag),
                _ => frag,
            };
            let mut ar = row;
            if r > 0 {
                let t = air_const_i32(ctx, i32_ty, r as i64 * 8);
                ar = air_bin_i32(ctx, rewriter, i32_ty, ar, t, BinOpAttr::Add);
            }
            if let Some(o) = org_r {
                ar = air_bin_i32(ctx, rewriter, i32_ty, ar, o, BinOpAttr::Add);
            }
            let mut ac0 = col;
            if c > 0 {
                let t = air_const_i32(ctx, i32_ty, c as i64 * 8);
                ac0 = air_bin_i32(ctx, rewriter, i32_ty, ac0, t, BinOpAttr::Add);
            }
            if let Some(o) = org_c {
                ac0 = air_bin_i32(ctx, rewriter, i32_ty, ac0, o, BinOpAttr::Add);
            }
            // The mask: `edgecheck` bounds the column axis (the pair's two halves),
            // zeroed out entirely when the row is beyond the extent.
            let mut mask = match ext_c {
                Some(ec) => {
                    let m = air::CallOp::new(
                        ctx,
                        "@llvm.agx3.edgecheck",
                        Some(i16_ty),
                        vec![ac0, c0, ec],
                    );
                    rewriter.insert_operation(ctx, m.get_operation());
                    let mv = m.get_operation().deref(ctx).get_result(0);
                    let a = air::BinOp::new(ctx, i16_ty, mv, m3_i16);
                    a.set_op(ctx, BinOpAttr::And);
                    rewriter.insert_operation(ctx, a.get_operation());
                    a.get_operation().deref(ctx).get_result(0)
                }
                None => m3_i16,
            };
            if let Some(er) = ext_r {
                let cmp = air::ICmpOp::new(ctx, i1_ty, ar, er);
                cmp.set_pred(ctx, ICmpPredAttr::Slt);
                rewriter.insert_operation(ctx, cmp.get_operation());
                let rok = cmp.get_operation().deref(ctx).get_result(0);
                let s = air::SelectOp::new(ctx, i16_ty, rok, mask, m0_i16);
                rewriter.insert_operation(ctx, s.get_operation());
                mask = s.get_operation().deref(ctx).get_result(0);
            }
            // Pack the lane's two owned elements into one vector value.
            let v0 = {
                let e = air::ExtractElemOp::new(ctx, frag_h, frag, c0);
                rewriter.insert_operation(ctx, e.get_operation());
                e.get_operation().deref(ctx).get_result(0)
            };
            let p0 = {
                let i = air::InsertElemOp::new(ctx, pair_frag_h, undef_pair, v0, c0);
                rewriter.insert_operation(ctx, i.get_operation());
                i.get_operation().deref(ctx).get_result(0)
            };
            let v1 = {
                let e = air::ExtractElemOp::new(ctx, frag_h, frag, c1);
                rewriter.insert_operation(ctx, e.get_operation());
                e.get_operation().deref(ctx).get_result(0)
            };
            let p1 = {
                let i = air::InsertElemOp::new(ctx, pair_frag_h, p0, v1, c1);
                rewriter.insert_operation(ctx, i.get_operation());
                i.get_operation().deref(ctx).get_result(0)
            };
            let pair = {
                let bc = air::CastOp::new(ctx, pair_int_h, p1);
                bc.set_op(ctx, CastOpAttr::Bitcast);
                rewriter.insert_operation(ctx, bc.get_operation());
                bc.get_operation().deref(ctx).get_result(0)
            };
            let roff = air_bin_i32(ctx, rewriter, i32_ty, ar, rs, BinOpAttr::Mul);
            let coff = air_bin_i32(ctx, rewriter, i32_ty, ac0, cs, BinOpAttr::Mul);
            let off = air_bin_i32(ctx, rewriter, i32_ty, roff, coff, BinOpAttr::Add);
            let g = air::GepOp::new(ctx, view.get_type(ctx), view, off);
            air::set_elem_ty(g.get_operation(), ctx, frag_h);
            rewriter.insert_operation(ctx, g.get_operation());
            let gep = g.get_operation().deref(ctx).get_result(0);
            let iptr = {
                let bc = air::CastOp::new(ctx, iptr_ty, gep);
                bc.set_op(ctx, CastOpAttr::Bitcast);
                rewriter.insert_operation(ctx, bc.get_operation());
                bc.get_operation().deref(ctx).get_result(0)
            };
            let call = air::CallOp::new(
                ctx,
                &emask_name,
                None,
                vec![iptr, pair, mask, m3_i16, bytes_i16],
            );
            rewriter.insert_operation(ctx, call.get_operation());
        }

        // The op's results are ordering tokens; AGX has nothing to order against.
        self.erase_to_undef(ctx, rewriter, op)
    }
    /// `sgmma_frag_to` as a cast on values: `<64 x have>` to `<64 x want>` when both are
    /// float vectors and the elements differ. Identity when they already agree, which is the
    /// common case and emits nothing.
    pub(crate) fn fpcast_frag(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        v: pliron::value::Value,
        have: &LlType,
        want: &LlType,
    ) -> pliron::value::Value {
        use crate::air_dialect::attrs::CastOpAttr;
        let (LlType::Vec(_, have_elem), LlType::Vec(_, want_elem)) = (have, want) else {
            return v;
        };
        if !have_elem.is_float() || !want_elem.is_float() || have_elem == want_elem {
            return v;
        }
        let to_ty = crate::air_dialect::types::to_pliron(ctx, want);
        let kind = if scalar_bits(have_elem) > scalar_bits(want_elem) {
            CastOpAttr::FPTrunc
        } else {
            CastOpAttr::FPExt
        };
        let c = air::CastOp::new(ctx, to_ty, v);
        c.set_op(ctx, kind);
        rewriter.insert_operation(ctx, c.get_operation());
        c.get_operation().deref(ctx).get_result(0)
    }
    /// `wax.mma_f` -> one `@air.simdgroup_matrix_8x8_multiply_accumulate` per output
    /// fragment, accumulated over the K sub-tiles.
    ///
    /// `C[r,col] = sum_k MAC(A[r,k], B[k,col], acc)`, with A an `nr_a x nc_a` grid, B
    /// `nc_a x nc_b` and C `nr_a x nc_b`, all row-major. Mirrors `mma_f`.
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
            self.bail(code, "mma needs three operands");
            return Ok(());
        };
        let (Some((nr_a, nc_a)), Some((_, nc_b))) = (
            self.frag_dims_of_grid(ctx, operands_info, a),
            self.frag_dims_of_grid(ctx, operands_info, b),
        ) else {
            self.bail(code, "mma operands are not fragment tiles");
            return Ok(());
        };
        // The accumulator must be f32 and A and B must share precision (f32/f16/bf16),
        // which is what `mma_reg_class` enforces. The fragment registers are
        // `<64 x float>` in every class; only the call name moves with the precision.
        let elem_of = |v: pliron::value::Value| -> Option<ScalarType> {
            use pliron::r#type::Typed;
            let ty = operands_info
                .lookup_most_recent_type(v)
                .unwrap_or_else(|| v.get_type(ctx));
            let ll = wax_type_of(ctx, ty)?;
            Some(crate::types::scalar_of(&ll))
        };
        let (a_elem, b_elem, c_elem) = match (elem_of(a), elem_of(b), elem_of(c)) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                self.bail(code, "mma operands are not scalar tiles");
                return Ok(());
            }
        };
        let mac_suf = match (&a_elem, &b_elem, &c_elem) {
            (ScalarType::F32, ScalarType::F32, ScalarType::F32) => {
                "v64f32.v64f32.v64f32.v64f32"
            }
            (ScalarType::F16, ScalarType::F16, ScalarType::F32) => {
                "v64f32.v64f16.v64f16.v64f32"
            }
            (ScalarType::BF16, ScalarType::BF16, ScalarType::F32) => {
                "v64f32.v64bf16.v64bf16.v64f32"
            }
            _ => {
                self.bail(code, "mma accumulator must be f32 and A/B must share precision");
                return Ok(());
            }
        };
        let (Some(ga), Some(gb), Some(gc)) = (
            self.grid_of(a).cloned(),
            self.grid_of(b).cloned(),
            self.grid_of(c).cloned(),
        ) else {
            self.bail(code, "an mma operand has no fragment grid");
            return Ok(());
        };
        let (nr, nc, ks) = (nr_a, nc_b, nc_a);
        if ga.len() != nr_a * nc_a || gb.len() != nc_a * nc_b || gc.len() != nr * nc {
            self.bail(code, "fragment grid sizes disagree with the tile dims");
            return Ok(());
        }
        let frag_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(64, LlType::Float));
        let mut out = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for col in 0..nc {
                let mut acc = gc[r * nc + col];
                for k in 0..ks {
                    let af = ga[r * nc_a + k];
                    let bf = gb[k * nc_b + col];
                    let call = air::CallOp::new(
                        ctx,
                        &format!("@air.simdgroup_matrix_8x8_multiply_accumulate.{mac_suf}"),
                        Some(frag_ty),
                        vec![af, bf, acc],
                    );
                    rewriter.insert_operation(ctx, call.get_operation());
                    acc = call.get_operation().deref(ctx).get_result(0);
                }
                out.push(acc);
            }
        }
        let Some(&rep) = out.first() else {
            self.bail(code, "mma produced no fragments");
            return Ok(());
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, out.clone());
        rewriter.replace_operation_with_values(ctx, op, vec![rep]);
        self.frags.grids.insert(rep, out);
        self.converted += 1;
        Ok(())
    }
    /// A fragment constant: `nr x nc` zero-initialised `<64 x float>` registers.
    ///
    /// Zero only. A non-zero multi-fragment splat would have to know the lane distribution to
    /// place its elements, and `walk.rs` declines it for the same reason.
    pub(crate) fn rewrite_fragment_constant(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        nr: usize,
        nc: usize,
    ) -> Result<()> {
        let is_zero = op
            .deref(ctx)
            .attributes
            .0
            .iter()
            .find_map(|(k, a)| {
                let a = a.downcast_ref::<wax_ir::dialect::attr_mirror::WaxAttr>()?;
                match (&k.to_string()[..], &a.0) {
                    ("value", wax_ir::Attribute::DenseElements(de)) => {
                        Some(de.data.iter().all(|&b| b == 0))
                    }
                    _ => None,
                }
            })
            .unwrap_or(false);
        if !is_zero {
            self.bail(Opcode::Constant, "non-zero fragment constant");
            return Ok(());
        }
        let frag_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(64, LlType::Float));
        let mut frags = Vec::with_capacity(nr * nc);
        for _ in 0..nr * nc {
            let z = air::ZeroOp::new(ctx, frag_ty);
            rewriter.insert_operation(ctx, z.get_operation());
            frags.push(z.get_operation().deref(ctx).get_result(0));
        }
        let Some(&rep) = frags.first() else {
            self.bail(Opcode::Constant, "empty fragment grid");
            return Ok(());
        };
        let result0 = op.deref(ctx).get_result(0);
        self.frags.grids.insert(result0, frags.clone());
        rewriter.replace_operation_with_values(ctx, op, vec![rep]);
        self.frags.grids.insert(rep, frags);
        self.converted += 1;
        Ok(())
    }
}
