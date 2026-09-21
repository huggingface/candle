//! The reduce and scan arms: column, batched and hierarchical reduces, the
//! inclusive scan, and the threadgroup scratch they share.

use super::*;

/// A run of adjacent `[32, 32]` column reduces `walk.rs` lowers as one shared scratch
/// round trip: the head (where the batch is emitted), the members in slot order, and each
/// member's compute/result types and combine intrinsic, read pre-conversion.
pub(crate) struct ReduceBatch {
    head: Ptr<Operation>,
    members: Vec<Ptr<Operation>>,
    info: Vec<(&'static str, String, LlType, LlType, u32)>,
}

/// The threadgroup scratch's row pitch, in floats: 32 lanes + 1 to avoid the bank
/// conflict. `walk.rs`'s `tg_row_pitch(32)`; both column-reduce emitters and the scratch
/// sizing read the same value.
/// What the reduce emitters share: the source value and the combine/reduce intrinsic's
/// spelling and types. Built once per `wax.reduce` op in `rewrite_reduce`, so the three
/// emitters take it instead of the six individual arguments.
/// The combine/reduce intrinsic's spelling: the op and the type suffix, which appear
/// together at every call site.
#[derive(Clone, Copy)]
pub(crate) struct Intrinsic {
    pub(crate) base: &'static str,
    pub(crate) suf: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct ReduceCtx<'a> {
    pub(crate) src: pliron::value::Value,
    pub(crate) intrinsic: Intrinsic,
    pub(crate) cty: &'a LlType,
    pub(crate) rty: &'a LlType,
}

pub(crate) const SCRATCH_PITCH: i64 = 33;

/// A constant-pool entry (not an instruction): created detached, the emitter finds it
/// through its uses, so its creation position is irrelevant to the emitted order.
pub(crate) fn air_const_i32(ctx: &mut Context, i32_ty: TypeHandle, v: i64) -> pliron::value::Value {
    let c = air::ConstBitsOp::new(ctx, i32_ty, v as u64);
    c.get_operation().deref(ctx).get_result(0)
}

/// The column reduce's shared scratch round trip, emitted in `walk.rs`'s order by
/// `scratch_round_trip`: the wait for the last reader (not for the kernel's first writer),
/// the tid/group/lane split, and the scratch reference. Carries the four operations both
/// emitters use, so `emit_reduce_batch` and `column_reduce` spell them identically.
pub(crate) struct ScratchRoundTrip {
    pub(crate) f32_h: TypeHandle,
    pub(crate) i32_ty: TypeHandle,
    ptr_h: TypeHandle,
    pub(crate) scratch_n: u32,
    zero: pliron::value::Value,
    grp: pliron::value::Value,
    pub(crate) lane: pliron::value::Value,
    c_pitch: pliron::value::Value,
    base_ptr: pliron::value::Value,
}

impl ScratchRoundTrip {
    pub(crate) fn const_i(&self, ctx: &mut Context, v: i64) -> pliron::value::Value {
        air_const_i32(ctx, self.i32_ty, v)
    }

    pub(crate) fn bin(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        a: pliron::value::Value,
        b: pliron::value::Value,
        k: crate::air_dialect::attrs::BinOpAttr,
    ) -> pliron::value::Value {
        air_bin_i32(ctx, rw, self.i32_ty, a, b, k)
    }

    pub(crate) fn barrier(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) {
        air_wg_barrier(ctx, rw, self.i32_ty)
    }

    /// A gep into `@_tg_user_scratch` at `idx` float units in.
    pub(crate) fn scratch_gep(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        idx: pliron::value::Value,
    ) -> pliron::value::Value {
        let g = air::GepArrayOp::new(ctx, self.ptr_h, self.base_ptr, idx);
        air::set_elem_ty(g.get_operation(), ctx, self.f32_h);
        air::set_count(g.get_operation(), ctx, self.scratch_n);
        rw.insert_operation(ctx, g.get_operation());
        g.get_operation().deref(ctx).get_result(0)
    }
}

pub(crate) fn record_column_reduce_sources(
    ctx: &Context,
    func: Ptr<Operation>,
) -> std::collections::HashMap<Ptr<Operation>, pliron::value::Value> {
    use pliron::r#type::Typed;
    let mut out = std::collections::HashMap::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if try_opcode_of(ctx, op) != Some(Opcode::Reduce) {
                continue;
            }
            let Some(src) = op.deref(ctx).operands().next() else {
                continue;
            };
            if reduce_dim_of(ctx, op) != Some(0) {
                continue;
            }
            let is_wide_row = matches!(
                wax_type_of(ctx, src.get_type(ctx)),
                Some(WaxType::Tile(t)) if t.shape.len() == 2 && t.shape[1] == 32 && t.shape[0] > 1
            );
            if is_wide_row {
                out.insert(op, src);
            }
        }
    });
    out
}

/// The `dim` attribute of a reduce, if it is an integer one.
pub(crate) fn reduce_dim_of(ctx: &Context, op: Ptr<Operation>) -> Option<i64> {
    use wax_ir::attr::Attribute as WA;
    use wax_ir::dialect::attr_mirror::WaxAttr;
    let k = "dim".try_into().ok()?;
    match &op.deref(ctx).attributes.get::<WaxAttr>(&k)?.0 {
        WA::Integer(i, _) => Some(*i),
        WA::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// The identity value a cross-SIMD reduction fills unused lanes/slots with, as the raw
/// bit pattern at the reduced type's width: sum -> 0, product -> 1, max -> -inf, min ->
/// +inf. `walk.rs`'s `reduce_identity` carries the text form and the bits separately;
/// this route needs only the bits.
pub(crate) fn reduce_identity_bits(base: &str, suf: &str) -> u64 {
    let is_max = base == "@air.simd_max";
    let is_min = base == "@air.simd_min";
    let is_prod = base == "@air.simd_product";
    match suf {
        "f16" if is_max => 0xFC00,
        "f16" if is_min => 0x7C00,
        "bf16" if is_max => 0xFF80,
        "bf16" if is_min => 0x7F80,
        _ if is_max => 0xFF80_0000,
        _ if is_min => 0x7F80_0000,
        _ if is_prod => 0x3F80_0000,
        _ => 0,
    }
}

/// The element count of `@_tg_user_scratch` for this kernel: its `tg_scratch_rows`
/// attribute times the padded pitch, or the default. `walk.rs` computes the identical
/// value (`tg_scratch_rows * tg_row_pitch(32)` / `TG_SCRATCH_FLOATS_DEFAULT`); the gep
/// element type and the global declaration both read it, so the two paths declare
/// byte-identical modules only while they agree.
pub(crate) fn tg_scratch_floats_of(ctx: &Context, func: Ptr<Operation>) -> u32 {
    const DEFAULT: u32 = 128 * 33; // TG_SCRATCH_FLOATS_DEFAULT
    use wax_ir::attr::Attribute as WA;
    use wax_ir::dialect::attr_mirror::WaxAttr;
    let k = "tg_scratch_rows".try_into().unwrap();
    let rows = func
        .deref(ctx)
        .attributes
        .get::<WaxAttr>(&k)
        .and_then(|a| match &a.0 {
            WA::DenseI32Array(arr) => arr.first().copied().map(|v| v as i64),
            _ => None,
        });
    rows.map(|r| (r * SCRATCH_PITCH) as u32).unwrap_or(DEFAULT)
}

/// The natural memory alignment (bytes) of a scalar LLVM type - `walk.rs`'s `align_of`
/// over the scalar set; only scalars occur on the column-reduce path.
pub(crate) fn scalar_align(ty: &LlType) -> u32 {
    match ty {
        LlType::Half | LlType::BFloat | LlType::I16 => 2,
        LlType::I64 | LlType::Double => 8,
        LlType::I8 | LlType::I1 => 1,
        _ => 4, // f32 / i32
    }
}

impl WaxToAir {
    /// `wax.reduce` -> one SIMD reduce intrinsic.
    ///
    /// The within-SIMD case only, which is the whole of `reduce()` once its three special
    /// paths are excluded:
    ///
    /// - a COLUMN reduce (`dim 0` of a `[rows, 32]` tile) spreads across SIMD groups and needs
    ///   threadgroup scratch, barriers and `batch_column_reduces`;
    /// - a reduce of an MMA FRAGMENT takes the per-row shuffle walk, not an intrinsic;
    /// - more than 32 elements without being a per-row reduce goes hierarchical.
    ///
    /// The fragment test is why `fragment_values` had to be shared: the classification decides
    /// which lowering is correct, and a conversion that guessed would route a fragment to the
    /// scalar intrinsic and read one lane's two elements as if they were the row.
    ///
    /// The combine is read from the reduce's REGION - `max`/`min`/`mul`/`add` pick the
    /// intrinsic - and the region is discarded, which is what takes its `wax.yield` with it.
    pub(crate) fn rewrite_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use wax_ir::types::Type as WT;
        use pliron::r#type::Typed;

        let Some(src) = op.deref(ctx).operands().next() else {
            self.bail(Opcode::Reduce, "no source operand");
            return Ok(());
        };
        // The source's WAX type, from the history the driver keeps: its operand may already
        // have been converted, and the shape only exists on the wax side.
        let src_wax = operands_info
            .lookup_most_recent_type(src)
            .or_else(|| Some(src.get_type(ctx)))
            .and_then(|h| wax_ir::dialect::types::from_pliron(ctx, h));
        let Some(src_wax) = src_wax else {
            self.bail(Opcode::Reduce, "source type unreadable");
            return Ok(());
        };
        let (shape, n_elems): (Vec<i64>, i64) = match &src_wax {
            WT::Tile(tt) => (tt.shape.clone(), tt.shape.iter().product()),
            _ => (Vec::new(), 1),
        };
        let dim = self.wax_attr_i64(ctx, op, "dim");
        let spans_groups = shape.len() == 2 && shape[1] == 32 && shape[0] > 1;

        // The combine op names the intrinsic; the result type names the element.
        // Both the scalar and the column path need them, so compute before branching.
        let base = self.combine_base(ctx, op);
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() >= 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            self.bail(Opcode::Reduce, "no result");
            return Ok(());
        };
        let Some((cty, suf, rty)) = self.intrin_ty(ctx, wax_res) else {
            self.bail(Opcode::Reduce, "result is not a float type");
            return Ok(());
        };

        if spans_groups && dim == Some(0) {
            // `walk.rs` tests this BEFORE the SGMMA classifier: a dim-0 reduce of a
            // `[rows, 32]` tile is a column reduce, full stop. `rows == 32` takes the
            // transposed SIMD-intrinsic round trip; anything else walks the column with
            // serial threadgroup loads.
            let rows = shape[0] as i32;
            // A run of adjacent `[32, 32]` reduces is ONE shared round trip in `walk.rs`
            // (`batch_column_reduces`); the batch was planned before conversion. The
            // source cast lands at THIS member's position, as `walk.rs`'s per-op
            // `fpext_if` does, before the body is emitted at the head's.
            if self.reduce.reduce_batches.contains_key(&op) {
                let src = self.fpcast_if(ctx, rewriter, src, &rty, &cty);
                return self.emit_reduce_batch(ctx, rewriter, op, src);
            }
            let rc = ReduceCtx { src, intrinsic: Intrinsic { base, suf }, cty: &cty, rty: &rty };
            return self.column_reduce(ctx, rewriter, op, rc, rows);
        }
        // Grid-aware: by the time this arm runs the operand is usually the representative
        // the mma registered, a new value the pre-conversion fragment set does not know.
        // `walk.rs` makes the same test via `sgmma_of`, which is provenance, not shape.
        if self.frag_dims_of_grid(ctx, operands_info, src).is_some() {
            return self.fragment_row_reduce(
                ctx, rewriter, op, operands_info,
                ReduceCtx { src, intrinsic: Intrinsic { base, suf }, cty: &cty, rty: &rty },
            );
        }
        let row_reduce = spans_groups && dim == Some(1);
        // The result's element count: the diamond is a whole-tile reduce, a scalar. A
        // multi-element result reaching it means the reduce is per-something and the
        // broadcast total would be a wrong answer, so the dispatch below decides by it.
        let res_elems: i64 = match &wax_ir::dialect::types::from_pliron(ctx, wax_res) {
            Some(WT::Tile(tt)) => tt.shape.iter().product(),
            _ => 1,
        };
        if n_elems > 32 && !row_reduce {
            let rc = ReduceCtx { src, intrinsic: Intrinsic { base, suf }, cty: &cty, rty: &rty };
            // A dim-1 reduce of a `[rows, cols]` tile with `cols < 32` is per-ROW: the
            // diamond (a whole-tile sum) answers it with a broadcast total, which is wrong
            // whenever the rows differ. That case takes the row round trip instead; the
            // diamond keeps only the whole-tile (scalar result) reduces it was for.
            let cols = shape.get(1).copied().unwrap_or(0);
            let rows = shape.get(0).copied().unwrap_or(0);
            if dim == Some(1)
                && shape.len() == 2
                && rows > 1
                && cols > 1
                && cols < 32
                && (rows as u64).is_power_of_two()
                && (cols as u64).is_power_of_two()
                && rows <= 32
            {
                return self.row_reduce_scratch(ctx, rewriter, op, rc, rows, cols);
            }
            if res_elems > 1 {
                self.bail(
                    Opcode::Reduce,
                    &format!("reduce: no lowering for a {res_elems}-element result here"),
                );
                return Ok(());
            }
            return self.reduce_hierarchical(ctx, rewriter, op, rc);
        }

        let cty_h = crate::air_dialect::types::to_pliron(ctx, &cty);
        // bf16 reduces in f32, like every other intrinsic on this backend.
        let arg = self.fpcast_if(ctx, rewriter, src, &rty, &cty);
        let call = air::CallOp::new(ctx, &format!("{base}.{suf}"), Some(cty_h), vec![arg]);
        air::set_fast(call.get_operation(), ctx);
        rewriter.insert_operation(ctx, call.get_operation());
        let r = call.get_operation().deref(ctx).get_result(0);
        let r = self.fpcast_if(ctx, rewriter, r, &cty, &rty);
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// The combine body names the reduce intrinsic: the FIRST matching binop wins, exactly
    /// as `walk.rs`'s `reduce_intrinsic` returns on it. Taking the last instead picks a
    /// different intrinsic of the same name length, so the metallib differs in content at
    /// identical size.
    pub(crate) fn combine_base(&self, ctx: &Context, op: Ptr<Operation>) -> &'static str {
        let mut found = "@air.simd_sum";
        use pliron::linked_list::ContainsLinkedList;
        let regions: Vec<_> = op.deref(ctx).regions().collect();
        if let Some(&region) = regions.first() {
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            for b in blocks {
                let ops: Vec<_> = b.deref(ctx).iter(ctx).collect();
                for bop in ops {
                    let hit = match try_opcode_of(ctx, bop) {
                        Some(Opcode::MaxF) => Some("@air.simd_max"),
                        Some(Opcode::MinF) => Some("@air.simd_min"),
                        Some(Opcode::MulF) => Some("@air.simd_product"),
                        Some(Opcode::AddF) => Some("@air.simd_sum"),
                        _ => None,
                    };
                    if let Some(h) = hit {
                        found = h;
                        break;
                    }
                }
            }
        }
        found
    }
    /// Record the scratch global a column reduce needs, once.
    pub(crate) fn needs_tg_scratch_mark(&mut self) {
        if !self.reduce.tg_globals.iter().any(|(n, _, _)| n == "@_tg_user_scratch") {
            self.reduce.tg_globals.push((
                "@_tg_user_scratch".to_string(),
                LlType::Float,
                self.reduce.tg_scratch_floats,
            ));
        }
    }
    /// A dim-0 reduce of a `[rows, 32]` float tile: `walk.rs`'s `reduce_columns` scratch
    /// round trip. Each SIMD group owns a row, so the store is `[group][lane]`. With
    /// `rows == 32` the transposed read `[lane][group]` turns the cross-group reduce into
    /// the within-group SIMD intrinsic, and the result then lives per GROUP while a
    /// `[32]` tile is per LANE, so a second transposed round trip redistributes it; every
    /// lane of a group writes the same value to its own slot, which needs no guard. Any
    /// other row count walks the column with `rows` serial threadgroup loads. The scratch
    /// is FLOAT memory and the round trip COMPUTES IN FLOAT: f16 members are extended to
    /// f32 before the store and truncated back after the load, exactly as the bf16 twin
    /// always did (its f32 compute is why the bf16 kernel builds a pipeline). A store that
    /// writes the raw 2-byte element into the 4-byte slot and a load that reads 4 bytes
    /// back is not a "half in a float slot" - it is half bits plus stale upper bytes, and
    /// the `simd_sum.f16` call on the f32-typed read is a type mismatch Metal 17.6 rejects
    /// at `materializeAll` (the `tg_colsum_f16` finding).
    /// The column reduce's shared scratch round trip scaffolding, in `walk.rs`'s emission
    /// order: wait for the last reader (the kernel's first writer waits for nobody -
    /// `walk.rs` tracks exactly this), split the thread position, take the scratch
    /// reference. Both column-reduce emitters call it at exactly that spot, so the emitted
    /// order agrees on both.
    pub(crate) fn scratch_round_trip(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
    ) -> Result<Option<ScratchRoundTrip>> {
        use crate::air_dialect::attrs::BinOpAttr;
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Reduce, "no thread position parameter");
            return Ok(None);
        };
        self.needs_tg_scratch_mark();
        let f32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::Float);
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let ptr_h = crate::air_dialect::types::to_pliron(
            ctx,
            &LlType::Ptr {
                pointee: Box::new(LlType::Float),
                addrspace: 3,
            },
        );
        let scratch_n = self.reduce.tg_scratch_floats;
        if self.reduce.tg_scratch_dirty {
            air_wg_barrier(ctx, rewriter, i32_ty);
        }
        self.reduce.tg_scratch_dirty = true;
        let zero = air_const_i32(ctx, i32_ty, 0);
        let tid = {
            let e = air::ExtractElemOp::new(ctx, i32_ty, tp, zero);
            rewriter.insert_operation(ctx, e.get_operation());
            e.get_operation().deref(ctx).get_result(0)
        };
        let c32 = air_const_i32(ctx, i32_ty, 32);
        let c31 = air_const_i32(ctx, i32_ty, 31);
        let c_pitch = air_const_i32(ctx, i32_ty, SCRATCH_PITCH);
        let grp = air_bin_i32(ctx, rewriter, i32_ty, tid, c32, BinOpAttr::UDiv);
        let lane = air_bin_i32(ctx, rewriter, i32_ty, tid, c31, BinOpAttr::And);
        // The reference to the scratch is a module value, not an instruction.
        let base_ptr = air::GlobalRefOp::new(ctx, ptr_h, "@_tg_user_scratch");
        rewriter.insert_operation(ctx, base_ptr.get_operation());
        let base_ptr = base_ptr.get_operation().deref(ctx).get_result(0);
        Ok(Some(ScratchRoundTrip {
            f32_h,
            i32_ty,
            ptr_h,
            scratch_n,
            zero,
            grp,
            lane,
            c_pitch,
            base_ptr,
        }))
    }
    /// One shared scratch round trip for a whole run of `[32, 32]` column reduces -
    /// `walk.rs::batch_column_reduces`, the conversion's spelling of it.
    ///
    /// The FIRST member the driver reaches emits the whole batch at the HEAD's position
    /// (where `walk.rs` puts it); every later member only answers from the recorded
    /// results and erases itself. `this_src` is the calling member's already-cast source;
    /// the other members' current source values go in as-is and are extended to the
    /// scratch's FLOAT element type at the store when they are not - the round trip
    /// computes in f32, see `column_reduce`'s note on the f16 finding.
    pub(crate) fn emit_reduce_batch(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        this_src: pliron::value::Value,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        use pliron::irbuild::inserter::OpInsertionPoint;
        use pliron::r#type::Typed;
        type V = pliron::value::Value;
        // Already emitted by the member the driver reached first: answer and erase.
        if let Some(&out) = self.reduce.reduce_batch_results.get(&op) {
            rewriter.replace_operation_with_values(ctx, op, vec![out]);
            return Ok(());
        }
        let (_slot, record) = *self.reduce.reduce_batches
            .get(&op)
            .expect("planned member has a record");
        let rec = &self.reduce.reduce_batch_records[record];
        let head = rec.head;
        let members = rec.members.clone();
        let info = rec.info.clone();
        // The driver's insertion point is before THIS member; the batch lives at the
        // HEAD's.
        if head != op {
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(head));
        }
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let f32_h = rt.f32_h;
        let grp = rt.grp;
        let lane = rt.lane;
        let c_pitch = rt.c_pitch;

        let n = members.len();
        // The scratch cap decides how many members phase 1 can serve per pass; the
        // redistribution below is compact (one float per member per group), so a single
        // round trip still serves them all. `walk.rs` refuses to batch past it too.
        let per_member = 32i64 * SCRATCH_PITCH;
        let chunk = (self.reduce.tg_scratch_floats as i64 / per_member).max(1) as usize;
        let mut per_group: Vec<V> = Vec::with_capacity(n);
        for (ci, sub) in (0..n).collect::<Vec<_>>().chunks(chunk).enumerate() {
            if ci > 0 {
                rt.barrier(ctx, rewriter);
            }
            // Phase 1: every member writes [row = its SIMD group][col = its lane] into
            // its own region.
            for &gi in sub {
                let &(_, _, _, _, _) = &info[gi];
                let m = members[gi];
                let mut src_v = if m == op {
                    this_src
                } else {
                    m.deref(ctx).operands().next().unwrap()
                };
                // The round trip computes in FLOAT, the scratch's element type: extend a
                // 2-byte element (f16, or a bf16 whose raw value would otherwise land in
                // the 4-byte slot) the way `column_reduce` does at its dispatch site.
                let vt = crate::air_dialect::types::from_pliron(ctx, src_v.get_type(ctx));
                if !matches!(vt, Some(LlType::Float)) {
                    src_v = self.fpcast_if(ctx, rewriter, src_v, vt.as_ref().unwrap(), &LlType::Float);
                }
                let region = rt.const_i(ctx, gi as i64 * per_member);
                let base_off = rt.bin(ctx, rewriter, grp, c_pitch, BinOpAttr::Mul);
                let flat = rt.bin(ctx, rewriter, base_off, lane, BinOpAttr::Add);
                let flat = rt.bin(ctx, rewriter, flat, region, BinOpAttr::Add);
                let gep = rt.scratch_gep(ctx, rewriter, flat);
                let st = air::StoreOp::new(ctx, src_v, gep);
                air::set_align(st.get_operation(), ctx, 4);
                rewriter.insert_operation(ctx, st.get_operation());
            }
            rt.barrier(ctx, rewriter);
            // Phase 2: the transposed read makes the cross-group reduce a within-group
            // one - the SIMD reduce intrinsic, log depth, in registers.
            for &gi in sub {
                let &(base, _, _, _, _) = &info[gi];
                let region = rt.const_i(ctx, gi as i64 * per_member);
                let row_off = rt.bin(ctx, rewriter, lane, c_pitch, BinOpAttr::Mul);
                let flat_t = rt.bin(ctx, rewriter, row_off, grp, BinOpAttr::Add);
                let flat_t = rt.bin(ctx, rewriter, flat_t, region, BinOpAttr::Add);
                let gep_t = rt.scratch_gep(ctx, rewriter, flat_t);
                let ld = air::LoadOp::new(ctx, f32_h, gep_t);
                air::set_align(ld.get_operation(), ctx, 4);
                rewriter.insert_operation(ctx, ld.get_operation());
                let v = ld.get_operation().deref(ctx).get_result(0);
                let call = air::CallOp::new(ctx, &format!("{base}.f32"), Some(f32_h), vec![v]);
                air::set_fast(call.get_operation(), ctx);
                rewriter.insert_operation(ctx, call.get_operation());
                per_group.push(call.get_operation().deref(ctx).get_result(0));
            }
        }
        // One compact redistribution for every member: the value each group holds is
        // SG-uniform, so one float per (member, group) serves them all - the difference
        // between three barriers and three per member.
        rt.barrier(ctx, rewriter);
        for (i, _) in members.iter().enumerate() {
            let (_, _, _, _, _) = &info[i];
            let off = rt.const_i(ctx, i as i64 * 32);
            let slot = rt.bin(ctx, rewriter, grp, off, BinOpAttr::Add);
            let gep_w = rt.scratch_gep(ctx, rewriter, slot);
            let st2 = air::StoreOp::new(ctx, per_group[i], gep_w);
            air::set_align(st2.get_operation(), ctx, 4);
            rewriter.insert_operation(ctx, st2.get_operation());
        }
        rt.barrier(ctx, rewriter);
        for (i, &m) in members.iter().enumerate() {
            let (_, _, _, rty, _) = &info[i];
            let off = rt.const_i(ctx, i as i64 * 32);
            let slot = rt.bin(ctx, rewriter, lane, off, BinOpAttr::Add);
            let gep_r = rt.scratch_gep(ctx, rewriter, slot);
            let ld2 = air::LoadOp::new(ctx, f32_h, gep_r);
            air::set_align(ld2.get_operation(), ctx, 4);
            rewriter.insert_operation(ctx, ld2.get_operation());
            let out = ld2.get_operation().deref(ctx).get_result(0);
            let out = self.fpcast_if(ctx, rewriter, out, &LlType::Float, rty);
            self.reduce.reduce_batch_results.insert(m, out);
        }
        let out = self.reduce.reduce_batch_results
            .get(&op)
            .copied()
            .expect("just recorded");
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    pub(crate) fn column_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
        rows: i32,
    ) -> Result<()> {
        let ReduceCtx { src, intrinsic, cty, rty } = rc;
        let Intrinsic { base, suf } = intrinsic;
        use crate::air_dialect::attrs::BinOpAttr;
        // The source cast happens FIRST - `walk.rs` applies its `fpext_if` at the
        // dispatch site, before the war barrier and the arithmetic (identity for
        // f32; bf16 and f16 extend to f32).
        let src = self.fpcast_if(ctx, rewriter, src, rty, cty);
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        // The round trip computes in FLOAT, the scratch's element type: f16 is extended
        // here (its f32 compute is what makes the metallib build - see the note on the
        // scratch above) and truncated back at the end. `suf` is the COMPUTE type's
        // suffix; the scratch path forces f32 for f16, so `walk.rs`'s `.f16` call on a
        // 2-byte value never comes out.
        let src = self.fpcast_if(ctx, rewriter, src, &cty, &LlType::Float);
        let al = 4;
        let st_suf = if *cty == LlType::Half { "f32" } else { suf };
        let f32_h = rt.f32_h;
        let zero = rt.zero;
        let lane = rt.lane;
        let grp = rt.grp;
        let c_pitch = rt.c_pitch;

        // Store this thread's element at `[row = its SIMD group][col = its lane]`.
        let base_off = rt.bin(ctx, rewriter, grp, c_pitch, BinOpAttr::Mul);
        let flat = rt.bin(ctx, rewriter, base_off, lane, BinOpAttr::Add);
        let gep = rt.scratch_gep(ctx, rewriter, flat);
        let st = air::StoreOp::new(ctx, src, gep);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, st.get_operation());
        rt.barrier(ctx, rewriter);

        if rows == 32 {
            // Transposed read: `[lane][group]`, so the SIMD reduce intrinsic is the
            // cross-group reduce - log depth, in registers. Only the `rows == 32` path
            // takes this; any other row count walks the column from the store above.
            // The load is FLOAT - the scratch's gep pointee - and the intrinsic runs on
            // it, in f32, in every element type.
            let row_off = rt.bin(ctx, rewriter, lane, c_pitch, BinOpAttr::Mul);
            let flat_t = rt.bin(ctx, rewriter, row_off, grp, BinOpAttr::Add);
            let gep_t = rt.scratch_gep(ctx, rewriter, flat_t);
            let ld = air::LoadOp::new(ctx, f32_h, gep_t);
            air::set_align(ld.get_operation(), ctx, al);
            rewriter.insert_operation(ctx, ld.get_operation());
            let v = ld.get_operation().deref(ctx).get_result(0);
            let call = air::CallOp::new(
                ctx,
                &format!("{base}.{st_suf}"),
                Some(f32_h),
                vec![v],
            );
            air::set_fast(call.get_operation(), ctx);
            rewriter.insert_operation(ctx, call.get_operation());
            let per_group = call.get_operation().deref(ctx).get_result(0);

            // Source constant along the lane dim: every column reduces to the same
            // answer, so the value this group just produced IS the answer for every
            // lane and the round trip below would redistribute a value to itself. The
            // question is asked of the PRE-conversion source: the operand was rewired
            // by then, and the analysis ran before the rewiring.
            let uniform = self.reduce.reduce_sources
                .get(&op)
                .is_some_and(|s| !self.reduce.divergent.contains(s));
            if uniform {
                let out = self.fpcast_if(ctx, rewriter, per_group, &LlType::Float, rty);
                rewriter.replace_operation_with_values(ctx, op, vec![out]);
                self.converted += 1;
                return Ok(());
            }

            rt.barrier(ctx, rewriter);
            let gep_w = rt.scratch_gep(ctx, rewriter, flat);
            let st2 = air::StoreOp::new(ctx, per_group, gep_w);
            air::set_align(st2.get_operation(), ctx, al);
            rewriter.insert_operation(ctx, st2.get_operation());
            rt.barrier(ctx, rewriter);
            let flat_r = rt.bin(ctx, rewriter, lane, c_pitch, BinOpAttr::Mul);
            let gep_r = rt.scratch_gep(ctx, rewriter, flat_r);
            let ld2 = air::LoadOp::new(ctx, f32_h, gep_r);
            air::set_align(ld2.get_operation(), ctx, al);
            rewriter.insert_operation(ctx, ld2.get_operation());
            let out = ld2.get_operation().deref(ctx).get_result(0);
            let out = self.fpcast_if(ctx, rewriter, out, &LlType::Float, rty);
            rewriter.replace_operation_with_values(ctx, op, vec![out]);
            self.converted += 1;
            return Ok(());
        }

        // Fallback for a row count that does not match the SIMD width: walk the
        // column with serial threadgroup loads. The loads are FLOAT (the scratch's
        // gep pointee) and the combine runs in float too, in every element type;
        // `walk.rs`'s `scan_binop` only carried the compute type in the fmax call's
        // name, which `emit_combine` already does via `suf`.
        let flat0 = rt.bin(ctx, rewriter, zero, lane, BinOpAttr::Add);
        let g0 = rt.scratch_gep(ctx, rewriter, flat0);
        let l0 = air::LoadOp::new(ctx, f32_h, g0);
        air::set_align(l0.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, l0.get_operation());
        let mut acc = l0.get_operation().deref(ctx).get_result(0);
        for r in 1..rows {
            let off = rt.const_i(ctx, r as i64 * SCRATCH_PITCH);
            let flat_r = rt.bin(ctx, rewriter, off, lane, BinOpAttr::Add);
            let g = rt.scratch_gep(ctx, rewriter, flat_r);
            let l = air::LoadOp::new(ctx, f32_h, g);
            air::set_align(l.get_operation(), ctx, al);
            rewriter.insert_operation(ctx, l.get_operation());
            let v = l.get_operation().deref(ctx).get_result(0);
            acc = self.emit_combine(ctx, rewriter, acc, v, intrinsic, f32_h);
        }
        // Everyone has read the column before any later write reuses the scratch.
        rt.barrier(ctx, rewriter);
        let out = self.fpcast_if(ctx, rewriter, acc, &LlType::Float, rty);
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// A dim-1 (per-row) reduce of a `[rows, cols]` tile with `1 < cols < 32` and `rows`
    /// a power of two, e.g. `[16, 16]`. The tile load gives one element per lane, so row
    /// `r` of the tile sits in `rows_pg = 32 / cols` consecutive lanes' worth: within its
    /// group, lanes `[cols * (r mod rows_pg), cols * (r mod rows_pg) + cols)` - the
    /// row's `cols` elements are the lower/upper `cols` lanes of the half their lane
    /// number picks. A `shuffle_xor` butterfly over masks `1 .. cols/2` therefore folds
    /// each such run into its lane without ever crossing a row boundary, and every lane
    /// ends holding its own row's reduce in registers - no store phase needed.
    ///
    /// The result is a `[rows]` tile, whose store maps lane `l` to element `l mod rows`.
    /// That is a different lane-to-row mapping than the butterfly's (group `g`'s lanes
    /// cover rows `g * rows_pg .. g * rows_pg + rows_pg`), so the rows are parked in
    /// `rows` slots past the column region - slot `g * rows_pg + l / cols` - with one
    /// barrier between the park and the final per-lane read of slot `l mod rows`. The
    /// reads are redundant by design: every lane that stores a given element reads the
    /// same slot, so the store is race-free.
    ///
    /// The round trip computes in FLOAT, the scratch's element type (see `column_reduce`).
    pub(crate) fn row_reduce_scratch(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
        rows: i64,
        cols: i64,
    ) -> Result<()> {
        let ReduceCtx { src, intrinsic, cty, rty } = rc;
        use crate::air_dialect::attrs::BinOpAttr;
        let src = self.fpcast_if(ctx, rewriter, src, &cty, &LlType::Float);
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let f32_h = rt.f32_h;
        let i16_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I16);
        let lane = rt.lane;
        let grp = rt.grp;

        // Phase 1: the within-row butterfly, in registers.
        let mut v = src;
        let mut mask: i64 = 1;
        while mask < cols {
            let m = {
                let c = air::ConstBitsOp::new(ctx, i16_h, mask as u64);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
            let partner = {
                let c = air::CallOp::new(
                    ctx,
                    crate::lower::AIR_SIMD_SHUFFLE_XOR_F32,
                    Some(f32_h),
                    vec![v, m],
                );
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
            v = self.emit_combine(ctx, rewriter, v, partner, intrinsic, f32_h);
            mask *= 2;
        }

        // Phase 2: park each row in its slot, past the column region (32 rows of pitch).
        let rows_pg = 32 / cols;
        let region = rt.const_i(ctx, 32 * SCRATCH_PITCH);
        let row_of_lane = {
            let rp = rt.const_i(ctx, rows_pg);
            let base = rt.bin(ctx, rewriter, grp, rp, BinOpAttr::Mul);
            let cs = rt.const_i(ctx, cols);
            let half = rt.bin(ctx, rewriter, lane, cs, BinOpAttr::UDiv);
            rt.bin(ctx, rewriter, base, half, BinOpAttr::Add)
        };
        let flat = rt.bin(ctx, rewriter, region, row_of_lane, BinOpAttr::Add);
        let gep_w = rt.scratch_gep(ctx, rewriter, flat);
        let st = air::StoreOp::new(ctx, v, gep_w);
        air::set_align(st.get_operation(), ctx, 4);
        rewriter.insert_operation(ctx, st.get_operation());
        rt.barrier(ctx, rewriter);

        // Phase 3: the `[rows]` store maps lane `l` to element `l mod rows`; hand every
        // lane its element's row. `rows` is a power of two, so `mod` is `and`.
        let slot = {
            let m = rt.const_i(ctx, rows - 1);
            let l = rt.bin(ctx, rewriter, lane, m, BinOpAttr::And);
            rt.bin(ctx, rewriter, region, l, BinOpAttr::Add)
        };
        let gep_r = rt.scratch_gep(ctx, rewriter, slot);
        let ld = air::LoadOp::new(ctx, f32_h, gep_r);
        air::set_align(ld.get_operation(), ctx, 4);
        rewriter.insert_operation(ctx, ld.get_operation());
        let out = ld.get_operation().deref(ctx).get_result(0);
        let out = self.fpcast_if(ctx, rewriter, out, &LlType::Float, rty);
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// A whole-tile reduce over more than 32 elements: `walk.rs`'s `reduce_hierarchical`.
    /// A block diamond through two threadgroup globals, with no phis - the values cross
    /// the blocks through threadgroup memory, exactly as the CFG spells it:
    ///
    /// ```text
    /// head:   s1 = simd-reduce(src); tid/grp/lane; cond_br grp==0 init/skip_init
    /// init:   partial[lane] = identity            skip_init:            (fall in)
    /// after_init: barrier; cond_br lane==0 write/skip_write
    /// write:  partial[grp] = s1                   skip_write:           (fall in)
    /// after_write: barrier; cond_br grp==0 final/skip_final
    /// final:  total = simd-reduce(partial[lane]); cond_br lane==0 wr/sk
    /// write_res: result = total                   skip_res:             (fall in)
    /// after_res / skip_final: (fall in)
    /// after_final: barrier; result = load
    /// ```
    ///
    /// `lane` is `urem` here and `and` in the column path: the two lowers are separate
    /// code in `walk.rs`, and matching the bytes means matching that.
    pub(crate) fn reduce_hierarchical(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
    ) -> Result<()> {
        let ReduceCtx { src, intrinsic, cty, rty } = rc;
        let Intrinsic { base, suf } = intrinsic;
        use crate::air_dialect::attrs::BinOpAttr;
        use crate::air_dialect::attrs::ICmpPredAttr;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::LinkedList;
        type V = pliron::value::Value;
        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::Reduce, "reduce is not inside a block");
            return Ok(());
        };
        let Some(region) = head.deref(ctx).get_container() else {
            self.bail(Opcode::Reduce, "block is not inside a region");
            return Ok(());
        };
        let al = match cty {
            LlType::Float | LlType::I32 => 4,
            LlType::Half | LlType::BFloat => 2,
            _ => {
                self.bail(Opcode::Reduce, "hierarchical reduce: unsupported element type");
                return Ok(());
            }
        };
        // One type per kernel: every reduce in a kernel shares the two 32-slot
        // threadgroup globals, so the first reduce declares them (in the compute type) and
        // the rest reuse them - a second declaration would be extra globals in the
        // signature. `walk.rs` makes the same assumption.
        if !self.reduce.tg_reduce_allocated {
            self.reduce.tg_globals.push(("@_tg_reduce_partial".to_string(), cty.clone(), 32));
            self.reduce.tg_globals.push(("@_tg_reduce_result".to_string(), cty.clone(), 1));
            self.reduce.tg_reduce_allocated = true;
        }
        let cty_h = crate::air_dialect::types::to_pliron(ctx, cty);
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let ptr_h = crate::air_dialect::types::to_pliron(
            ctx,
            &LlType::Ptr {
                pointee: Box::new(cty.clone()),
                addrspace: 3,
            },
        );
        let intrinsic = format!("{base}.{suf}");

        // The split puts the reduce (and everything after it) in `after_final`; the other
        // eleven blocks are created in CFG order. `walk.rs` appends each block it creates
        // to the end of the function's block list, so we must too: anchoring them just
        // before `after_final` lands the diamond between a loop body and its back edge when
        // the reduce sits inside a loop, reordering the blocks and the metallib.
        let after_final = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        let mut mk = |rw: &mut DialectConversionRewriter| {
            rw.create_block(ctx, BlockInsertionPoint::AtRegionEnd(region), None, vec![])
        };
        let init = mk(rewriter);
        let skip_init = mk(rewriter);
        let after_init = mk(rewriter);
        let write = mk(rewriter);
        let skip_write = mk(rewriter);
        let after_write = mk(rewriter);
        let final_b = mk(rewriter);
        let write_res = mk(rewriter);
        let skip_res = mk(rewriter);
        let after_res = mk(rewriter);
        let skip_final = mk(rewriter);
        // `split_block` left `after_final` where the reduce sat. The walk's counterpart -
        // the block the body resumes in after the diamond - is appended AFTER the
        // diamond, so move it there; leaving it in place reorders the blocks.
        rewriter.move_block(ctx, after_final, BlockInsertionPoint::AtRegionEnd(region));

        let const_i = |ctx: &mut Context, v: i64| -> V {
            let c = air::ConstBitsOp::new(ctx, i32_h, v as u64);
            c.get_operation().deref(ctx).get_result(0)
        };
        let barrier = |ctx: &mut Context, rw: &mut DialectConversionRewriter| {
            let two = const_i(ctx, 2);
            let one = const_i(ctx, 1);
            let b = air::CallOp::new(ctx, crate::lower::AIR_WG_BARRIER, None, vec![two, one]);
            rw.insert_operation(ctx, b.get_operation());
        };
        let icmp_eq = |ctx: &mut Context, rw: &mut DialectConversionRewriter, a: V, b: V| -> V {
            let c = air::ICmpOp::new(ctx, i1_h, a, b);
            c.set_pred(ctx, ICmpPredAttr::Eq);
            rw.insert_operation(ctx, c.get_operation());
            c.get_operation().deref(ctx).get_result(0)
        };
        let partial_ptr = air::GlobalRefOp::new(ctx, ptr_h, "@_tg_reduce_partial");
        rewriter.insert_operation(ctx, partial_ptr.get_operation());
        let partial_ptr = partial_ptr.get_operation().deref(ctx).get_result(0);
        let result_ptr = air::GlobalRefOp::new(ctx, ptr_h, "@_tg_reduce_result");
        rewriter.insert_operation(ctx, result_ptr.get_operation());
        let result_ptr = result_ptr.get_operation().deref(ctx).get_result(0);
        let partial_gep =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, idx: V| -> V {
                let g = air::GepArrayOp::new(ctx, ptr_h, partial_ptr, idx);
                air::set_elem_ty(g.get_operation(), ctx, cty_h);
                air::set_count(g.get_operation(), ctx, 32);
                rw.insert_operation(ctx, g.get_operation());
                g.get_operation().deref(ctx).get_result(0)
            };
        let simd_reduce = |ctx: &mut Context, rw: &mut DialectConversionRewriter, v: V| -> V {
            let c = air::CallOp::new(ctx, &intrinsic, Some(cty_h), vec![v]);
            air::set_fast(c.get_operation(), ctx);
            rw.insert_operation(ctx, c.get_operation());
            c.get_operation().deref(ctx).get_result(0)
        };

        // head: within-SIMD reduce + group/lane indices, then the phase-0 branch.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(head));
        let s1 = simd_reduce(ctx, rewriter, src);
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Reduce, "no thread position parameter");
            return Ok(());
        };
        let zero = const_i(ctx, 0);
        let tid = {
            let e = air::ExtractElemOp::new(ctx, i32_h, tp, zero);
            rewriter.insert_operation(ctx, e.get_operation());
            e.get_operation().deref(ctx).get_result(0)
        };
        let c32 = const_i(ctx, 32);
        let grp = {
            let o = air::BinOp::new(ctx, i32_h, tid, c32);
            o.set_op(ctx, BinOpAttr::UDiv);
            rewriter.insert_operation(ctx, o.get_operation());
            o.get_operation().deref(ctx).get_result(0)
        };
        let lane = {
            let o = air::BinOp::new(ctx, i32_h, tid, c32);
            o.set_op(ctx, BinOpAttr::URem);
            rewriter.insert_operation(ctx, o.get_operation());
            o.get_operation().deref(ctx).get_result(0)
        };
        let is_grp0_a = icmp_eq(ctx, rewriter, grp, zero);
        let cbr0 = air::CondBrOp::new(ctx, is_grp0_a, init, vec![], skip_init, vec![]);
        cbr0.get_operation().insert_at_back(head, ctx);

        // init: group 0 writes the identity into every partial slot.
        let identity = {
            let c = air::ConstBitsOp::new(ctx, cty_h, reduce_identity_bits(base, suf));
            c.get_operation().deref(ctx).get_result(0)
        };
        // The gep belongs in its own diamond block: bitcode has no block boundaries, so an
        // instruction emitted at the stale head position lands at the wrong stream offset
        // whenever blocks sit between the head and the diamond (a reduce inside a loop).
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(init));
        let g = partial_gep(ctx, rewriter, lane);
        let st = air::StoreOp::new(ctx, identity, g);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_init, vec![]);
        b.get_operation().insert_at_back(init, ctx);
        let b = air::BrOp::new(ctx, after_init, vec![]);
        b.get_operation().insert_at_back(skip_init, ctx);

        // after_init: lane 0 of each group publishes its partial.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_init));
        barrier(ctx, rewriter);
        let is_lane0_a = icmp_eq(ctx, rewriter, lane, zero);
        let cbr1 = air::CondBrOp::new(ctx, is_lane0_a, write, vec![], skip_write, vec![]);
        cbr1.get_operation().insert_at_back(after_init, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(write));
        let g = partial_gep(ctx, rewriter, grp);
        let st = air::StoreOp::new(ctx, s1, g);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_write, vec![]);
        b.get_operation().insert_at_back(write, ctx);
        let b = air::BrOp::new(ctx, after_write, vec![]);
        b.get_operation().insert_at_back(skip_write, ctx);

        // after_write: group 0 reduces the partials; its lane 0 writes the result.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_write));
        barrier(ctx, rewriter);
        let is_grp0_b = icmp_eq(ctx, rewriter, grp, zero);
        let cbr2 = air::CondBrOp::new(ctx, is_grp0_b, final_b, vec![], skip_final, vec![]);
        cbr2.get_operation().insert_at_back(after_write, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(final_b));
        let g = partial_gep(ctx, rewriter, lane);
        let ld = air::LoadOp::new(ctx, cty_h, g);
        air::set_align(ld.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, ld.get_operation());
        let v = ld.get_operation().deref(ctx).get_result(0);
        let total = simd_reduce(ctx, rewriter, v);
        let is_lane0_b = icmp_eq(ctx, rewriter, lane, zero);
        let cbr3 = air::CondBrOp::new(ctx, is_lane0_b, write_res, vec![], skip_res, vec![]);
        cbr3.get_operation().insert_at_back(final_b, ctx);
        let st = air::StoreOp::new(ctx, total, result_ptr);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(write_res));
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_res, vec![]);
        b.get_operation().insert_at_back(write_res, ctx);
        let b = air::BrOp::new(ctx, after_res, vec![]);
        b.get_operation().insert_at_back(skip_res, ctx);
        let b = air::BrOp::new(ctx, after_final, vec![]);
        b.get_operation().insert_at_back(after_res, ctx);
        let b = air::BrOp::new(ctx, after_final, vec![]);
        b.get_operation().insert_at_back(skip_final, ctx);

        // after_final: everyone reads the broadcast result. The reduce sits at the
        // block's head; the barrier and load go in front of it, and it is replaced by
        // the loaded value.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        barrier(ctx, rewriter);
        let ld = air::LoadOp::new(ctx, cty_h, result_ptr);
        air::set_align(ld.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, ld.get_operation());
        let v = ld.get_operation().deref(ctx).get_result(0);
        let result = self.fpcast_if(ctx, rewriter, v, cty, rty);
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        self.converted += 1;
        Ok(())
    }
    /// The combine step of a reduce or scan, spelled the way `walk.rs`'s `scan_binop`
    /// spells it for `base`: fadd / fmul, `call fast @air.fast_fmax.{suf}` for max,
    /// `fcmp olt` + `select` for min. Not the `air.bin` the generic elementwise arm
    /// emits - byte parity means spelling parity.
    pub(crate) fn emit_combine(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        a: pliron::value::Value,
        b: pliron::value::Value,
        intrinsic: Intrinsic,
        ty_h: TypeHandle,
    ) -> pliron::value::Value {
        let Intrinsic { base, suf } = intrinsic;
        use crate::air_dialect::attrs::BinOpAttr;
        use crate::air_dialect::attrs::FCmpPredAttr;
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let o = match base {
            "@air.simd_product" => {
                let m = air::BinOp::new(ctx, ty_h, a, b);
                m.set_op(ctx, BinOpAttr::Mul);
                m.get_operation()
            }
            "@air.simd_max" => {
                let c = air::CallOp::new(
                    ctx,
                    &format!("@air.fast_fmax.{suf}"),
                    Some(ty_h),
                    vec![a, b],
                );
                air::set_fast(c.get_operation(), ctx);
                c.get_operation()
            }
            "@air.simd_min" => {
                let c = air::FCmpOp::new(ctx, i1_h, a, b);
                c.set_pred(ctx, FCmpPredAttr::Olt);
                rewriter.insert_operation(ctx, c.get_operation());
                let cond = c.get_operation().deref(ctx).get_result(0);
                let s = air::SelectOp::new(ctx, ty_h, cond, a, b);
                s.get_operation()
            }
            _ => {
                let d = air::BinOp::new(ctx, ty_h, a, b);
                d.set_op(ctx, BinOpAttr::Add);
                d.get_operation()
            }
        };
        rewriter.insert_operation(ctx, o);
        o.deref(ctx).get_result(0)
    }
    /// `walk.rs`'s `scan`: a prefix scan. At one SIMD group or fewer it is a within-SIMD
    /// shuffle scan; past that, the four-phase threadgroup version - each group scans
    /// within itself, the boundary lane publishes the group total, group 0 prefix-scans
    /// the totals, and every thread combines the exclusive group prefix with its
    /// within-group result.
    pub(crate) fn rewrite_scan(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinOpAttr;
        use crate::air_dialect::attrs::ICmpPredAttr;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::LinkedList;
        use pliron::r#type::Typed;
        type V = pliron::value::Value;
        let Some(src) = op.deref(ctx).operands().next() else {
            self.bail(Opcode::Scan, "no source operand");
            return Ok(());
        };
        if self.frag_dims_of_grid(ctx, operands_info, src).is_some() {
            self.bail(Opcode::Scan, "fragment scan is not implemented");
            return Ok(());
        }
        let src_wax = operands_info
            .lookup_most_recent_type(src)
            .or_else(|| {
                use pliron::r#type::Typed;
                Some(src.get_type(ctx))
            })
            .and_then(|h| wax_type_of(ctx, h));
        let Some(src_wax) = src_wax else {
            self.bail(Opcode::Scan, "source type unreadable");
            return Ok(());
        };
        let n_elems: i64 = match &src_wax {
            WaxType::Tile(t) => t.shape.iter().product(),
            _ => 1,
        };
        let base = self.combine_base(ctx, op);
        let src_ty_h = operands_info
            .lookup_most_recent_type(src)
            .unwrap_or_else(|| src.get_type(ctx));
        let Some((cty, suf, rty)) = self.intrin_ty(ctx, src_ty_h) else {
            self.bail(Opcode::Scan, "source is not a float type");
            return Ok(());
        };
        let intrinsic = Intrinsic { base, suf };
        let reverse = self.wax_attr_i64(ctx, op, "reverse") == Some(1);
        let al = match cty {
            LlType::Float | LlType::I32 => 4,
            LlType::Half | LlType::BFloat => 2,
            _ => {
                self.bail(Opcode::Scan, "scan: unsupported element type");
                return Ok(());
            }
        };
        let cty_h = crate::air_dialect::types::to_pliron(ctx, &cty);
        let i32_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let i16_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I16);
        let shuffle = format!(
            "@air.simd_shuffle_{}.{suf}",
            if reverse { "down" } else { "up" }
        );
        let const_i = |ctx: &mut Context, v: i64| -> V {
            let c = air::ConstBitsOp::new(ctx, i32_h, v as u64);
            c.get_operation().deref(ctx).get_result(0)
        };
        let i1_h = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let icmp =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, p: ICmpPredAttr, a: V, b: V| -> V {
                let c = air::ICmpOp::new(ctx, i1_h, a, b);
                c.set_pred(ctx, p);
                rw.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
        let select = |ctx: &mut Context,
                      rw: &mut DialectConversionRewriter,
                      c: V,
                      a: V,
                      b: V,
                      ty: TypeHandle| -> V {
            let s = air::SelectOp::new(ctx, ty, c, a, b);
            rw.insert_operation(ctx, s.get_operation());
            s.get_operation().deref(ctx).get_result(0)
        };
        let shuffle_call =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, v: V, delta: i64| -> V {
                let d = air::ConstBitsOp::new(ctx, i16_h, delta as u64);
                rw.insert_operation(ctx, d.get_operation());
                let dv = d.get_operation().deref(ctx).get_result(0);
                let c = air::CallOp::new(ctx, &shuffle, Some(cty_h), vec![v, dv]);
                air::set_fast(c.get_operation(), ctx);
                rw.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
        let barrier = |ctx: &mut Context, rw: &mut DialectConversionRewriter| {
            let two = const_i(ctx, 2);
            let one = const_i(ctx, 1);
            let b = air::CallOp::new(ctx, crate::lower::AIR_WG_BARRIER, None, vec![two, one]);
            rw.insert_operation(ctx, b.get_operation());
        };

        // bf16 scans run in f32, like every other intrinsic on this backend.
        let src = self.fpcast_if(ctx, rewriter, src, &rty, &cty);
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Scan, "no thread position parameter");
            return Ok(());
        };
        let zero = const_i(ctx, 0);
        let tid = {
            let e = air::ExtractElemOp::new(ctx, i32_h, tp, zero);
            rewriter.insert_operation(ctx, e.get_operation());
            e.get_operation().deref(ctx).get_result(0)
        };
        let c32 = const_i(ctx, 32);
        let lane = {
            let o = air::BinOp::new(ctx, i32_h, tid, c32);
            o.set_op(ctx, BinOpAttr::URem);
            rewriter.insert_operation(ctx, o.get_operation());
            o.get_operation().deref(ctx).get_result(0)
        };

        // The within-SIMD phase: deltas 1, 2, 4, ... up to the group width.
        let simd_n = n_elems.min(32);
        let mut steps = Vec::new();
        let mut d = 1i64;
        while d < simd_n {
            steps.push(d);
            d <<= 1;
        }
        if steps.is_empty() {
            // D == 1: the scan is the identity on the single element, spelled as an
            // add of the zero identity exactly as `walk.rs` does.
            let zc = air::ConstBitsOp::new(ctx, cty_h, 0);
            rewriter.insert_operation(ctx, zc.get_operation());
            let z = zc.get_operation().deref(ctx).get_result(0);
            let r = self.emit_combine(ctx, rewriter, z, src, intrinsic, cty_h);
            let r = self.fpcast_if(ctx, rewriter, r, &cty, &rty);
            rewriter.replace_operation_with_values(ctx, op, vec![r]);
            self.converted += 1;
            return Ok(());
        }
        let mut cur = src;
        for &delta in &steps {
            let shuf = shuffle_call(ctx, rewriter, cur, delta);
            // Lanes without a neighbor in the shuffle direction keep their value.
            let limit = const_i(ctx, simd_n - delta);
            let cd = const_i(ctx, delta);
            let cond = if reverse {
                icmp(ctx, rewriter, ICmpPredAttr::Uge, lane, limit)
            } else {
                icmp(ctx, rewriter, ICmpPredAttr::Ult, lane, cd)
            };
            let combined = self.emit_combine(ctx, rewriter, shuf, cur, intrinsic, cty_h);
            cur = select(ctx, rewriter, cond, cur, combined, cty_h);
        }
        if n_elems <= 32 {
            let r = self.fpcast_if(ctx, rewriter, cur, &cty, &rty);
            rewriter.replace_operation_with_values(ctx, op, vec![r]);
            self.converted += 1;
            return Ok(());
        }

        // ---- Phases 2-4: across the SIMD groups, through @_tg_scan_partials. ----
        // One partials global per kernel: every scan in a kernel shares the same 32-slot
        // threadgroup area, so the first scan declares it (in the compute type) and the
        // rest reuse it - a second declaration would be a second global in the signature.
        if !self.reduce.tg_scan_allocated {
            self.reduce.tg_globals.push(("@_tg_scan_partials".to_string(), cty.clone(), 32));
            self.reduce.tg_scan_allocated = true;
        }
        let n_groups = n_elems / 32;
        let identity_bits = reduce_identity_bits(base, suf);
        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::Scan, "scan is not inside a block");
            return Ok(());
        };
        let ptr_h = crate::air_dialect::types::to_pliron(
            ctx,
            &LlType::Ptr {
                pointee: Box::new(cty.clone()),
                addrspace: 3,
            },
        );
        let partials_ref = air::GlobalRefOp::new(ctx, ptr_h, "@_tg_scan_partials");
        rewriter.insert_operation(ctx, partials_ref.get_operation());
        let partials_ref = partials_ref.get_operation().deref(ctx).get_result(0);
        let partial_gep =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, idx: V| -> V {
                let g = air::GepArrayOp::new(ctx, ptr_h, partials_ref, idx);
                air::set_elem_ty(g.get_operation(), ctx, cty_h);
                air::set_count(g.get_operation(), ctx, 32);
                rw.insert_operation(ctx, g.get_operation());
                g.get_operation().deref(ctx).get_result(0)
            };

        // The split puts the scan (and everything after it) in `after_gs`; the other five
        // blocks are created in CFG order, each appended at the region end - see the reduce
        // for why the diamond must not anchor to `after_gs`.
        let after_gs = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        let region = head.deref(ctx).get_container();
        let Some(region) = region else {
            self.bail(Opcode::Scan, "scan is not inside a region");
            return Ok(());
        };
        let mut mk = |rw: &mut DialectConversionRewriter| {
            rw.create_block(ctx, BlockInsertionPoint::AtRegionEnd(region), None, vec![])
        };
        let wt = mk(rewriter);
        let skip_wt = mk(rewriter);
        let after_wt = mk(rewriter);
        let gs = mk(rewriter);
        let skip_gs = mk(rewriter);
        // `split_block` left `after_gs` where the scan sat; the walk appends its
        // counterpart block last, so move it there (see the reduce for why it matters).
        rewriter.move_block(ctx, after_gs, BlockInsertionPoint::AtRegionEnd(region));

        // tail of the head: group index, then the phase-2 branch. `create_block` moves the
        // insertion point, so pin it to the head explicitly: these feed the branch the
        // head takes, and landing them in a split-off block is invalid SSA.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(head));
        let simd_grp = {
            let o = air::BinOp::new(ctx, i32_h, tid, c32);
            o.set_op(ctx, BinOpAttr::UDiv);
            rewriter.insert_operation(ctx, o.get_operation());
            o.get_operation().deref(ctx).get_result(0)
        };
        let boundary = if reverse { 0 } else { 31 };
        let c_bnd = const_i(ctx, boundary);
        let is_bnd = icmp(ctx, rewriter, ICmpPredAttr::Eq, lane, c_bnd);
        let cbr0 = air::CondBrOp::new(ctx, is_bnd, wt, vec![], skip_wt, vec![]);
        cbr0.get_operation().insert_at_back(head, ctx);

        // Phase 2: the boundary lane writes its group total.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(wt));
        let g = partial_gep(ctx, rewriter, simd_grp);
        let st = air::StoreOp::new(ctx, cur, g);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_wt, vec![]);
        b.get_operation().insert_at_back(wt, ctx);
        let b = air::BrOp::new(ctx, after_wt, vec![]);
        b.get_operation().insert_at_back(skip_wt, ctx);

        // Phase 3: group 0 prefix-scans the group totals.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_wt));
        barrier(ctx, rewriter);
        let is_grp0 = icmp(ctx, rewriter, ICmpPredAttr::Eq, simd_grp, zero);
        let cbr1 = air::CondBrOp::new(ctx, is_grp0, gs, vec![], skip_gs, vec![]);
        cbr1.get_operation().insert_at_back(after_wt, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(gs));
        let g = partial_gep(ctx, rewriter, lane);
        let ld = air::LoadOp::new(ctx, cty_h, g);
        air::set_align(ld.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, ld.get_operation());
        let gt_raw = ld.get_operation().deref(ctx).get_result(0);
        let c_ng = const_i(ctx, n_groups);
        let oob = icmp(ctx, rewriter, ICmpPredAttr::Uge, lane, c_ng);
        let idc = air::ConstBitsOp::new(ctx, cty_h, identity_bits);
        rewriter.insert_operation(ctx, idc.get_operation());
        let identity = idc.get_operation().deref(ctx).get_result(0);
        let mut gv = select(ctx, rewriter, oob, identity, gt_raw, cty_h);
        for &gd in &[1i64, 2, 4, 8, 16] {
            let gshuf = shuffle_call(ctx, rewriter, gv, gd);
            let c_lim = const_i(ctx, 32 - gd);
            let c_gd = const_i(ctx, gd);
            let gc = if reverse {
                icmp(ctx, rewriter, ICmpPredAttr::Uge, lane, c_lim)
            } else {
                icmp(ctx, rewriter, ICmpPredAttr::Ult, lane, c_gd)
            };
            let gcomb = self.emit_combine(ctx, rewriter, gshuf, gv, intrinsic, cty_h);
            gv = select(ctx, rewriter, gc, gv, gcomb, cty_h);
        }
        let g = partial_gep(ctx, rewriter, lane);
        let st = air::StoreOp::new(ctx, gv, g);
        air::set_align(st.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_gs, vec![]);
        b.get_operation().insert_at_back(gs, ctx);
        let b = air::BrOp::new(ctx, after_gs, vec![]);
        b.get_operation().insert_at_back(skip_gs, ctx);

        // Phase 4: the exclusive group prefix, combined with the within-group result.
        // The scan sits at the head of `after_gs`; the phase goes in front of it.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        barrier(ctx, rewriter);
        let edge_grp = if reverse { n_groups - 1 } else { 0 };
        let c_edge = const_i(ctx, edge_grp);
        let is_edge = icmp(ctx, rewriter, ICmpPredAttr::Eq, simd_grp, c_edge);
        let one = const_i(ctx, 1);
        let adj_raw = {
            let o = if reverse {
                let o = air::BinOp::new(ctx, i32_h, simd_grp, one);
                o.set_op(ctx, BinOpAttr::Add);
                o
            } else {
                let o = air::BinOp::new(ctx, i32_h, simd_grp, one);
                o.set_op(ctx, BinOpAttr::Sub);
                o
            };
            rewriter.insert_operation(ctx, o.get_operation());
            o.get_operation().deref(ctx).get_result(0)
        };
        let adj_safe = select(ctx, rewriter, is_edge, zero, adj_raw, i32_h);
        let g = partial_gep(ctx, rewriter, adj_safe);
        let ld = air::LoadOp::new(ctx, cty_h, g);
        air::set_align(ld.get_operation(), ctx, al);
        rewriter.insert_operation(ctx, ld.get_operation());
        let adj_loaded = ld.get_operation().deref(ctx).get_result(0);
        let group_prefix = select(ctx, rewriter, is_edge, identity, adj_loaded, cty_h);
        let result = self.emit_combine(ctx, rewriter, group_prefix, cur, intrinsic, cty_h);
        let result = self.fpcast_if(ctx, rewriter, result, &cty, &rty);
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        self.converted += 1;
        Ok(())
    }
    /// The runs of adjacent `[32, 32]` column reduces `walk.rs` lowers as one shared scratch
    /// round trip, planned BEFORE conversion.
    ///
    /// `walk.rs::batch_column_reduces` takes a maximal run of consecutive column reduces -
    /// same shape, sources in hand, none lane-invariant - and serves it with three barriers
    /// instead of three per member: one transposed pass per scratch chunk, then one compact
    /// redistribution that overlays the phase-1 blocks. The conversion must emit the
    /// identical shared spelling, and for the same reason the lane-vector runs are planned:
    /// the driver reaches the members in its own order, so the run is decided on the
    /// pre-conversion function and the emission lands at the HEAD's position whatever
    /// member is processed first.
    pub(crate) fn collect_reduce_batches(&mut self, ctx: &Context, func: Ptr<Operation>) {
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        for_each_block(ctx, func, &mut |ops| {
            for (i, &op) in ops.iter().enumerate() {
                if try_opcode_of(ctx, op) != Some(Opcode::Reduce) {
                    continue;
                }
                let mut members: Vec<Ptr<Operation>> = Vec::new();
                let mut info: Vec<(&'static str, String, LlType, LlType, u32)> = Vec::new();
                for &cand in &ops[i..] {
                    if try_opcode_of(ctx, cand) != Some(Opcode::Reduce) {
                        break;
                    }
                    let operands: Vec<pliron::value::Value> = cand.deref(ctx).operands().collect();
                    let Some(&src_v) = operands.first() else {
                        break;
                    };
                    let Some(WT::Tile(tt)) = wax_type_of(ctx, src_v.get_type(ctx)) else {
                        break;
                    };
                    let tt = &tt;
                    if tt.shape.len() != 2 || tt.shape[1] != 32 || tt.shape[0] != 32 {
                        break;
                    }
                    if self.wax_attr_i64(ctx, cand, "dim") != Some(0) {
                        break;
                    }
                    // A lane-invariant source skips the redistribution and is cheap already;
                    // `walk.rs` breaks the run there.
                    if self.reduce.reduce_sources
                        .get(&cand)
                        .is_some_and(|s| !self.reduce.divergent.contains(s))
                    {
                        break;
                    }
                    let o = cand.deref(ctx);
                    let Some(res) = (o.get_num_results() >= 1).then(|| o.get_result(0)) else {
                        break;
                    };
                    let res_ty = res.get_type(ctx);
                    drop(o);
                    let Some((cty, suf, rty)) = self.intrin_ty(ctx, res_ty) else {
                        break;
                    };
                    let base = self.combine_base(ctx, cand);
                    let al = scalar_align(&cty);
                    members.push(cand);
                    info.push((base, suf.to_string(), cty, rty, al));
                }
                if members.len() < 2 {
                    continue;
                }
                let record = self.reduce.reduce_batch_records.len();
                for (slot, &member) in members.iter().enumerate() {
                    self.reduce.reduce_batches.insert(member, (slot, record));
                }
                self.reduce.reduce_batch_records.push(ReduceBatch {
                    head: op,
                    members,
                    info,
                });
            }
        });
    }
}
