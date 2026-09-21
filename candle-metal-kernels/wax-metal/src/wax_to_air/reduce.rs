//! Reduce and scan support. Columned, batched, and hierarchical reduces, inclusive scan, and
//! the threadgroup scratch they use.

// TODO: Support non-f32 scratch/reduce.

use super::*;
use crate::air_dialect::attrs::BinKind;
use crate::air_dialect::type_interfaces::{align_bytes, bitcode_ty, special_bits};
use pliron::r#type::Typed;
use pliron::value::Value;
use std::collections::HashMap;
use wax_ir::dialect::ops::BITS_16;

/// Threadgroup scratch row pitch. 32 lanes + 1. To avoid bank conflicts.
pub(crate) const SCRATCH_PITCH: i64 = 33;

/// Alignment for threadgroup scratch buffer for provided compute type.
// TODO: Alignment of type. Should be simple to generalize.
fn scratch_align(ctx: &Context, cty: TypeHandle) -> Option<u32> {
    if cty == t::f32_ty(ctx) || cty == t::i32_ty(ctx) {
        Some(4)
    } else if cty == t::f16_ty(ctx) || cty == t::bf16_ty(ctx) {
        Some(2)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct BatchId(pub(crate) usize);

#[derive(Clone, Copy)]
pub(crate) struct ReduceMember {
    base: &'static str,
    rty: TypeHandle,
}

/// Batched `[32, 32]` column reduces lowered as a single shared scratch round trip.
/// Emission location, the reduce members, and each members' result type and reduce intrinsic.
#[derive(Clone)]
pub(crate) struct ReduceBatch {
    head: Ptr<Operation>,
    members: Vec<Ptr<Operation>>,
    info: Vec<ReduceMember>,
}

/// The combine/reduce intrinsic.
#[derive(Clone, Copy)]
pub(crate) struct Intrinsic {
    pub(crate) base: &'static str,
    pub(crate) suf: &'static str,
}

/// Shared state for the reduce emitters.
#[derive(Clone, Copy)]
pub(crate) struct ReduceCtx {
    pub(crate) src: Value,
    pub(crate) intrinsic: Intrinsic,
    pub(crate) cty: TypeHandle,
    pub(crate) rty: TypeHandle,
}

/// Shared scratch round trip scaffolding for reduces.
pub(crate) struct ScratchRoundTrip {
    ptr_ty: TypeHandle,
    base_ptr: Value,
    pub(crate) scratch_n: u32,
    zero: Value,
    group: Value,
    pub(crate) lane: Value,
    c_pitch: Value,
}

impl ScratchRoundTrip {
    /// A gep into `@_tg_user_scratch` at `idx` float units in.
    pub(crate) fn scratch_gep(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        idx: Value,
    ) -> Value {
        let f32_ty = t::f32_ty(ctx);
        let g = air::GepArrayOp::new(ctx, self.ptr_ty, self.base_ptr, idx, f32_ty, self.scratch_n);
        emit_op(ctx, rw, g)
    }
}

pub(crate) fn record_column_reduce_sources(
    ctx: &Context,
    func: Ptr<Operation>,
) -> HashMap<Ptr<Operation>, Value> {
    let mut out = HashMap::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if opcode_of(ctx, op) != Some(Opcode::Reduce) {
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

/// The `dim` attribute of a reduce, if present.
pub(crate) fn reduce_dim_of(ctx: &Context, op: Ptr<Operation>) -> Option<i64> {
    wax_int_attr(ctx, op, &keys::ATTR_DIM)
}

/// The init element of a reduce, as bits at the compute type.
// TODO: Rename to `reduce_identity_bits` to reflect cuda-tile's `identities` attribute.
pub(crate) fn reduce_init_bits(ctx: &Context, base: &str, cty: TypeHandle) -> Option<u64> {
    use crate::air_dialect::type_interfaces::Special;
    let v = match base {
        "@air.simd_max" => Special::NegInf,
        "@air.simd_min" => Special::PosInf,
        "@air.simd_product" => Special::One,
        "@air.simd_sum" => Special::Zero,
        _ => return None,
    };
    special_bits(ctx, cty, v)
}

/// The element count of `@_tg_user_scratch` for this kernel.
pub(crate) fn tg_scratch_floats_of(ctx: &Context, func: Ptr<Operation>) -> u32 {
    const DEFAULT: u32 = 128 * 33; // TG_SCRATCH_FLOATS_DEFAULT
    use wax_ir::attr::Attribute as WA;
    use wax_ir::dialect::attr_mirror::WaxAttr;
    let rows = func
        .deref(ctx)
        .attributes
        .get::<WaxAttr>(&keys::ATTR_TG_SCRATCH_ROWS)
        .and_then(|a| match &a.0 {
            WA::DenseI32Array(arr) => arr.first().copied().map(|v| v as i64),
            _ => None,
        });
    // Treat malformed row count same as missing (default).
    let asked = rows
        .filter(|r| *r > 0)
        .and_then(|r| u32::try_from(r * SCRATCH_PITCH).ok())
        .unwrap_or(DEFAULT);
    asked.max(broadcast_scratch_floats_of(ctx, func))
}

/// Required elements to perform broadcast redistribution.
/// The widest source tile plus dump slot.
fn broadcast_scratch_floats_of(ctx: &Context, func: Ptr<Operation>) -> u32 {
    let mut need = 0u32;
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if opcode_of(ctx, op) != Some(Opcode::Broadcast) {
                continue;
            }
            let shapes = (|| {
                let s = match wax_type_of(ctx, op.deref(ctx).operands().next()?.get_type(ctx))? {
                    WaxType::Tile(t) => t.shape,
                    _ => return None,
                };
                let d = match wax_type_of(ctx, op.deref(ctx).get_result(0).get_type(ctx))? {
                    WaxType::Tile(t) => t.shape,
                    _ => return None,
                };
                Some((s, d))
            })();
            let Some((s, d)) = shapes else { continue };
            if broadcast_is_forwardable(&s, &d) {
                continue;
            }
            let n: i64 = s.iter().product();
            // `+ 1` for dump slot.
            if let Ok(want) = u32::try_from(n + 1) {
                need = need.max(want);
            }
        }
    });
    need
}

impl WaxToAir {
    /// `wax.reduce` -> singular SIMD reduce intrinsic.
    pub(crate) fn rewrite_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let Some(src) = op.deref(ctx).operands().next() else {
            return self.refuse(Opcode::Reduce, "no source operand");
        };
        // Original source wax type.
        let src_wax = operands_info
            .lookup_most_recent_type(src)
            .or_else(|| Some(src.get_type(ctx)))
            .and_then(|h| wax_ir::dialect::types::from_pliron(ctx, h));
        let Some(src_wax) = src_wax else {
            return self.refuse(Opcode::Reduce, "source type unreadable");
        };
        let (shape, n_elems): (Vec<i64>, i64) = match &src_wax {
            WaxType::Tile(tt) => (tt.shape.clone(), tt.shape.iter().product()),
            _ => (Vec::new(), 1),
        };
        let dim = wax_int_attr(ctx, op, &keys::ATTR_DIM);
        let spans_groups = shape.len() == 2 && shape[1] == 32 && shape[0] > 1;

        let Some(base) = combine_base(ctx, op) else {
            return self.refuse(Opcode::Reduce, "combine body is not a known reduction");
        };
        let wax_res = {
            let o = op.deref(ctx);
            (o.get_num_results() >= 1).then(|| o.get_result(0).get_type(ctx))
        };
        let Some(wax_res) = wax_res else {
            return self.refuse(Opcode::Reduce, "no result");
        };
        let Some((cty, suf, rty)) = fp_intr_info(ctx, wax_res) else {
            return self.refuse(Opcode::Reduce, "result is not a float type");
        };

        // Reduce over a single number is a noop.
        if let Some(d) = dim
            && d >= 0
            && shape.get(d as usize).copied() == Some(1)
        {
            let from = src.get_type(ctx);
            let Some(v) = self.fpcast_if(ctx, rewriter, src, from, rty) else {
                return Ok(());
            };
            rewriter.replace_operation_with_values(ctx, op, vec![v]);
            self.converted += 1;
            return Ok(());
        }

        if spans_groups && dim == Some(0) {
            let rows = shape[0] as i32;
            // Emit adjacent `[32, 32]` reduces as a batch.
            if self.reduce.of_member.contains_key(&op) {
                let Some(src) = self.fpcast_if(ctx, rewriter, src, rty, cty) else {
                    return Ok(());
                };
                return self.emit_reduce_batch(ctx, rewriter, op, src);
            }
            let rc = ReduceCtx {
                src,
                intrinsic: Intrinsic { base, suf },
                cty,
                rty,
            };
            return self.column_reduce(ctx, rewriter, op, rc, rows);
        }
        // If reduce is over mma fragments we emit fragment compatible reduce.
        if self.frag_dims_of_grid(ctx, operands_info, src).is_some() {
            return self.fragment_row_reduce(
                ctx,
                rewriter,
                op,
                operands_info,
                ReduceCtx {
                    src,
                    intrinsic: Intrinsic { base, suf },
                    cty,
                    rty,
                },
            );
        }
        let row_reduce = spans_groups && dim == Some(1);
        let res_elems: i64 = match &wax_ir::dialect::types::from_pliron(ctx, wax_res) {
            Some(WaxType::Tile(tt)) => tt.shape.iter().product(),
            _ => 1,
        };
        if n_elems > 32 && !row_reduce {
            let rc = ReduceCtx {
                src,
                intrinsic: Intrinsic { base, suf },
                cty,
                rty,
            };
            // Dim-1 reduce of a `[rows, cols]` tile with `cols < 32` is per-row.
            let cols = shape.get(1).copied().unwrap_or(0);
            let rows = shape.first().copied().unwrap_or(0);
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
                return self.refuse(Opcode::Reduce, "no lowering for multi-element result here");
            }
            return self.reduce_hierarchical(ctx, rewriter, op, rc);
        }

        let Some(arg) = self.fpcast_if(ctx, rewriter, src, rty, cty) else {
            return Ok(());
        };
        // A tile narrower than the SIMD group is only valid here when replicated across lanes.
        // This only applies for lane term `tp.x & (n-1)` with widths that are power-of-two.
        // Otherwise there is no cross group broadcast.
        if n_elems < 32 && !row_reduce && !(n_elems as u64).is_power_of_two() {
            return self.refuse(Opcode::Reduce, "narrow tile is not power of two");
        }
        // `@air.simd_*` reduces all 32 lanes. Apply mask if needed.
        let mask_replicas =
            n_elems < 32 && !row_reduce && base != "@air.simd_max" && base != "@air.simd_min";
        let arg = if mask_replicas {
            let Some(tp) = self.control.tp else {
                return self.refuse(Opcode::Reduce, "no thread position parameter");
            };
            let i32_ty = t::i32_ty(ctx);
            let i1_ty = t::i1_ty(ctx);
            let zero = air_const_i32(ctx, 0);
            let tid = {
                let e = air::extract_elem(ctx, i32_ty, tp, zero);
                emit(ctx, rewriter, e)
            };
            let c32 = air_const_i32(ctx, 32);
            let lane = air_bin_i32(ctx, rewriter, tid, c32, BinKind::URem);
            let n_c = air_const_i32(ctx, n_elems);
            let keep = {
                let c = air::icmp(
                    ctx,
                    pliron_llvm::attributes::ICmpPredicateAttr::ULT,
                    i1_ty,
                    lane,
                    n_c,
                );
                emit(ctx, rewriter, c)
            };
            let Some(init) = reduce_init_bits(ctx, base, cty) else {
                return self.refuse(Opcode::Reduce, "no reduce init for type");
            };
            let Some(id) = air_const(ctx, cty, init) else {
                return self.refuse(Opcode::Reduce, "no constant for type");
            };
            let sel = air::select(ctx, cty, keep, arg, id);
            emit(ctx, rewriter, sel)
        } else {
            arg
        };
        let call = air::call_fast(ctx, &format!("{base}.{suf}"), Some(cty), vec![arg]);
        let r = emit(ctx, rewriter, call);
        let Some(r) = self.fpcast_if(ctx, rewriter, r, cty, rty) else {
            return Ok(());
        };
        // Row reduce leaves the result in all 32 lanes. Mark flag here.
        if row_reduce {
            self.reduce.row_replicated.insert(r);
        }
        rewriter.replace_operation_with_values(ctx, op, vec![r]);
        self.converted += 1;
        Ok(())
    }
    /// Record the scratch global a column reduce needs, once.
    pub(crate) fn needs_tg_scratch_mark(&mut self) {
        let floats = self.reduce.tg_scratch_floats;
        self.reduce
            .declare_global("@_tg_user_scratch", crate::bitcode::Ty::Float, floats);
    }
    /// Scratch round trip scaffolding.
    pub(crate) fn scratch_round_trip(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
    ) -> Result<Option<ScratchRoundTrip>> {
        use crate::air_dialect::attrs::BinKind;
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Reduce, "no thread position parameter");
            return Ok(None);
        };
        self.needs_tg_scratch_mark();
        let i32_ty = t::i32_ty(ctx);
        let ptr_ty = t::ptr_ty(ctx, t::f32_ty(ctx), 3);
        // Wait for readers of the scratch before writing.
        air_wg_barrier(ctx, rewriter);
        let zero = air_const_i32(ctx, 0);
        let tid = {
            let e = air::extract_elem(ctx, i32_ty, tp, zero);
            emit(ctx, rewriter, e)
        };
        let c32 = air_const_i32(ctx, 32);
        let c31 = air_const_i32(ctx, 31);
        let c_pitch = air_const_i32(ctx, SCRATCH_PITCH);
        let group = air_bin_i32(ctx, rewriter, tid, c32, BinKind::UDiv);
        let lane = air_bin_i32(ctx, rewriter, tid, c31, BinKind::And);
        // The reference to the scratch is a module value, not an instruction.
        let base_ptr = air::GlobalRefOp::new(ctx, ptr_ty, "@_tg_user_scratch");
        let base_ptr = emit_op(ctx, rewriter, base_ptr);
        Ok(Some(ScratchRoundTrip {
            ptr_ty,
            base_ptr,
            scratch_n: self.reduce.tg_scratch_floats,
            zero,
            group,
            lane,
            c_pitch,
        }))
    }
    /// One batch of `[32, 32]` column reduces.
    pub(crate) fn emit_reduce_batch(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        this_src: pliron::value::Value,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        use pliron::irbuild::inserter::OpInsertionPoint;
        // Already covered by previously emitted member.
        if let Some(&out) = self.reduce.batch_results.get(&op) {
            rewriter.replace_operation_with_values(ctx, op, vec![out]);
            return Ok(());
        }
        let Some(&(batch, _slot)) = self.reduce.of_member.get(&op) else {
            return self.refuse(Opcode::Reduce, "reduce member has no planned batch");
        };
        let ReduceBatch {
            head,
            members,
            info,
        } = self.reduce.batches[batch.0].clone();
        // Set insertion point before head if needed.
        if head != op {
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(head));
        }
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let f32_ty = t::f32_ty(ctx);
        let group = rt.group;
        let lane = rt.lane;
        let c_pitch = rt.c_pitch;

        let n = members.len();
        let per_member = 32i64 * SCRATCH_PITCH;
        let chunk = (self.reduce.tg_scratch_floats as i64 / per_member).max(1) as usize;
        let mut per_group: Vec<Value> = Vec::with_capacity(n);
        for (ci, start) in (0..n).step_by(chunk).enumerate() {
            let sub: Vec<usize> = (start..n.min(start + chunk)).collect();
            let sub = &sub[..];
            if ci > 0 {
                air_wg_barrier(ctx, rewriter);
            }
            // 1: All members store from `src_v` to `[row = simdgroup][col = lane]` in shared mem.
            for &gi in sub {
                let m = members[gi];
                let mut src_v = if m == op {
                    this_src
                } else {
                    let Some(v) = m.deref(ctx).operands().next() else {
                        return self.refuse(Opcode::Reduce, "no source operand");
                    };
                    v
                };
                // Widen if needed.
                let vt = src_v.get_type(ctx);
                if vt != t::f32_ty(ctx) {
                    src_v = match self.fpcast_if(ctx, rewriter, src_v, vt, t::f32_ty(ctx)) {
                        Some(v) => v,
                        None => return Ok(()),
                    };
                }
                let region = air_const_i32(ctx, gi as i64 * per_member);
                let base_off = air_bin_i32(ctx, rewriter, group, c_pitch, BinKind::Mul);
                let flat = air_bin_i32(ctx, rewriter, base_off, lane, BinKind::Add);
                let flat = air_bin_i32(ctx, rewriter, flat, region, BinKind::Add);
                let gep = rt.scratch_gep(ctx, rewriter, flat);
                let st = air::StoreOp::new(ctx, src_v, gep, 4);
                rewriter.insert_operation(ctx, st.get_operation());
            }
            air_wg_barrier(ctx, rewriter);
            // 2: Transposed read means reduce can be within simdgroup.
            for &gi in sub {
                let base = info[gi].base;
                let region = air_const_i32(ctx, gi as i64 * per_member);
                let row_off = air_bin_i32(ctx, rewriter, lane, c_pitch, BinKind::Mul);
                let flat_t = air_bin_i32(ctx, rewriter, row_off, group, BinKind::Add);
                let flat_t = air_bin_i32(ctx, rewriter, flat_t, region, BinKind::Add);
                let gep_t = rt.scratch_gep(ctx, rewriter, flat_t);
                let ld = air::LoadOp::new(ctx, f32_ty, gep_t, 4);
                let v = emit_op(ctx, rewriter, ld);
                let call = air::call_fast(ctx, &format!("{base}.f32"), Some(f32_ty), vec![v]);
                per_group.push(emit(ctx, rewriter, call));
            }
        }
        // Redistribute uniform.
        air_wg_barrier(ctx, rewriter);
        for (i, &pg) in per_group.iter().enumerate() {
            let off = air_const_i32(ctx, i as i64 * 32);
            let slot = air_bin_i32(ctx, rewriter, group, off, BinKind::Add);
            let gep_w = rt.scratch_gep(ctx, rewriter, slot);
            let st2 = air::StoreOp::new(ctx, pg, gep_w, 4);
            rewriter.insert_operation(ctx, st2.get_operation());
        }
        air_wg_barrier(ctx, rewriter);
        for (i, &m) in members.iter().enumerate() {
            let rty = info[i].rty;
            let off = air_const_i32(ctx, i as i64 * 32);
            let slot = air_bin_i32(ctx, rewriter, lane, off, BinKind::Add);
            let gep_r = rt.scratch_gep(ctx, rewriter, slot);
            let ld2 = air::LoadOp::new(ctx, f32_ty, gep_r, 4);
            let out = emit_op(ctx, rewriter, ld2);
            let Some(out) = self.fpcast_if(ctx, rewriter, out, t::f32_ty(ctx), rty) else {
                return Ok(());
            };
            self.reduce.batch_results.insert(m, out);
        }
        let Some(out) = self.reduce.batch_results.get(&op).copied() else {
            return self.refuse(Opcode::Reduce, "reduce batch does not list its own member");
        };
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// Dim-0 reduce of a `[rows, 32]` tile.
    pub(crate) fn column_reduce(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
        rows: i32,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let ReduceCtx {
            src,
            intrinsic,
            cty,
            rty,
        } = rc;
        let Intrinsic { base, suf } = intrinsic;
        // Widen source if needed.
        let Some(src) = self.fpcast_if(ctx, rewriter, src, rty, cty) else {
            return Ok(());
        };
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        // Widen output if needed (for f32 shared memory).
        let Some(src) = self.fpcast_if(ctx, rewriter, src, cty, t::f32_ty(ctx)) else {
            return Ok(());
        };
        let al = 4;
        let st_suf = if cty == t::f16_ty(ctx) { "f32" } else { suf };
        let f32_ty = t::f32_ty(ctx);

        // Store this thread's element at `[row = simdgroup][col = lane]` in shared mem.
        let base_off = air_bin_i32(ctx, rewriter, rt.group, rt.c_pitch, BinKind::Mul);
        let flat = air_bin_i32(ctx, rewriter, base_off, rt.lane, BinKind::Add);
        let gep = rt.scratch_gep(ctx, rewriter, flat);
        let st = air::StoreOp::new(ctx, src, gep, al);
        rewriter.insert_operation(ctx, st.get_operation());
        air_wg_barrier(ctx, rewriter);

        if rows == 32 {
            // Transposed read means reduce can be within simdgroup.
            let row_off = air_bin_i32(ctx, rewriter, rt.lane, rt.c_pitch, BinKind::Mul);
            let flat_t = air_bin_i32(ctx, rewriter, row_off, rt.group, BinKind::Add);
            let gep_t = rt.scratch_gep(ctx, rewriter, flat_t);
            let ld = air::LoadOp::new(ctx, f32_ty, gep_t, al);
            let v = emit_op(ctx, rewriter, ld);
            let call = air::call_fast(ctx, &format!("{base}.{st_suf}"), Some(f32_ty), vec![v]);
            let per_group = emit(ctx, rewriter, call);

            // If uniform there's no need to redistribute.
            let uniform = self
                .reduce
                .reduce_sources
                .get(&op)
                .is_some_and(|s| !self.reduce.divergent.contains(s));
            if uniform {
                let Some(out) = self.fpcast_if(ctx, rewriter, per_group, f32_ty, rty) else {
                    return Ok(());
                };
                rewriter.replace_operation_with_values(ctx, op, vec![out]);
                self.converted += 1;
                return Ok(());
            }

            // Redistribute.
            air_wg_barrier(ctx, rewriter);
            let gep_w = rt.scratch_gep(ctx, rewriter, flat);
            let st2 = air::StoreOp::new(ctx, per_group, gep_w, al);
            rewriter.insert_operation(ctx, st2.get_operation());
            air_wg_barrier(ctx, rewriter);
            let flat_r = air_bin_i32(ctx, rewriter, rt.lane, rt.c_pitch, BinKind::Mul);
            let gep_r = rt.scratch_gep(ctx, rewriter, flat_r);
            let ld2 = air::LoadOp::new(ctx, f32_ty, gep_r, al);
            let out = emit_op(ctx, rewriter, ld2);
            // Narrow.
            let Some(out) = self.fpcast_if(ctx, rewriter, out, t::f32_ty(ctx), rty) else {
                return Ok(());
            };
            rewriter.replace_operation_with_values(ctx, op, vec![out]);
            self.converted += 1;
            return Ok(());
        }

        // Reduce.
        let flat0 = air_bin_i32(ctx, rewriter, rt.zero, rt.lane, BinKind::Add);
        let g0 = rt.scratch_gep(ctx, rewriter, flat0);
        let l0 = air::LoadOp::new(ctx, f32_ty, g0, al);
        let mut acc = emit_op(ctx, rewriter, l0);
        for r in 1..rows {
            let off = air_const_i32(ctx, r as i64 * SCRATCH_PITCH);
            let flat_r = air_bin_i32(ctx, rewriter, off, rt.lane, BinKind::Add);
            let g = rt.scratch_gep(ctx, rewriter, flat_r);
            let l = air::LoadOp::new(ctx, f32_ty, g, al);
            let v = emit_op(ctx, rewriter, l);
            acc = emit_combine(ctx, rewriter, acc, v, intrinsic, f32_ty);
        }
        air_wg_barrier(ctx, rewriter);
        // Narrow.
        let Some(out) = self.fpcast_if(ctx, rewriter, acc, t::f32_ty(ctx), rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// Dim-1 (per-row) reduce of a `[rows, cols]` tile, `1 < cols < 32`, `rows` power-of-two.
    /// Butterfly fold each row in registers, then transform layout to result tile's via shared
    /// memory.
    pub(crate) fn row_reduce_scratch(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
        rows: i64,
        cols: i64,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        let ReduceCtx {
            src,
            intrinsic,
            cty,
            rty,
        } = rc;
        let Some(src) = self.fpcast_if(ctx, rewriter, src, cty, t::f32_ty(ctx)) else {
            return Ok(());
        };
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let f32_ty = t::f32_ty(ctx);
        let lane = rt.lane;
        let grp = rt.group;

        // 1: Within row butterfly, in registers.
        let mut v = src;
        let mut mask: i64 = 1;
        while mask < cols {
            let m = air_const_int(ctx, BITS_16, mask as u64);
            let partner = {
                let c = air::call(
                    ctx,
                    crate::lower::AIR_SIMD_SHUFFLE_XOR_F32,
                    Some(f32_ty),
                    vec![v, m],
                );
                emit(ctx, rewriter, c)
            };
            v = emit_combine(ctx, rewriter, v, partner, intrinsic, f32_ty);
            mask *= 2;
        }

        // 2: Park each row in its shared memory slot, past the column region (32 rows of pitch).
        let rows_pg = 32 / cols;
        let region = air_const_i32(ctx, 32 * SCRATCH_PITCH);
        let row_of_lane = {
            let rp = air_const_i32(ctx, rows_pg);
            let base = air_bin_i32(ctx, rewriter, grp, rp, BinKind::Mul);
            let cs = air_const_i32(ctx, cols);
            let half = air_bin_i32(ctx, rewriter, lane, cs, BinKind::UDiv);
            air_bin_i32(ctx, rewriter, base, half, BinKind::Add)
        };
        let flat = air_bin_i32(ctx, rewriter, region, row_of_lane, BinKind::Add);
        let gep_w = rt.scratch_gep(ctx, rewriter, flat);
        let st = air::StoreOp::new(ctx, v, gep_w, 4);
        rewriter.insert_operation(ctx, st.get_operation());
        air_wg_barrier(ctx, rewriter);

        // 3: `[rows]` store maps lane `l` to element `l mod rows`.
        // `rows` is a power-of-two, so `mod` == `and`.
        let slot = {
            let m = air_const_i32(ctx, rows - 1);
            let l = air_bin_i32(ctx, rewriter, lane, m, BinKind::And);
            air_bin_i32(ctx, rewriter, region, l, BinKind::Add)
        };
        let gep_r = rt.scratch_gep(ctx, rewriter, slot);
        let ld = air::LoadOp::new(ctx, f32_ty, gep_r, 4);
        let out = emit_op(ctx, rewriter, ld);
        let Some(out) = self.fpcast_if(ctx, rewriter, out, t::f32_ty(ctx), rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
    /// Whole tile reduce via threadgroup memory.
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
    pub(crate) fn reduce_hierarchical(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rc: ReduceCtx,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::LinkedList;
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let ReduceCtx {
            src,
            intrinsic,
            cty,
            rty,
        } = rc;
        let Intrinsic { base, suf } = intrinsic;
        let Some(head) = op.deref(ctx).get_container() else {
            return self.refuse(Opcode::Reduce, "reduce is not inside a block");
        };
        let Some(region) = head.deref(ctx).get_container() else {
            return self.refuse(Opcode::Reduce, "block is not inside a region");
        };
        let al = match scratch_align(ctx, cty) {
            Some(a) => a,
            None => {
                return self.refuse(Opcode::Reduce, "unsupported element type");
            }
        };
        if !self.reduce.has_global("@_tg_reduce_partial") {
            let Some(t) = bitcode_ty(ctx, cty) else {
                return self.refuse(Opcode::Reduce, "reduce type has no bitcode serialisation");
            };
            self.reduce
                .declare_global("@_tg_reduce_partial", t.clone(), 32);
            self.reduce.declare_global("@_tg_reduce_result", t, 1);
        }
        let i32_ty = t::i32_ty(ctx);
        let ptr_ty = t::ptr_ty(ctx, cty, 3);
        let intrinsic = format!("{base}.{suf}");

        // Ensure correct block ordering.
        let after_final =
            rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
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
        // Move our reduce block to after the blocks we created
        rewriter.move_block(ctx, after_final, BlockInsertionPoint::AtRegionEnd(region));

        let partial_ptr = air::GlobalRefOp::new(ctx, ptr_ty, "@_tg_reduce_partial");
        let partial_ptr = emit_op(ctx, rewriter, partial_ptr);
        let result_ptr = air::GlobalRefOp::new(ctx, ptr_ty, "@_tg_reduce_result");
        let result_ptr = emit_op(ctx, rewriter, result_ptr);

        let partial_gep =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, idx: Value| -> Value {
                let g = air::GepArrayOp::new(ctx, ptr_ty, partial_ptr, idx, cty, 32);
                emit_op(ctx, rw, g)
            };
        let simd_reduce =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, v: Value| -> Value {
                let c = air::call_fast(ctx, &intrinsic, Some(cty), vec![v]);
                emit(ctx, rw, c)
            };

        // head: within SIMD reduce, and group/lane indices.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(head));
        let s1 = simd_reduce(ctx, rewriter, src);
        let Some(tp) = self.control.tp else {
            return self.refuse(Opcode::Reduce, "no thread position parameter");
        };
        let zero = air_const_i32(ctx, 0);
        let tid = {
            let e = air::extract_elem(ctx, i32_ty, tp, zero);
            emit(ctx, rewriter, e)
        };
        let c32 = air_const_i32(ctx, 32);
        let group = {
            let o = air::bin(ctx, BinKind::UDiv, i32_ty, tid, c32);
            emit(ctx, rewriter, o)
        };
        let lane = {
            let o = air::bin(ctx, BinKind::URem, i32_ty, tid, c32);
            emit(ctx, rewriter, o)
        };
        // `group == 0 { init } else { skip_init }`
        let is_grp0_a = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, group, zero);
        let cbr0 = air::CondBrOp::new(ctx, is_grp0_a, init, vec![], skip_init, vec![]);
        cbr0.get_operation().insert_at_back(head, ctx);

        // Extract identity bits
        let Some(identity_bits) = reduce_init_bits(ctx, base, cty) else {
            return self.refuse(Opcode::Reduce, "no reduce identity at the compute type");
        };
        let Some(identity) = air_const(ctx, cty, identity_bits) else {
            return self.refuse(Opcode::Reduce, "no constant at the compute type");
        };
        // Ensure gep is inserted correctly
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(init));
        let g = partial_gep(ctx, rewriter, lane);
        let st = air::StoreOp::new(ctx, identity, g, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_init, vec![]);
        b.get_operation().insert_at_back(init, ctx);
        let b = air::BrOp::new(ctx, after_init, vec![]);
        b.get_operation().insert_at_back(skip_init, ctx);

        // after_init: lane 0 stores partial.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_init));
        air_wg_barrier(ctx, rewriter);
        let is_lane0_a = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, lane, zero);
        let cbr1 = air::CondBrOp::new(ctx, is_lane0_a, write, vec![], skip_write, vec![]);
        cbr1.get_operation().insert_at_back(after_init, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(write));
        let g = partial_gep(ctx, rewriter, group);
        let st = air::StoreOp::new(ctx, s1, g, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_write, vec![]);
        b.get_operation().insert_at_back(write, ctx);
        let b = air::BrOp::new(ctx, after_write, vec![]);
        b.get_operation().insert_at_back(skip_write, ctx);

        // after_write: group 0 reduces partials, group 0 lane 0 writes result.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_write));
        air_wg_barrier(ctx, rewriter);
        let is_grp0_b = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, group, zero);
        let cbr2 = air::CondBrOp::new(ctx, is_grp0_b, final_b, vec![], skip_final, vec![]);
        cbr2.get_operation().insert_at_back(after_write, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(final_b));
        let g = partial_gep(ctx, rewriter, lane);
        let ld = air::LoadOp::new(ctx, cty, g, al);
        let v = emit_op(ctx, rewriter, ld);
        let total = simd_reduce(ctx, rewriter, v);
        let is_lane0_b = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, lane, zero);
        let cbr3 = air::CondBrOp::new(ctx, is_lane0_b, write_res, vec![], skip_res, vec![]);
        cbr3.get_operation().insert_at_back(final_b, ctx);
        let st = air::StoreOp::new(ctx, total, result_ptr, al);
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

        // after_final: everyone reads broadcasted result.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        air_wg_barrier(ctx, rewriter);
        let ld = air::LoadOp::new(ctx, cty, result_ptr, al);
        let v = emit_op(ctx, rewriter, ld);
        let Some(result) = self.fpcast_if(ctx, rewriter, v, cty, rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        self.converted += 1;
        Ok(())
    }
    /// Prefix scan. Within SIMD shuffle scan if possible, fall back to threadgroup variant.
    /// Each group scans within itself, edge lane publishes group total, group 0 prefix scans
    /// totals, and every thread combines its result with the exclusive group prefix.
    pub(crate) fn rewrite_scan(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::LinkedList;
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let Some(src) = op.deref(ctx).operands().next() else {
            return self.refuse(Opcode::Scan, "no source operand");
        };
        if self.frag_dims_of_grid(ctx, operands_info, src).is_some() {
            return self.refuse(Opcode::Scan, "fragment scan is not implemented");
        }
        let src_wax = operands_info
            .lookup_most_recent_type(src)
            .or_else(|| Some(src.get_type(ctx)))
            .and_then(|h| wax_type_of(ctx, h));
        let Some(src_wax) = src_wax else {
            return self.refuse(Opcode::Scan, "source type unreadable");
        };
        let n_elems: i64 = match &src_wax {
            WaxType::Tile(t) => t.shape.iter().product(),
            _ => 1,
        };
        let Some(base) = combine_base(ctx, op) else {
            return self.refuse(Opcode::Scan, "combine body is not a known reduction");
        };
        let src_ty = operands_info
            .lookup_most_recent_type(src)
            .unwrap_or_else(|| src.get_type(ctx));
        let Some((cty, suf, rty)) = fp_intr_info(ctx, src_ty) else {
            return self.refuse(Opcode::Scan, "source is not a float type");
        };
        let intrinsic = Intrinsic { base, suf };
        let reverse = wax_int_attr(ctx, op, &keys::ATTR_REVERSE) == Some(1);
        let Some(al) = scratch_align(ctx, cty) else {
            return self.refuse(Opcode::Scan, "unsupported element type");
        };
        let i32_ty = t::i32_ty(ctx);
        let shuffle = format!(
            "@air.simd_shuffle_{}.{suf}",
            if reverse { "down" } else { "up" }
        );
        let shuffle_call = |ctx: &mut Context,
                            rw: &mut DialectConversionRewriter,
                            v: Value,
                            delta: i64|
         -> Value {
            let dv = air_const_int(ctx, BITS_16, delta as u64);
            let c = air::call_fast(ctx, &shuffle, Some(cty), vec![v, dv]);
            emit(ctx, rw, c)
        };

        let Some(src) = self.fpcast_if(ctx, rewriter, src, rty, cty) else {
            return Ok(());
        };
        let Some(tp) = self.control.tp else {
            return self.refuse(Opcode::Scan, "no thread position parameter");
        };
        let zero = air_const_i32(ctx, 0);
        let tid = {
            let e = air::extract_elem(ctx, i32_ty, tp, zero);
            emit(ctx, rewriter, e)
        };
        let c32 = air_const_i32(ctx, 32);
        let lane = {
            let o = air::bin(ctx, BinKind::URem, i32_ty, tid, c32);
            emit(ctx, rewriter, o)
        };

        // Within SIMD.
        let simd_n = n_elems.min(32);
        // Tile narrower than the SIMD group is replicated across its lanes, so prefix restarts at edge.
        let lane_in_tile = if simd_n < 32 {
            let n_c = air_const_i32(ctx, simd_n);
            let m = air::bin(ctx, BinKind::URem, i32_ty, lane, n_c);
            emit(ctx, rewriter, m)
        } else {
            lane
        };
        let mut steps = Vec::new();
        let mut d = 1i64;
        while d < simd_n {
            steps.push(d);
            d <<= 1;
        }
        if steps.is_empty() {
            // n_elems == 1
            let Some(z) = air_const(ctx, cty, 0) else {
                return self.refuse(Opcode::Scan, "no constant at the compute type");
            };
            let r = emit_combine(ctx, rewriter, z, src, intrinsic, cty);
            let Some(r) = self.fpcast_if(ctx, rewriter, r, cty, rty) else {
                return Ok(());
            };
            rewriter.replace_operation_with_values(ctx, op, vec![r]);
            self.converted += 1;
            return Ok(());
        }
        let mut cur = src;
        for &delta in &steps {
            let shuf = shuffle_call(ctx, rewriter, cur, delta);
            // Lanes without a neighbor in the shuffle direction keep their value.
            let limit = air_const_i32(ctx, simd_n - delta);
            let cd = air_const_i32(ctx, delta);
            let cond = if reverse {
                air_icmp(ctx, rewriter, ICmpPredicateAttr::UGE, lane_in_tile, limit)
            } else {
                air_icmp(ctx, rewriter, ICmpPredicateAttr::ULT, lane_in_tile, cd)
            };
            let combined = emit_combine(ctx, rewriter, shuf, cur, intrinsic, cty);
            cur = air_select(ctx, rewriter, cond, cur, combined, cty);
        }
        if n_elems <= 32 {
            let Some(r) = self.fpcast_if(ctx, rewriter, cur, cty, rty) else {
                return Ok(());
            };
            rewriter.replace_operation_with_values(ctx, op, vec![r]);
            self.converted += 1;
            return Ok(());
        }

        // Threadgroup memory approach.
        if !self.reduce.has_global("@_tg_scan_partials") {
            let Some(t) = bitcode_ty(ctx, cty) else {
                return self.refuse(Opcode::Scan, "scan type has no bitcode serialisation");
            };
            self.reduce.declare_global("@_tg_scan_partials", t, 32);
        }
        let n_groups = n_elems / 32;
        let Some(identity_bits) = reduce_init_bits(ctx, base, cty) else {
            return self.refuse(Opcode::Scan, "no scan identity at the compute type");
        };
        let Some(head) = op.deref(ctx).get_container() else {
            return self.refuse(Opcode::Scan, "scan is not inside a block");
        };
        let ptr_ty = t::ptr_ty(ctx, cty, 3);
        let partials_ref = air::GlobalRefOp::new(ctx, ptr_ty, "@_tg_scan_partials");
        let partials_ref = emit_op(ctx, rewriter, partials_ref);
        let partial_gep =
            |ctx: &mut Context, rw: &mut DialectConversionRewriter, idx: Value| -> Value {
                let g = air::GepArrayOp::new(ctx, ptr_ty, partials_ref, idx, cty, 32);
                emit_op(ctx, rw, g)
            };

        // Ensure correct block ordering.
        let after_gs = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        let region = head.deref(ctx).get_container();
        let Some(region) = region else {
            return self.refuse(Opcode::Scan, "scan is not inside a region");
        };
        let mut mk = |rw: &mut DialectConversionRewriter| {
            rw.create_block(ctx, BlockInsertionPoint::AtRegionEnd(region), None, vec![])
        };
        let wt = mk(rewriter);
        let skip_wt = mk(rewriter);
        let after_wt = mk(rewriter);
        let gs = mk(rewriter);
        let skip_gs = mk(rewriter);
        // Move our scan block to after the blocks we created
        rewriter.move_block(ctx, after_gs, BlockInsertionPoint::AtRegionEnd(region));

        // Set insertion point to after head block
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(head));
        let simd_grp = {
            let o = air::bin(ctx, BinKind::UDiv, i32_ty, tid, c32);
            emit(ctx, rewriter, o)
        };
        // `if lane == edge { wt } else { skip_wt }`
        let edge = if reverse { 0 } else { 31 };
        let c_edge = air_const_i32(ctx, edge);
        let is_edge = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, lane, c_edge);
        let cbr0 = air::CondBrOp::new(ctx, is_edge, wt, vec![], skip_wt, vec![]);
        cbr0.get_operation().insert_at_back(head, ctx);

        // Edge lane writes group total.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(wt));
        let g = partial_gep(ctx, rewriter, simd_grp);
        let st = air::StoreOp::new(ctx, cur, g, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_wt, vec![]);
        b.get_operation().insert_at_back(wt, ctx);
        let b = air::BrOp::new(ctx, after_wt, vec![]);
        b.get_operation().insert_at_back(skip_wt, ctx);

        // Group 0 prefix scans group totals.
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(after_wt));
        air_wg_barrier(ctx, rewriter);
        let is_grp0 = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, simd_grp, zero);
        let cbr1 = air::CondBrOp::new(ctx, is_grp0, gs, vec![], skip_gs, vec![]);
        cbr1.get_operation().insert_at_back(after_wt, ctx);
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(gs));
        let g = partial_gep(ctx, rewriter, lane);
        let ld = air::LoadOp::new(ctx, cty, g, al);
        let gt_raw = emit_op(ctx, rewriter, ld);
        let c_ng = air_const_i32(ctx, n_groups);
        let oob = air_icmp(ctx, rewriter, ICmpPredicateAttr::UGE, lane, c_ng);
        let Some(identity) = air_const(ctx, cty, identity_bits) else {
            return self.refuse(Opcode::Scan, "no constant at the compute type");
        };
        let mut gv = air_select(ctx, rewriter, oob, identity, gt_raw, cty);
        for &gd in &[1i64, 2, 4, 8, 16] {
            let gshuf = shuffle_call(ctx, rewriter, gv, gd);
            let c_lim = air_const_i32(ctx, 32 - gd);
            let c_gd = air_const_i32(ctx, gd);
            let gc = if reverse {
                air_icmp(ctx, rewriter, ICmpPredicateAttr::UGE, lane, c_lim)
            } else {
                air_icmp(ctx, rewriter, ICmpPredicateAttr::ULT, lane, c_gd)
            };
            let gcomb = emit_combine(ctx, rewriter, gshuf, gv, intrinsic, cty);
            gv = air_select(ctx, rewriter, gc, gv, gcomb, cty);
        }
        let g = partial_gep(ctx, rewriter, lane);
        let st = air::StoreOp::new(ctx, gv, g, al);
        rewriter.insert_operation(ctx, st.get_operation());
        let b = air::BrOp::new(ctx, after_gs, vec![]);
        b.get_operation().insert_at_back(gs, ctx);
        let b = air::BrOp::new(ctx, after_gs, vec![]);
        b.get_operation().insert_at_back(skip_gs, ctx);

        // Exclusive group prefix, combined with group result.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(op));
        air_wg_barrier(ctx, rewriter);
        let edge_grp = if reverse { n_groups - 1 } else { 0 };
        let c_edge = air_const_i32(ctx, edge_grp);
        let is_edge = air_icmp(ctx, rewriter, ICmpPredicateAttr::EQ, simd_grp, c_edge);
        let one = air_const_i32(ctx, 1);
        let adj_raw = {
            let o = if reverse {
                air::bin(ctx, BinKind::Add, i32_ty, simd_grp, one)
            } else {
                air::bin(ctx, BinKind::Sub, i32_ty, simd_grp, one)
            };
            emit(ctx, rewriter, o)
        };
        let adj_safe = air_select(ctx, rewriter, is_edge, zero, adj_raw, i32_ty);
        let g = partial_gep(ctx, rewriter, adj_safe);
        let ld = air::LoadOp::new(ctx, cty, g, al);
        let adj_loaded = emit_op(ctx, rewriter, ld);
        let group_prefix = air_select(ctx, rewriter, is_edge, identity, adj_loaded, cty);
        let result = emit_combine(ctx, rewriter, group_prefix, cur, intrinsic, cty);
        let Some(result) = self.fpcast_if(ctx, rewriter, result, cty, rty) else {
            return Ok(());
        };
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        self.converted += 1;
        Ok(())
    }
    /// Adjacent `[32, 32]` column reduces lowered as one shared scratch round trip.
    pub(crate) fn collect_reduce_batches(&mut self, ctx: &Context, func: Ptr<Operation>) {
        for_each_block(ctx, func, &mut |ops| {
            for (i, &op) in ops.iter().enumerate() {
                if opcode_of(ctx, op) != Some(Opcode::Reduce) {
                    continue;
                }
                let mut members: Vec<Ptr<Operation>> = Vec::new();
                let mut info: Vec<ReduceMember> = Vec::new();
                for &cand in &ops[i..] {
                    if opcode_of(ctx, cand) != Some(Opcode::Reduce) {
                        break;
                    }
                    let operands: Vec<pliron::value::Value> = cand.deref(ctx).operands().collect();
                    let Some(&src_v) = operands.first() else {
                        break;
                    };
                    let Some(WaxType::Tile(tt)) = wax_type_of(ctx, src_v.get_type(ctx)) else {
                        break;
                    };
                    let tt = &tt;
                    if tt.shape.len() != 2 || tt.shape[1] != 32 || tt.shape[0] != 32 {
                        break;
                    }
                    if wax_int_attr(ctx, cand, &keys::ATTR_DIM) != Some(0) {
                        break;
                    }
                    // Uniform source skips the redistribution and is cheap already.
                    if self
                        .reduce
                        .reduce_sources
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
                    let Some((cty, _suf, rty)) = fp_intr_info(ctx, res_ty) else {
                        break;
                    };
                    let Some(base) = combine_base(ctx, cand) else {
                        break;
                    };
                    if align_bytes(ctx, cty).is_none() {
                        break;
                    }
                    members.push(cand);
                    info.push(ReduceMember { base, rty });
                }
                if members.len() < 2 {
                    continue;
                }
                let record = self.reduce.batches.len();
                for (slot, &member) in members.iter().enumerate() {
                    self.reduce
                        .of_member
                        .insert(member, (BatchId(record), Slot(slot)));
                }
                self.reduce.batches.push(ReduceBatch {
                    head: op,
                    members,
                    info,
                });
            }
        });
    }
}

pub(crate) struct ColumnReduceSources {
    pub(crate) sources: HashMap<Ptr<Operation>, Value>,
}

impl pliron::pass::Analysis for ColumnReduceSources {
    fn name(&self) -> &str {
        "air-column-reduce-sources"
    }

    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        _analyses: &mut pliron::pass::AnalysisManager,
    ) -> pliron::result::Result<Self> {
        Ok(ColumnReduceSources {
            sources: record_column_reduce_sources(ctx, op),
        })
    }
}

pub(crate) struct TgScratchFloats {
    pub(crate) floats: u32,
}

impl pliron::pass::Analysis for TgScratchFloats {
    fn name(&self) -> &str {
        "air-tg-scratch-floats"
    }

    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        _analyses: &mut pliron::pass::AnalysisManager,
    ) -> pliron::result::Result<Self> {
        Ok(TgScratchFloats {
            floats: tg_scratch_floats_of(ctx, op),
        })
    }
}

fn combine_base(ctx: &Context, op: Ptr<Operation>) -> Option<&'static str> {
    use pliron::linked_list::ContainsLinkedList;
    let mut found = None;
    let regions: Vec<_> = op.deref(ctx).regions().collect();
    if let Some(&region) = regions.first() {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for b in blocks {
            let ops: Vec<_> = b.deref(ctx).iter(ctx).collect();
            for bop in ops {
                let hit = match opcode_of(ctx, bop) {
                    Some(Opcode::MaxF) => Some("@air.simd_max"),
                    Some(Opcode::MinF) => Some("@air.simd_min"),
                    Some(Opcode::MulF) => Some("@air.simd_product"),
                    Some(Opcode::AddF) => Some("@air.simd_sum"),
                    _ => None,
                };
                if let Some(h) = hit {
                    found = Some(h);
                    break;
                }
            }
        }
    }
    found
}

/// The combine step of a reduce or scan, for `base`.
pub(crate) fn emit_combine(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    a: Value,
    b: Value,
    intrinsic: Intrinsic,
    ty_h: TypeHandle,
) -> Value {
    let Intrinsic { base, suf } = intrinsic;
    use crate::air_dialect::attrs::BinKind;
    use pliron_llvm::attributes::FCmpPredicateAttr;
    let i1_ty = t::i1_ty(ctx);
    let o = match base {
        "@air.simd_product" => air::bin(ctx, BinKind::FMul, ty_h, a, b),
        "@air.simd_max" => air::call_fast(
            ctx,
            &format!("@air.fast_fmax.{suf}"),
            Some(ty_h),
            vec![a, b],
        ),
        "@air.simd_min" => {
            let c = air::fcmp(ctx, FCmpPredicateAttr::OLT, i1_ty, a, b);
            let cond = emit(ctx, rewriter, c);
            air::select(ctx, ty_h, cond, a, b)
        }
        _ => air::bin(ctx, BinKind::FAdd, ty_h, a, b),
    };
    emit(ctx, rewriter, o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air_dialect::type_interfaces::scalar_bits;
    use crate::air_dialect::types as t;

    #[test]
    fn unknown_combine_base_has_no_identity() {
        let ctx = &mut Context::new();
        let f32_ty = t::f32_ty(ctx);
        for base in [
            "@air.simd_max",
            "@air.simd_min",
            "@air.simd_product",
            "@air.simd_sum",
        ] {
            assert!(reduce_init_bits(ctx, base, f32_ty).is_some(), "{base}");
        }
        assert_eq!(reduce_init_bits(ctx, "@air.simd_xor", f32_ty), None);
    }

    #[test]
    fn malformed_tg_scratch_rows_falls_back() {
        let size = |r: i64| -> Option<u32> {
            Some(r)
                .filter(|r| *r > 0)
                .and_then(|r| u32::try_from(r * SCRATCH_PITCH).ok())
        };
        assert_eq!(size(128), Some(128 * 33));
        for bad in [-1, 0, i64::from(i32::MAX)] {
            assert_eq!(size(bad), None, "rows {bad} must not become a size");
        }
    }

    #[test]
    fn reduce_init_bits_fits_compute_type() {
        let ctx = &mut Context::new();
        let bases = [
            "@air.simd_sum",
            "@air.simd_product",
            "@air.simd_max",
            "@air.simd_min",
        ];
        for cty in [t::f16_ty(ctx), t::bf16_ty(ctx), t::f32_ty(ctx)] {
            let w = scalar_bits(ctx, cty).unwrap();
            for base in bases {
                let bits = reduce_init_bits(ctx, base, cty)
                    .unwrap_or_else(|| panic!("no init for {base} at {w} bits"));
                assert!(
                    bits < (1u64 << w),
                    "{base} init {bits:#x} does not fit in {w} bits"
                );
            }
        }
    }

    #[test]
    fn reduce_init_bits_produces_expected_values() {
        let ctx = &mut Context::new();
        let f16 = t::f16_ty(ctx);
        let f32_ty = t::f32_ty(ctx);
        assert_eq!(reduce_init_bits(ctx, "@air.simd_sum", f32_ty), Some(0));
        assert_eq!(
            reduce_init_bits(ctx, "@air.simd_product", f32_ty),
            Some(f32::to_bits(1.0) as u64)
        );
        assert_eq!(
            reduce_init_bits(ctx, "@air.simd_max", f32_ty),
            Some(f32::to_bits(f32::NEG_INFINITY) as u64)
        );
        assert_eq!(
            reduce_init_bits(ctx, "@air.simd_min", f32_ty),
            Some(f32::to_bits(f32::INFINITY) as u64)
        );
        assert_eq!(
            reduce_init_bits(ctx, "@air.simd_product", f16),
            Some(0x3C00)
        );
        assert_eq!(reduce_init_bits(ctx, "@air.simd_max", f16), Some(0xFC00));
        assert_eq!(reduce_init_bits(ctx, "@air.simd_min", f16), Some(0x7C00));
    }
}
