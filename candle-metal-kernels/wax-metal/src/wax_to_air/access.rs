//! Per-access lowering. Load/store, element addressing, access bounds.

use super::*;
use crate::air_dialect::attrs::{BinKind, CastOpAttr};
use crate::air_dialect::types as t;
use crate::air_dialect::types::AirPtrType;
use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
use pliron::linked_list::LinkedList as _;
use pliron::r#type::{Typed, TypedHandle};
use pliron::value::Value;

type Refused<T> = std::result::Result<T, &'static str>;

fn as_air_ptr(ctx: &Context, ptr: TypeHandle) -> Option<TypedHandle<AirPtrType>> {
    TypedHandle::from_handle(ptr, ctx).ok()
}

pub(crate) fn ptr_addrspace(ctx: &Context, ptr: TypeHandle) -> Option<u32> {
    as_air_ptr(ctx, ptr).map(|p| p.deref(ctx).addrspace())
}

pub(crate) fn pointee_and_addrspace(ctx: &Context, ptr: TypeHandle) -> Option<(TypeHandle, u32)> {
    as_air_ptr(ctx, ptr).map(|p| {
        let p = p.deref(ctx);
        (p.pointee(), p.addrspace())
    })
}

/// Per dimension tile coordinates of a lane.
fn lane_decompose(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    tile_shape: &[i64],
    rank: usize,
    tp: Value,
    mut f: impl FnMut(usize, Value, &mut Context, &mut DialectConversionRewriter) -> Refused<()>,
) -> Refused<()> {
    if rank != tile_shape.len() {
        return Err("provided rank does not match tile shape");
    }
    let i32_ty = t::i32_ty(ctx);
    let n_elems: i64 = tile_shape.iter().product();
    let zero = air_const_i32(ctx, 0);
    let x = air::extract_elem(ctx, i32_ty, tp, zero);
    let x = emit(ctx, rw, x);
    let mut running = if n_elems > 32 || !(n_elems as u64).is_power_of_two() {
        x
    } else {
        let m = air_const_i32(ctx, n_elems - 1);
        air_bin_i32(ctx, rw, x, m, BinKind::And)
    };
    let mut remaining = n_elems;
    for d in (0..rank).rev() {
        let extent = tile_shape[d];
        if extent <= 1 {
            continue; // c_d is always 0
        }
        // The modulo is redundant once nothing outer is left to strip.
        let coord = if remaining == extent {
            running
        } else {
            let e = air_const_i32(ctx, extent);
            air_bin_i32(ctx, rw, running, e, BinKind::URem)
        };
        f(d, coord, ctx, rw)?;
        remaining /= extent;
        if remaining > 1 {
            let e = air_const_i32(ctx, extent);
            running = air_bin_i32(ctx, rw, running, e, BinKind::UDiv);
        }
    }
    Ok(())
}

/// A per dim bound to check an access against.
#[derive(Clone, Copy, Debug)]
pub enum Bound {
    /// A runtime extent, already promoted to a parameter and loaded.
    Dyn(Value),
    /// A static extent that tiles raggedly.
    Fixed(i64),
}

/// Per access bounds.
pub(crate) fn collect_access_bounds(
    ctx: &Context,
    func: Ptr<Operation>,
    promoted: &[Value],
) -> HashMap<Ptr<Operation>, Vec<(usize, Bound)>> {
    let mut accesses: Vec<(Ptr<Operation>, usize)> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            match opcode_of(ctx, op) {
                Some(Opcode::LoadViewTko) => accesses.push((op, 0)),
                Some(Opcode::StoreViewTko) => accesses.push((op, 1)),
                _ => {}
            }
        }
    });

    let mut out = HashMap::new();
    for (op, view_at) in accesses {
        let Some(view) = op.deref(ctx).operands().nth(view_at) else {
            continue;
        };
        let Some(WaxType::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx)) else {
            continue;
        };
        if pv.padding_value.is_none() {
            continue;
        }
        let shape = &pv.tensor_view.shape;
        let mut bounds: Vec<(usize, Bound)> = Vec::new();
        for (d, &tile_d) in pv.tile_shape.iter().enumerate() {
            let Some(&ext) = shape.get(d) else { continue };
            if ext == wax_ir::DYNAMIC {
                // Only a promoted extent yields a bound. Unpromoted extent reads as constant 0,
                // masking away every access. No extent `fold_view` erased the originating view.
                match extent_value_for_view(ctx, view, shape, d) {
                    Some(v) if promoted.contains(&v) => bounds.push((d, Bound::Dyn(v))),
                    Some(_) => why("bound: extent found but not promoted"),
                    None => why("bound: extent value not traceable"),
                }
                continue;
            }
            let t = i64::from(tile_d);
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

/// Get dim `d`'s extent of view, if present.
pub(crate) fn extent_value_for_view(
    ctx: &Context,
    view: Value,
    shape: &[i64],
    d: usize,
) -> Option<Value> {
    if shape.get(d).copied() != Some(wax_ir::DYNAMIC) {
        return None;
    }
    let tv_op = tensor_view_def(ctx, view)?;
    let dyn_idx = shape[..=d]
        .iter()
        .filter(|&&e| e == wax_ir::DYNAMIC)
        .count()
        - 1;
    let (n_base, n_dyn_shape, _) = wax_operand_segments(ctx, tv_op);
    if dyn_idx >= n_dyn_shape {
        return None;
    }
    let v = tv_op.deref(ctx).operands().nth(n_base + dyn_idx)?;
    Some(trace_through_assume(ctx, v))
}

/// Bit pattern of a view's padding value at an element type.
pub(crate) fn padding_bits(
    ctx: &Context,
    elem: TypeHandle,
    pad: wax_ir::PaddingValue,
) -> Option<u64> {
    use crate::air_dialect::type_interfaces::{Special, special_bits};
    use wax_ir::PaddingValue as P;
    let v = match pad {
        P::Zero => Special::Zero,
        P::NegZero => Special::NegZero,
        P::Nan => Special::Nan,
        P::PosInf => Special::PosInf,
        P::NegInf => Special::NegInf,
    };
    special_bits(ctx, elem, v)
}

/// Access information required to rewrite load/store ops.
struct AccessParts {
    code: Opcode,
    view: Value,
    view_ty: WaxType,
    res_tys: Vec<TypeHandle>,
    index_ops: Vec<Value>,
    /// View element type
    velem: TypeHandle,
    /// Element alignment.
    align: u32,
    pad_value: Option<u64>,
}

type StoreGuard = (Ptr<BasicBlock>, Ptr<BasicBlock>, OpInsertionPoint);

impl WaxToAir {
    /// Replace result with operand it forwards.
    /// E.g. `assume`, view constructor, or noop `reshape`.
    pub(crate) fn fold_view(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
    ) -> Result<()> {
        let base = op.deref(ctx).operands().next();
        let n_results = op.deref(ctx).get_num_results();
        let Some(base) = base else {
            self.bail(code, "nothing to forward: no operand");
            return Ok(());
        };
        if n_results != 1 {
            self.bail(code, "cannot forward to other than one result");
            return Ok(());
        }
        rewriter.replace_operation_with_values(ctx, op, vec![base]);
        self.converted += 1;
        Ok(())
    }

    /// Read shared shape of an access.
    fn access_parts(
        &mut self,
        ctx: &mut Context,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
        code: Opcode,
        view_at: usize,
        idx_at: usize,
    ) -> Option<AccessParts> {
        let operands: Vec<Value> = op.deref(ctx).operands().collect();
        let res_tys = op_result_types(ctx, op);
        let Some(&view) = operands.get(view_at) else {
            self.bail(code, "no-view-operand");
            return None;
        };
        let view_h = operands_info
            .lookup_most_recent_type(view)
            .unwrap_or_else(|| view.get_type(ctx));
        let Some(view_ty) = wax_ir::dialect::types::from_pliron(ctx, view_h) else {
            self.bail(code, "view-type-unreadable");
            return None;
        };
        let WaxType::PartitionView(pv) = &view_ty else {
            self.bail(code, "view-not-partition");
            return None;
        };
        let Some(velem) = crate::types::mem_elem_ty(ctx, &pv.tensor_view.element_type) else {
            self.bail(code, "view-element-type");
            return None;
        };
        let pad_value = pv.padding_value.and_then(|p| padding_bits(ctx, velem, p));
        let rank = pv.tile_shape.len();
        let align = crate::types::scalar_byte_width(&pv.tensor_view.element_type);

        // Extract indices from operands `[idx_at..idx_at+rank]`.
        let index_ops: Vec<_> = operands.iter().skip(idx_at).take(rank).copied().collect();
        if index_ops.len() != rank {
            self.bail(code, "index-count");
            return None;
        }
        Some(AccessParts {
            code,
            view,
            view_ty,
            res_tys,
            index_ops,
            velem,
            align,
            pad_value,
        })
    }

    /// `wax.load_view` -> one of four strategies, in the order they claim the access.
    pub(crate) fn rewrite_load(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let Some(p) = self.access_parts(ctx, op, operands_info, Opcode::LoadViewTko, 0, 1) else {
            return Ok(());
        };

        // Fragment tile uses simdgroup-matrix intrinsics.
        let res0 = {
            let o = op.deref(ctx);
            (o.get_num_results() > 0).then(|| o.get_result(0))
        };
        if let Some((nr, nc)) = res0.and_then(|v| self.frag_dims(ctx, operands_info, v)) {
            let fa = FragAccess {
                code: Opcode::LoadViewTko,
                view: p.view,
                view_ty: &p.view_ty,
                index_ops: &p.index_ops,
            };
            // Masked access addresses absolutely.
            return if self.is_masked(op) {
                self.rewrite_fragment_load_masked(ctx, rewriter, op, fa, nr, nc)
            } else {
                self.rewrite_fragment_load(ctx, rewriter, op, fa, nr, nc)
            };
        }

        // Lane vector fusion. W contiguous per lane loads become one wide load. Planned by
        // `collect_lane_runs`.
        if self.lanes.of_member.contains_key(&op) {
            return self.emit_fused_load(ctx, rewriter, op);
        }

        // Cross-loop reuse. Consumer loop is served from the cache. No global address, bounds,
        // or padding. The producer/staging loop uses the scalar path below.
        let cache_member = self.staging.cache_members.get(&op).copied();
        if let Some((pi, false)) = cache_member {
            return self.rewrite_cached_load(ctx, rewriter, op, pi, &p.res_tys);
        }
        self.rewrite_scalar_load(ctx, rewriter, op, &p, cache_member)
    }

    /// `wax.store_view` -> one of three strategies, in the order they claim the access.
    pub(crate) fn rewrite_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let Some(p) = self.access_parts(ctx, op, operands_info, Opcode::StoreViewTko, 1, 2) else {
            return Ok(());
        };
        let value = op.deref(ctx).operands().next();
        let Some(value) = value else {
            self.bail(p.code, "store-value-missing");
            return Ok(());
        };

        // Fragment tile uses simdgroup-matrix intrinsics.
        if let Some((nr, nc)) = self.frag_dims_of_grid(ctx, operands_info, value) {
            let fa = FragAccess {
                code: Opcode::StoreViewTko,
                view: p.view,
                view_ty: &p.view_ty,
                index_ops: &p.index_ops,
            };
            return if self.is_masked(op) {
                self.rewrite_fragment_store_masked(ctx, rewriter, op, value, fa, nr, nc)
            } else {
                self.rewrite_fragment_store(ctx, rewriter, op, value, fa, nr, nc)
            };
        }

        // Lane vector fused store mirror.
        if self.lanes.of_member.contains_key(&op) {
            return self.emit_fused_store(ctx, rewriter, op);
        }

        self.rewrite_scalar_store(ctx, rewriter, op, &p, value)
    }

    fn is_masked(&self, op: Ptr<Operation>) -> bool {
        self.addr.bounds.get(&op).is_some_and(|b| !b.is_empty())
    }

    /// Single scalar per lane. Address arithmetic, `air.gep`, `air.load`.
    fn rewrite_scalar_load(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        p: &AccessParts,
        cache_member: Option<(usize, bool)>,
    ) -> Result<()> {
        let mask = match self.in_bounds_mask(ctx, rewriter, op, &p.view_ty, &p.index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(p.code, e),
        };
        let off = match self.elem_offset(ctx, rewriter, &p.view_ty, &p.index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(p.code, e),
        };
        // Out of bounds -> clamp -> pad
        let off = match mask {
            Some(pv) => {
                let i32_ty = t::i32_ty(ctx);
                let zv = air_const_i32(ctx, 0);
                let sel = air::select(ctx, i32_ty, pv, off, zv);
                emit(ctx, rewriter, sel)
            }
            None => off,
        };
        let Some(addr) = self.access_gep(ctx, rewriter, p, off) else {
            return Ok(());
        };

        let Some(&res_ty) = p.res_tys.first() else {
            self.bail(p.code, "load-result-missing");
            return Ok(());
        };
        let Some(res_ty) = air_ty_for(ctx, res_ty) else {
            self.bail(p.code, "load-result-type");
            return Ok(());
        };
        let ld = air::LoadOp::new(ctx, res_ty, addr, p.align);
        let mut val = emit_op(ctx, rewriter, ld);
        if let Some(pv) = mask {
            let Some(pad) = p.pad_value else {
                self.bail(p.code, "padding value has no constant at this element type");
                return Ok(());
            };
            let Some(pc) = air::const_bits(ctx, res_ty, pad) else {
                self.bail(p.code, "padding value has no constant at the result type");
                return Ok(());
            };
            let pcv = emit_op(ctx, rewriter, pc);
            let sel = air::select(ctx, res_ty, pv, val, pcv);
            val = emit(ctx, rewriter, sel);
        }
        // Producing/staging loop writes value to cache (padding select included).
        if let Some((pi, true)) = cache_member {
            self.emit_cache_store(ctx, rewriter, pi, val)?;
        }
        // Remaining results are ordering tokens.
        // TODO: Figure out how to use the tokens properly.
        self.finish_replacement(ctx, rewriter, op, &p.res_tys, vec![val]);
        Ok(())
    }

    /// One scalar element per lane. Address arithmetic, `air.gep`, `air.store`.
    fn rewrite_scalar_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        p: &AccessParts,
        value: Value,
    ) -> Result<()> {
        // `in_bounds_mask` emits lane decomposition so it must be before the store guard.
        let in_bounds = match self.in_bounds_mask(ctx, rewriter, op, &p.view_ty, &p.index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(p.code, e),
        };
        let guard = match in_bounds {
            Some(pv) => match self.open_store_guard(ctx, rewriter, op, p.code, pv)? {
                Some(g) => Some(g),
                None => return Ok(()),
            },
            None => None,
        };
        let off = match self.elem_offset(ctx, rewriter, &p.view_ty, &p.index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(p.code, e),
        };
        let Some(addr) = self.access_gep(ctx, rewriter, p, off) else {
            return Ok(());
        };
        let st = air::StoreOp::new(ctx, value, addr, p.align);
        rewriter.insert_operation(ctx, st.get_operation());
        if let Some((do_b, skip, saved)) = guard {
            let br = air::BrOp::new(ctx, skip, vec![]);
            br.get_operation().insert_at_back(do_b, ctx);
            rewriter.set_insertion_point(saved);
        }
        self.finish_replacement(ctx, rewriter, op, &p.res_tys, vec![]);
        Ok(())
    }

    /// Split block around masked store and set insertion point into the taken arm.
    /// `Ok(None)` means the store was refused.
    fn open_store_guard(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        code: Opcode,
        pred: Value,
    ) -> Result<Option<StoreGuard>> {
        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(code, "store is not inside a block");
            return Ok(None);
        };
        let Some(region) = head.deref(ctx).get_container() else {
            self.bail(code, "block is not inside a region");
            return Ok(None);
        };
        let skip = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        // `_pad{n}_store` is created before `_pad{n}_skip` and both are appended, so the region
        // order is do-block then skip-block.
        let do_b =
            rewriter.create_block(ctx, BlockInsertionPoint::AtRegionEnd(region), None, vec![]);
        rewriter.move_block(ctx, skip, BlockInsertionPoint::AtRegionEnd(region));
        let cbr = air::CondBrOp::new(ctx, pred, do_b, vec![], skip, vec![]);
        cbr.get_operation().insert_at_back(head, ctx);
        let saved = rewriter.get_insertion_point();
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(do_b));
        Ok(Some((do_b, skip, saved)))
    }

    /// Turns the offset in elements into a typed address.
    fn access_gep(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        p: &AccessParts,
        off: Value,
    ) -> Option<Value> {
        let base_ptr_ty = p.view.get_type(ctx);
        let Some(addrspace) = ptr_addrspace(ctx, base_ptr_ty) else {
            self.bail(p.code, "element-type");
            return None;
        };
        let ptr_ty = t::ptr_ty(ctx, p.velem, addrspace);
        let view = if ptr_ty == base_ptr_ty {
            p.view
        } else {
            let bc = air::cast(ctx, CastOpAttr::Bitcast, ptr_ty, p.view);
            emit(ctx, rewriter, bc)
        };
        let gep = air::GepOp::new(ctx, ptr_ty, view, off, p.velem);
        Some(emit_op(ctx, rewriter, gep))
    }

    /// Calculate lane's per dimension tile coordinates. Noop when tile has one element per lane.
    fn lane_coords(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        tile_shape: &[i64],
        rank: usize,
        f: impl FnMut(usize, Value, &mut Context, &mut DialectConversionRewriter) -> Refused<()>,
    ) -> Refused<()> {
        if tile_shape.iter().product::<i64>() <= 1 {
            return Ok(());
        }
        let Some(tp) = self.control.tp else {
            return Err("no thread position parameter");
        };
        lane_decompose(ctx, rewriter, tile_shape, rank, tp, f)
    }

    /// Get element offset for a static view.
    pub(crate) fn elem_offset(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        view_ty: &WaxType,
        index_ops: &[Value],
    ) -> Refused<Value> {
        let (pv, tile_shape) = partition_tile(view_ty)?;
        let strides = &pv.tensor_view.strides;
        // View must be static.
        if strides.contains(&wax_ir::DYNAMIC) {
            return Err("view has a dynamic stride");
        }

        // Add lane position within tile to offset.
        let rank = index_ops.len();
        let mut acc = None;
        self.lane_coords(ctx, rewriter, &tile_shape, rank, |d, coord, ctx, rw| {
            let Some(&sd) = strides.get(d) else {
                return Err("lane stride missing for a tile dim");
            };
            let s = air_const_i32(ctx, sd);
            let term = air_bin_i32(ctx, rw, coord, s, BinKind::Mul);
            acc = Some(match acc {
                None => term,
                Some(a) => air_bin_i32(ctx, rw, a, term, BinKind::Add),
            });
            Ok(())
        })?;

        // If no lane position default to const 0.
        let mut acc = acc.unwrap_or_else(|| air_const_i32(ctx, 0));

        // Add tile position (which tile) to offset.
        for (d, &idx) in index_ops.iter().enumerate() {
            let Some(mult) = strides
                .get(d)
                .and_then(|s| s.checked_mul(*tile_shape.get(d).unwrap_or(&1)))
            else {
                return Err("index stride missing for a tile dim");
            };
            let m = air_const_i32(ctx, mult);
            let term = air_bin_i32(ctx, rewriter, idx, m, BinKind::Mul);
            acc = air_bin_i32(ctx, rewriter, acc, term, BinKind::Add);
        }
        Ok(acc)
    }

    /// Per dimension absolute coordinates of this lane's element, `index[d] * tile[d] + c_d`.
    pub(crate) fn elem_coords(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        view_ty: &WaxType,
        index_ops: &[Value],
    ) -> Refused<Vec<Value>> {
        let (_, tile_shape) = partition_tile(view_ty)?;
        let rank = index_ops.len();

        // Get lane positions within tile.
        let mut coords: Vec<Option<Value>> = vec![None; rank];
        self.lane_coords(ctx, rewriter, &tile_shape, rank, |d, coord, _ctx, _rw| {
            coords[d] = Some(coord);
            Ok(())
        })?;

        // Calculate absolute coordinates of tile's origin.
        let mut out = Vec::with_capacity(rank);
        for (d, &idx) in index_ops.iter().enumerate() {
            let tile_d = *tile_shape.get(d).unwrap_or(&1);
            let base = if tile_d == 1 {
                idx
            } else {
                let m = air_const_i32(ctx, tile_d);
                air_bin_i32(ctx, rewriter, idx, m, BinKind::Mul)
            };
            out.push(match coords[d] {
                None => base,
                Some(c) => air_bin_i32(ctx, rewriter, base, c, BinKind::Add),
            });
        }
        Ok(out)
    }

    /// Create mask that accepts lanes whose element lies inside the tensor.
    /// `Ok(None)` needs no mask.
    pub(crate) fn in_bounds_mask(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        view_ty: &WaxType,
        index_ops: &[Value],
    ) -> Refused<Option<Value>> {
        use pliron_llvm::attributes::ICmpPredicateAttr;
        let Some(bounds) = self.addr.bounds.get(&op).cloned() else {
            return Ok(None);
        };
        if bounds.is_empty() {
            return Ok(None);
        }
        let coords = self.elem_coords(ctx, rewriter, view_ty, index_ops)?;
        let i1_ty = t::i1_ty(ctx);
        let mut mask: Option<Value> = None;
        for (d, b) in bounds {
            let ext = match b {
                Bound::Dyn(v) => v,
                Bound::Fixed(n) => {
                    let c = air::const_int(ctx, wax_ir::dialect::ops::BITS_32, n as u64);
                    emit_op(ctx, rewriter, c)
                }
            };
            let Some(&coord) = coords.get(d) else {
                return Err("no lane coordinate for a bounded dim");
            };
            let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, coord, ext);
            let ok = emit(ctx, rewriter, cmp);
            mask = Some(match mask {
                None => ok,
                Some(p) => {
                    let a = air::bin(ctx, BinKind::And, i1_ty, p, ok);
                    emit(ctx, rewriter, a)
                }
            });
        }
        Ok(mask)
    }
}

/// The `PartitionView` a view access addresses, tile shape widened to i64.
fn partition_tile(view_ty: &WaxType) -> Refused<(&wax_ir::types::PartitionViewType, Vec<i64>)> {
    let WaxType::PartitionView(pv) = view_ty else {
        return Err("view is not a partition view");
    };
    Ok((pv, pv.tile_shape.iter().copied().map(i64::from).collect()))
}
