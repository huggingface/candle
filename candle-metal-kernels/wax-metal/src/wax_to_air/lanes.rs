//! The lane-vector fusion.

use super::*;
use crate::air_dialect::types as t;
use pliron::value::Value;
use wax_ir::dialect::ops::BITS_32;

fn outer_index_count(view_ty: &WaxType) -> Option<usize> {
    let WaxType::PartitionView(pv) = view_ty else {
        return None;
    };
    pv.tile_shape.len().checked_sub(1).filter(|n| *n > 0)
}

/// Whether an air type is a vector.
fn is_vector(ctx: &Context, h: pliron::r#type::TypeHandle) -> bool {
    use pliron::r#type::TypedHandle;
    TypedHandle::<pliron_llvm::types::VectorType>::from_handle(h, ctx).is_ok()
}

/// Check that indices are the same value or that their constant i64 values are equal.
fn index_eq(ctx: &Context, a: Value, b: Value) -> bool {
    a == b || matches!((wax_const_i64(ctx, a), wax_const_i64(ctx, b)), (Some(x), Some(y)) if x == y)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RunId(pub(crate) usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Slot(pub(crate) usize);

/// Lane run fused into one wide load or store.
/// The emission point (head for load, tail for store), members of the op, and original view type.
#[derive(Clone)]
pub(crate) struct LaneRun {
    head: Ptr<Operation>,
    tail: Ptr<Operation>,
    members: Vec<Ptr<Operation>>,
    view_ty: wax_ir::types::Type,
    values: Vec<Option<Value>>,
    emitted: bool,
}

/// Information the lane run needs from its view.
fn lane_run_shape(
    ctx: &Context,
    op: Ptr<Operation>,
    view_at: usize,
) -> Option<(Value, wax_ir::types::PartitionViewType, usize, usize)> {
    use pliron::r#type::Typed;
    let view = op.deref(ctx).operands().nth(view_at)?;
    let WaxType::PartitionView(pv) = wax_type_of(ctx, view.get_type(ctx))? else {
        return None;
    };
    let rank = pv.tile_shape.len();
    if rank < 2 || pv.tile_shape[rank - 1] != 1 || pv.tensor_view.strides[rank - 1] != 1 {
        return None;
    }
    // Lane dim is innermost dim with extent > 1.
    let lane_dim = (0..rank - 1).rev().find(|&d| pv.tile_shape[d] > 1)?;
    let w = pv.tensor_view.strides[lane_dim];
    (2..=8).contains(&w).then_some((view, pv, rank, w as usize))
}

/// Record a run. Members get slots, the run gets its emission point.
fn record_run(
    lane_runs: &mut std::collections::HashMap<Ptr<Operation>, (RunId, Slot)>,
    runs: &mut Vec<LaneRun>,
    members: Vec<Ptr<Operation>>,
    head: Ptr<Operation>,
    tail: Ptr<Operation>,
    pv: wax_ir::types::PartitionViewType,
) {
    let record = runs.len();
    for (slot, &m) in members.iter().enumerate() {
        lane_runs.insert(m, (RunId(record), Slot(slot)));
    }
    let values = vec![None; members.len()];
    runs.push(LaneRun {
        head,
        tail,
        members,
        view_ty: pv.into(),
        values,
        emitted: false,
    });
}

impl WaxToAir {
    /// Fuse all eligible lane runs into wide loads/stores.
    ///
    /// Masked loads/stores cannot fuse.
    // TODO: Investigate if this is true for all masking (col stride 1 -> emask?)
    pub(crate) fn collect_lane_runs(&mut self, ctx: &Context, func: Ptr<Operation>) {
        let bounds = self.addr.bounds.clone();
        let mut runs: Vec<LaneRun> = Vec::new();
        // Loads
        for_each_block(ctx, func, &mut |ops| {
            for (i, &op) in ops.iter().enumerate() {
                if opcode_of(ctx, op) != Some(Opcode::LoadViewTko) {
                    continue;
                }
                let Some((view, pv, rank, w)) = lane_run_shape(ctx, op, 0) else {
                    continue;
                };
                let operands: Vec<Value> = op.deref(ctx).operands().collect();
                // The head must be element 0 of its run.
                if wax_const_i64(ctx, operands[rank]) != Some(0) {
                    continue;
                }
                if bounds.contains_key(&op) {
                    continue;
                }
                let mut group: Vec<Option<Ptr<Operation>>> = vec![None; w];
                group[0] = Some(op);
                for &cand in &ops[i + 1..] {
                    if opcode_of(ctx, cand) != Some(Opcode::LoadViewTko) {
                        continue;
                    }
                    let c: Vec<Value> = cand.deref(ctx).operands().collect();
                    if c[0] != view {
                        continue;
                    }
                    if !(1..rank).all(|d| index_eq(ctx, c[d], operands[d])) {
                        continue;
                    }
                    let Some(j) = wax_const_i64(ctx, c[rank]) else {
                        continue;
                    };
                    if (1..w as i64).contains(&j) && group[j as usize].is_none() {
                        group[j as usize] = Some(cand);
                    }
                    if group.iter().all(Option::is_some) {
                        break;
                    }
                }
                let Some(members) = group.into_iter().collect::<Option<Vec<Ptr<Operation>>>>()
                else {
                    continue;
                };
                record_run(&mut self.lanes.of_member, &mut runs, members, op, op, pv);
            }
        });
        // Stores
        for_each_block(ctx, func, &mut |ops| {
            for &op in ops {
                if opcode_of(ctx, op) != Some(Opcode::StoreViewTko) {
                    continue;
                }
                let Some((view, pv, rank, w)) = lane_run_shape(ctx, op, 1) else {
                    continue;
                };
                let operands: Vec<Value> = op.deref(ctx).operands().collect();
                // A bounds predicate would need a guarded store, which is currently unsupported.
                if bounds.get(&op).is_some_and(|b| !b.is_empty()) {
                    continue;
                }
                let mut group: Vec<Option<Ptr<Operation>>> = vec![None; w];
                let mut last_pos = (usize::MAX, op);
                for (ci, &cand) in ops.iter().enumerate() {
                    if opcode_of(ctx, cand) != Some(Opcode::StoreViewTko) {
                        continue;
                    }
                    let c: Vec<Value> = cand.deref(ctx).operands().collect();
                    if c.len() != operands.len() || c.get(1).copied() != Some(view) {
                        continue;
                    }
                    if !(2..1 + rank).all(|d| index_eq(ctx, c[d], operands[d])) {
                        continue;
                    }
                    let Some(j) = wax_const_i64(ctx, c[1 + rank]) else {
                        continue;
                    };
                    if (0..w as i64).contains(&j) && group[j as usize].is_none() {
                        group[j as usize] = Some(cand);
                        if ci > last_pos.0 {
                            last_pos = (ci, cand);
                        }
                    }
                }
                let Some(members) = group.into_iter().collect::<Option<Vec<Ptr<Operation>>>>()
                else {
                    continue;
                };
                if !members.contains(&op) {
                    continue;
                }
                if members
                    .iter()
                    .any(|&m| bounds.get(&m).is_some_and(|b| !b.is_empty()))
                {
                    continue;
                }
                let last = last_pos.1;
                record_run(
                    &mut self.lanes.of_member,
                    &mut runs,
                    members,
                    last,
                    last,
                    pv,
                );
            }
        });
        self.lanes.runs = runs;
    }
    /// Emit fused wide load for this lane run member. Subsequent members of the same plan are
    /// covered by the first one and so they are simply erased.
    pub(crate) fn emit_fused_load(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        use pliron::irbuild::inserter::OpInsertionPoint;
        use pliron::r#type::Typed;
        let res_tys = op_result_types(ctx, op);
        let Some(&value_ty) = res_tys.first() else {
            return self.refuse(Opcode::LoadViewTko, "no result types");
        };
        let Some(value_ty) = air_ty_for(ctx, value_ty) else {
            return self.refuse(Opcode::LoadViewTko, "invalid result type");
        };
        // Already covered by previous member.
        let Some(&(record, slot)) = self.lanes.of_member.get(&op) else {
            return self.refuse(Opcode::LoadViewTko, "no lane run recorded");
        };
        if let Some(ev) = self.lanes.runs[record.0].values[slot.0] {
            self.finish_replacement(ctx, rewriter, op, &res_tys, vec![ev]);
            return Ok(());
        }
        let rec = &self.lanes.runs[record.0];
        let head = rec.head;
        let members = rec.members.clone();
        let view_ty = rec.view_ty.clone();
        if head != op {
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(head));
        }
        // Get outer index count
        let Some(n_outer) = outer_index_count(&view_ty) else {
            return self.refuse(Opcode::LoadViewTko, "no outer index");
        };
        let mut index_ops: Vec<Value> = op.deref(ctx).operands().skip(1).take(n_outer).collect();
        let zero = air::const_int(ctx, BITS_32, 0);
        let zero_v = emit_op(ctx, rewriter, zero);
        index_ops.push(zero_v);

        let off = match self.elem_offset(ctx, rewriter, &view_ty, &index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(Opcode::LoadViewTko, e),
        };
        let Some(view) = op.deref(ctx).operands().next() else {
            return self.refuse(Opcode::LoadViewTko, "no view operand");
        };
        let ptr_ty = view.get_type(ctx);
        let Some((elem, addrspace)) = pointee_and_addrspace(ctx, ptr_ty) else {
            return self.refuse(Opcode::LoadViewTko, "could not extract info from ptr");
        };
        let gep = air::GepOp::new(ctx, ptr_ty, view, off, elem);
        let base = emit_op(ctx, rewriter, gep);

        // An element that is already a vector would nest
        // TODO: Either `refuse` or ensure unreachable state.
        assert!(
            !is_vector(ctx, elem),
            "lane vector element type must be scalar"
        );
        let w = members.len();
        let vec_ty = t::vec_ty(ctx, w as u32, elem);
        let vptr_ty = t::ptr_ty(ctx, vec_ty, addrspace);
        let bc = air::cast(ctx, CastOpAttr::Bitcast, vptr_ty, base);
        let vptr = emit(ctx, rewriter, bc);

        let Some(elem_ty) = elem_scalar_of(&view_ty) else {
            return self.refuse(Opcode::LoadViewTko, "view is not a partition view");
        };
        let align = w as u32 * crate::types::scalar_byte_width(&elem_ty);
        let ld = air::LoadOp::new(ctx, vec_ty, vptr, align);
        let wide = emit_op(ctx, rewriter, ld);

        // One extract per member
        for j in 0..members.len() {
            let idx = if j == 0 {
                zero_v
            } else {
                let c = air::const_int(ctx, BITS_32, j as u64);
                emit_op(ctx, rewriter, c)
            };
            let e = air::extract_elem(ctx, value_ty, wide, idx);
            let ev = emit(ctx, rewriter, e);
            self.lanes.runs[record.0].values[j] = Some(ev);
        }
        let Some(mine) = self.lanes.runs[record.0].values[slot.0] else {
            return self.refuse(Opcode::LoadViewTko, "lane vector member missing");
        };
        self.finish_replacement(ctx, rewriter, op, &res_tys, vec![mine]);
        Ok(())
    }
    /// Emit fused wide store for this lane run member. Subsequent members of the same plan are
    /// covered by the first one and so they are simply erased.
    pub(crate) fn emit_fused_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let Some(&(record, slot)) = self.lanes.of_member.get(&op) else {
            return self.refuse(Opcode::StoreViewTko, "no planned run");
        };
        let Some(my_value) = op.deref(ctx).operands().next() else {
            return self.refuse(Opcode::StoreViewTko, "store member has no value operand");
        };
        self.lanes.runs[record.0].values[slot.0] = Some(my_value);
        let rec = self.lanes.runs[record.0].clone();

        let not_done = !rec.emitted;
        let all_ready = rec.members.iter().enumerate().all(|(j, &m)| {
            if rec.values[j].is_some() {
                return true;
            }
            let Some(v) = m.deref(ctx).operands().next() else {
                return false;
            };
            match v.defining_entity() {
                pliron::value::DefiningEntity::Op(def) => opcode_of(ctx, def).is_none(),
                _ => true,
            }
        });
        if not_done && (all_ready || rec.tail == op) {
            if !all_ready {
                return self.refuse(Opcode::StoreViewTko, "did not finish");
            } else {
                self.emit_lane_vector_store_run(ctx, rewriter, op, &rec)?;
                self.lanes.runs[record.0].emitted = true;
            }
        }
        self.erase_to_undef(ctx, rewriter, op)
    }
    /// The wide store itself. Pack with `insertelement`, address at first element, store once.
    pub(crate) fn emit_lane_vector_store_run(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        rec: &LaneRun,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        use pliron::irbuild::inserter::OpInsertionPoint;
        use pliron::r#type::Typed;
        let view_ty = &rec.view_ty;
        let Some(n_outer) = outer_index_count(view_ty) else {
            return self.refuse(Opcode::StoreViewTko, "no outer index");
        };
        let Some(view) = op.deref(ctx).operands().nth(1) else {
            return self.refuse(Opcode::StoreViewTko, "no view operand");
        };
        let ptr_ty = view.get_type(ctx);
        let Some((elem, addrspace)) = pointee_and_addrspace(ctx, ptr_ty) else {
            return self.refuse(Opcode::StoreViewTko, "could not extract info from ptr");
        };
        // An element that is already a vector would nest
        // TODO: Either `refuse` or ensure unreachable state.
        assert!(
            !is_vector(ctx, elem),
            "lane vector element type must be scalar"
        );

        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(rec.tail));

        let zero = air::const_int(ctx, BITS_32, 0);
        let zero_v = emit_op(ctx, rewriter, zero);
        let mut index_ops: Vec<Value> = op.deref(ctx).operands().skip(2).take(n_outer).collect();
        index_ops.push(zero_v);

        let w = rec.members.len();
        let vec_ty = t::vec_ty(ctx, w as u32, elem);
        // Packing
        let undef = air::UndefOp::new(ctx, vec_ty);
        let mut packed = emit_op(ctx, rewriter, undef);
        for (j, &member) in rec.members.iter().enumerate() {
            let val = rec.values[j].or_else(|| member.deref(ctx).operands().next());
            let Some(val) = val else {
                return self.refuse(Opcode::StoreViewTko, "store has no value operand");
            };
            let idx = if j == 0 {
                zero_v
            } else {
                let c = air::const_int(ctx, BITS_32, j as u64);
                emit_op(ctx, rewriter, c)
            };
            let i = air::insert_elem(ctx, vec_ty, packed, val, idx);
            packed = emit(ctx, rewriter, i);
        }
        let off = match self.elem_offset(ctx, rewriter, view_ty, &index_ops) {
            Ok(v) => v,
            Err(e) => return self.refuse(Opcode::StoreViewTko, e),
        };
        let gep = air::GepOp::new(ctx, ptr_ty, view, off, elem);
        let base = emit_op(ctx, rewriter, gep);
        let vptr_ty = t::ptr_ty(ctx, vec_ty, addrspace);
        let bc = air::cast(ctx, CastOpAttr::Bitcast, vptr_ty, base);
        let vptr = emit(ctx, rewriter, bc);
        let Some(elem_ty) = elem_scalar_of(view_ty) else {
            return self.refuse(Opcode::StoreViewTko, "not a partition view");
        };
        let align = w as u32 * crate::types::scalar_byte_width(&elem_ty);
        let st = air::StoreOp::new(ctx, packed, vptr, align);
        rewriter.insert_operation(ctx, st.get_operation());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wax_ir::types::{PartitionViewType, ScalarType, TensorViewType};

    fn view(tile: Vec<i32>) -> WaxType {
        WaxType::PartitionView(PartitionViewType {
            tile_shape: tile,
            tensor_view: TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![8, 8],
                strides: vec![8, 1],
            },
            dim_map: vec![],
            padding_value: None,
        })
    }

    #[test]
    fn view_with_no_outer_index_yields_none() {
        assert_eq!(outer_index_count(&view(vec![4, 1])), Some(1));
        assert_eq!(outer_index_count(&view(vec![2, 4, 1])), Some(2));
        assert_eq!(outer_index_count(&view(vec![1])), None,);
        assert_eq!(outer_index_count(&view(vec![])), None);
        assert_eq!(outer_index_count(&WaxType::Scalar(ScalarType::F32)), None);
    }
}
