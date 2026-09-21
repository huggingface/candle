//! The lane-vector fusion: the runs planned before conversion and the wide
//! load and store emitters.

use super::*;

/// Two index operands denote the same coordinate: the same value, or two constants with
/// the same bits. `walk.rs`'s `same_index` makes exactly this concession - a per-use
/// constant is not a new coordinate - and the fusion condition rides on it. The constant
/// reader is a parameter: the run is planned before conversion, where the indices are
/// `wax.constant` ops read with `wax_const_i64`.
pub(crate) fn index_eq(
    ctx: &Context,
    a: pliron::value::Value,
    b: pliron::value::Value,
    const_of: fn(&Context, pliron::value::Value) -> Option<i64>,
) -> bool {
    a == b || matches!((const_of(ctx, a), const_of(ctx, b)), (Some(x), Some(y)) if x == y)
}

/// A per-lane run `walk.rs` fuses into one wide load, planned before conversion: the head
/// op (the run's element 0, where the wide load is emitted), the members in slot order, and
/// the type the head's view carried before `fold_view` erased it.
#[derive(Clone)]
pub(crate) struct LaneRun {
    head: Ptr<Operation>,
    /// The member the RUN is emitted at: the head for a load, the LAST member in block
    /// order for a store (the first point where every stored value is lowered).
    last: Ptr<Operation>,
    members: Vec<Ptr<Operation>>,
    view_ty: wax_ir::types::Type,
}

impl WaxToAir {
    /// The per-lane runs `walk.rs` fuses into one wide load, planned BEFORE conversion: member
    /// op -> (its slot in the run, the run's record index).
    ///
    /// `walk.rs` fuses at walk time, when the loads are still all alive and in block order. The
    /// conversion driver's order is its own - a later member commonly rewrites before the first,
    /// its siblings already erased - so the run must be decided on the PRE-conversion function,
    /// the way `bounds` and `fragments` are. The conditions mirror `walk.rs::fuse_lane_vector`
    /// one for one: the innermost dim is the unit run, the lane's own dim steps by exactly W
    /// (2..=8), the head is the run's element 0, and the next W-1 loads in block order take the
    /// same view with the same outer indices and the next innermost element. A bounds-masked
    /// load cannot fuse: the CFG fuses before it would compute a mask, and clamping a wide load
    /// has no spelling this route has.
    pub(crate) fn collect_lane_runs(&mut self, ctx: &Context, func: Ptr<Operation>) {
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        let bounds = self.addr.bounds.clone();
        let mut runs: Vec<LaneRun> = Vec::new();
        for_each_block(ctx, func, &mut |ops| {
            for (i, &op) in ops.iter().enumerate() {
                if try_opcode_of(ctx, op) != Some(Opcode::LoadViewTko) {
                    continue;
                }
                let operands: Vec<pliron::value::Value> = op.deref(ctx).operands().collect();
                let view = operands[0];
                let Some(WT::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx)) else {
                    continue;
                };
                let rank = pv.tile_shape.len();
                if rank < 2 || pv.tile_shape[rank - 1] != 1 || pv.tensor_view.strides[rank - 1] != 1
                {
                    continue;
                }
                // The lane's own dim is the innermost dim with extent > 1; its stride is the run width.
                let Some(lane_dim) = (0..rank - 1).rev().find(|&d| pv.tile_shape[d] > 1) else {
                    continue;
                };
                let w = pv.tensor_view.strides[lane_dim];
                if !(2..=8).contains(&w) {
                    continue;
                }
                let w = w as usize;
                // The head must be element 0 of its run.
                if wax_const_i64(ctx, operands[rank]) != Some(0) {
                    continue;
                }
                // A bounds predicate would need clamping the wide load, which has no spelling.
                if bounds.contains_key(&op) {
                    continue;
                }
                let mut group: Vec<Option<Ptr<Operation>>> = vec![None; w];
                group[0] = Some(op);
                for &cand in &ops[i + 1..] {
                    if try_opcode_of(ctx, cand) != Some(Opcode::LoadViewTko) {
                        continue;
                    }
                    let c: Vec<pliron::value::Value> = cand.deref(ctx).operands().collect();
                    if c[0] != view {
                        continue;
                    }
                    if !(1..rank).all(|d| index_eq(ctx, c[d], operands[d], wax_const_i64)) {
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
                let Some(members) = group.into_iter().collect::<Option<Vec<Ptr<Operation>>>>() else {
                    continue;
                };
                let record = runs.len();
                for (slot, &member) in members.iter().enumerate() {
                    self.lanes.lane_runs.insert(member, (slot, record));
                }
                runs.push(LaneRun {
                    head: op,
                    last: op,
                    members,
                    view_ty: pv.clone().into(),
                });
            }
        });
        // Store runs, `walk.rs::lane_run_group` on a store: the group is scanned over the
        // WHOLE block (store j's value may be computed after store 0) and any member may
        // trigger it; the emission lands at the LAST member in block order.
        for_each_block(ctx, func, &mut |ops| {
            for &op in ops {
                if try_opcode_of(ctx, op) != Some(Opcode::StoreViewTko) {
                    continue;
                }
                let operands: Vec<pliron::value::Value> = op.deref(ctx).operands().collect();
                let Some(&view) = operands.get(1) else {
                    continue;
                };
                let Some(WT::PartitionView(pv)) = wax_type_of(ctx, view.get_type(ctx)) else {
                    continue;
                };
                let rank = pv.tile_shape.len();
                if rank < 2 || pv.tile_shape[rank - 1] != 1 || pv.tensor_view.strides[rank - 1] != 1
                {
                    continue;
                }
                let Some(lane_dim) = (0..rank - 1).rev().find(|&d| pv.tile_shape[d] > 1) else {
                    continue;
                };
                let w = pv.tensor_view.strides[lane_dim];
                if !(2..=8).contains(&w) {
                    continue;
                }
                let w = w as usize;
                // A bounds predicate would need a guarded store; the fused wide store has
                // no such spelling (and on &mut tensor params the walk's store bounds are
                // inert, so this never fires on a kernel it fuses).
                if bounds.get(&op).is_some_and(|b| !b.is_empty()) {
                    continue;
                }
                let mut group: Vec<Option<Ptr<Operation>>> = vec![None; w];
                let mut last_pos = (usize::MAX, op);
                for (ci, &cand) in ops.iter().enumerate() {
                    if try_opcode_of(ctx, cand) != Some(Opcode::StoreViewTko) {
                        continue;
                    }
                    let c: Vec<pliron::value::Value> = cand.deref(ctx).operands().collect();
                    if c.len() != operands.len() || c.get(1).copied() != Some(view) {
                        continue;
                    }
                    if !(2..1 + rank).all(|d| index_eq(ctx, c[d], operands[d], wax_const_i64)) {
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
                let Some(members) = group.into_iter().collect::<Option<Vec<Ptr<Operation>>>>() else {
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
                let record = runs.len();
                for (slot, &member) in members.iter().enumerate() {
                    self.lanes.lane_runs.insert(member, (slot, record));
                }
                let last = last_pos.1;
                runs.push(LaneRun {
                    head: last,
                    last,
                    members,
                    view_ty: pv.clone().into(),
                });
            }
        });
        self.lanes.lane_records = runs;
    }
    /// One member of a planned lane-vector run.
    ///
    /// The FIRST member the driver reaches emits the whole run - the base offset, the gep, the
    /// bitcast, the one wide load, an extract per member - at the HEAD's position, where
    /// `walk.rs` puts it, whatever order the driver visits the members in. Every later member
    /// only answers its uses from the recorded extracts and erases itself; the token results
    /// are `undef`s, as on the scalar path.
    pub(crate) fn emit_lane_vector_member(
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
            self.bail(Opcode::LoadViewTko, "load-result-missing");
            return Ok(());
        };
        let Some(value_ty) = self.air_ty(ctx, value_ty) else {
            self.bail(Opcode::LoadViewTko, "load-result-type");
            return Ok(());
        };
        // Already emitted by the member the driver reached first: answer and erase.
        if let Some((_, ev)) = self.lanes.lane_vectors.get(&op) {
            self.finish_replacement(ctx, rewriter, op, &res_tys, vec![*ev]);
            return Ok(());
        }
        let (_, rec) = {
            let (slot, record) = *self.lanes.lane_runs.get(&op).expect("planned member has a record");
            (slot, &self.lanes.lane_records[record])
        };
        let head = rec.head;
        let members = rec.members.clone();
        let view_ty = rec.view_ty.clone();
        // The driver's insertion point is before THIS member; the run lives at the HEAD's.
        if head != op {
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(head));
        }
        // The base is element 0 of the run: this member's own (already-converted) outer
        // indices, the innermost one forced to 0 - the head's, as `walk.rs` uses it. Equal
        // constants are the same instruction, so which member emits makes no difference to
        // the bytes.
        let rank = match &view_ty {
            wax_ir::types::Type::PartitionView(pv) => pv.tile_shape.len(),
            _ => 0,
        };
        let mut index_ops: Vec<pliron::value::Value> = op
            .deref(ctx)
            .operands()
            .skip(1)
            .take(rank - 1)
            .collect();
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let zero = air::ConstBitsOp::new(ctx, i32_ty, 0);
        rewriter.insert_operation(ctx, zero.get_operation());
        index_ops.push(zero.get_operation().deref(ctx).get_result(0));

        let Some(off) = self.elem_offset(ctx, rewriter, &view_ty, &index_ops) else {
            self.bail(Opcode::LoadViewTko, "address-unsupported");
            return Ok(());
        };
        let view = op.deref(ctx).operands().next().unwrap();
        let ptr_ty = view.get_type(ctx);
        let Some(elem) = element_of(ctx, ptr_ty) else {
            self.bail(Opcode::LoadViewTko, "element-type");
            return Ok(());
        };
        let gep = air::GepOp::new(ctx, ptr_ty, view, off);
        air::set_elem_ty(gep.get_operation(), ctx, elem);
        rewriter.insert_operation(ctx, gep.get_operation());
        let base = gep.get_operation().deref(ctx).get_result(0);

        let elem_ll = crate::air_dialect::types::from_pliron(ctx, elem)
            .filter(|l| !matches!(l, LlType::Vec(_, _)))
            .expect("element type is a scalar");
        let w = members.len();
        let vec_ll = LlType::vec(w as u32, elem_ll.clone());
        let vec_ty = crate::air_dialect::types::to_pliron(ctx, &vec_ll);
        let vptr_ty = {
            let LlType::Ptr { addrspace, .. } =
                crate::air_dialect::types::from_pliron(ctx, ptr_ty).expect("a pointer")
            else {
                unreachable!("element_of returned a non-pointer")
            };
            crate::air_dialect::types::to_pliron(
                ctx,
                &LlType::Ptr {
                    pointee: Box::new(vec_ll.clone()),
                    addrspace,
                },
            )
        };
        let bc = air::CastOp::new(ctx, vptr_ty, base);
        bc.set_op(ctx, CastOpAttr::Bitcast);
        rewriter.insert_operation(ctx, bc.get_operation());
        let vptr = bc.get_operation().deref(ctx).get_result(0);

        let align = w as u32 * crate::types::scalar_byte_width(&elem_scalar_of(&view_ty));
        let ld = air::LoadOp::new(ctx, vec_ty, vptr);
        air::set_align(ld.get_operation(), ctx, align);
        rewriter.insert_operation(ctx, ld.get_operation());
        let wide = ld.get_operation().deref(ctx).get_result(0);

        // One extract per member, slot 0 reusing the zero above.
        let zero_v = *index_ops.last().unwrap();
        for (j, member) in members.iter().enumerate() {
            let idx = if j == 0 {
                zero_v
            } else {
                let c = air::ConstBitsOp::new(ctx, i32_ty, j as u64);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
            let e = air::ExtractElemOp::new(ctx, value_ty, wide, idx);
            rewriter.insert_operation(ctx, e.get_operation());
            let ev = e.get_operation().deref(ctx).get_result(0);
            self.lanes.lane_vectors.insert(*member, (j, ev));
        }
        let (_, mine) = self.lanes.lane_vectors.get(&op).copied().expect("just recorded");
        self.finish_replacement(ctx, rewriter, op, &res_tys, vec![mine]);
        Ok(())
    }
    /// The store mirror of [`Self::emit_lane_vector_member`]: one wide store per run,
    /// emitted at the LAST member - the first point at which every stored value is
    /// guaranteed to have been lowered. Each member only ever binds its token; the
    /// emission itself happens exactly once, by the first processed member that can
    /// see every value (the last member can, always).
    pub(crate) fn emit_lane_vector_store_member(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let (slot, record) = *self.lanes.lane_runs
            .get(&op)
            .expect("planned store member has a record");
        let rec = self.lanes.lane_records[record].clone();
        // My own value is readable while I am alive: capture it for the members that
        // process after me - by then my op is erased.
        let my_value = op.deref(ctx).operands().next().unwrap();
        self.lanes.lane_store_values
            .entry(record)
            .or_default()
            .insert(slot, my_value);
        let not_done = !self.lanes.lane_store_done.contains(&record);
        // A member is ready when it has been processed (value captured) or, still in
        // the block, its value operand no longer defines a wax op.
        let all_ready = rec.members.iter().enumerate().all(|(j, &m)| {
            if self.lanes.lane_store_values.get(&record).map(|t| t.contains_key(&j)).unwrap_or(false)
            {
                return true;
            }
            let v = m.deref(ctx).operands().next().unwrap();
            match v.defining_entity() {
                pliron::value::DefiningEntity::Op(def) => try_opcode_of(ctx, def).is_none(),
                _ => true,
            }
        });
        if not_done && (all_ready || rec.last == op) {
            if !all_ready {
                self.bail(
                    Opcode::StoreViewTko,
                    "lane-vector store run reached its last member with an unconverted value",
                );
            } else {
                self.emit_lane_vector_store_run(ctx, rewriter, record, op, &rec)?;
                self.lanes.lane_store_done.insert(record);
            }
        }
        // A store result is an ordering token; AGX has nothing to order against.
        self.erase_to_undef(ctx, rewriter, op)
    }
    /// The wide store itself: pack the members' values with `insertelement`, address the
    /// run from its element-0 member, store once - `walk.rs::fuse_lane_vector_store`.
    pub(crate) fn emit_lane_vector_store_run(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        record: usize,
        op: Ptr<Operation>,
        rec: &LaneRun,
    ) -> Result<()> {
        use crate::air_dialect::attrs::CastOpAttr;
        use pliron::irbuild::inserter::OpInsertionPoint;
        use pliron::r#type::Typed;
        let view_ty = &rec.view_ty;
        let rank = match view_ty {
            wax_ir::types::Type::PartitionView(pv) => pv.tile_shape.len(),
            _ => 0,
        };
        // The run lives at the LAST member, whatever member the driver reached first.
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(rec.last));
        // The view and outer indices come from THIS member - the one op guaranteed alive.
        // Every member shares them by the grouping condition (same value or equal
        // constant), and equal constants are the same instruction, so the bytes are the
        // same whichever member the walk's first member would have been.
        let view = op.deref(ctx).operands().nth(1).unwrap();
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let zero = air::ConstBitsOp::new(ctx, i32_ty, 0);
        rewriter.insert_operation(ctx, zero.get_operation());
        let zero_v = zero.get_operation().deref(ctx).get_result(0);
        let mut index_ops: Vec<pliron::value::Value> = op
            .deref(ctx)
            .operands()
            .skip(2)
            .take(rank - 1)
            .collect();
        index_ops.push(zero_v);
        let ptr_ty = view.get_type(ctx);
        let Some(elem) = element_of(ctx, ptr_ty) else {
            self.bail(Opcode::StoreViewTko, "element-type");
            return Ok(());
        };
        let elem_ll = crate::air_dialect::types::from_pliron(ctx, elem)
            .filter(|l| !matches!(l, LlType::Vec(_, _)))
            .expect("element type is a scalar");
        let w = rec.members.len();
        let vec_ll = LlType::vec(w as u32, elem_ll.clone());
        let vec_ty = crate::air_dialect::types::to_pliron(ctx, &vec_ll);
        // Pack FIRST, exactly as `walk.rs::fuse_lane_vector_store` does: the inserts
        // precede the address arithmetic in the stream.
        let undef = air::UndefOp::new(ctx, vec_ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        let mut packed = undef.get_operation().deref(ctx).get_result(0);
        for (j, &member) in rec.members.iter().enumerate() {
            // Captured if the member already processed (its op is erased), live otherwise.
            let val = self.lanes.lane_store_values
                .get(&record)
                .and_then(|t| t.get(&j))
                .copied()
                .unwrap_or_else(|| member.deref(ctx).operands().next().unwrap());
            let idx = if j == 0 {
                zero_v
            } else {
                let c = air::ConstBitsOp::new(ctx, i32_ty, j as u64);
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
            let i = air::InsertElemOp::new(ctx, vec_ty, packed, val, idx);
            rewriter.insert_operation(ctx, i.get_operation());
            packed = i.get_operation().deref(ctx).get_result(0);
        }
        let Some(off) = self.elem_offset(ctx, rewriter, view_ty, &index_ops) else {
            self.bail(Opcode::StoreViewTko, "address-unsupported");
            return Ok(());
        };
        let gep = air::GepOp::new(ctx, ptr_ty, view, off);
        air::set_elem_ty(gep.get_operation(), ctx, elem);
        rewriter.insert_operation(ctx, gep.get_operation());
        let base = gep.get_operation().deref(ctx).get_result(0);
        let vptr_ty = {
            let LlType::Ptr { addrspace, .. } =
                crate::air_dialect::types::from_pliron(ctx, ptr_ty).expect("a pointer")
            else {
                unreachable!("element_of returned a non-pointer")
            };
            crate::air_dialect::types::to_pliron(
                ctx,
                &LlType::Ptr {
                    pointee: Box::new(vec_ll.clone()),
                    addrspace,
                },
            )
        };
        let bc = air::CastOp::new(ctx, vptr_ty, base);
        bc.set_op(ctx, CastOpAttr::Bitcast);
        rewriter.insert_operation(ctx, bc.get_operation());
        let vptr = bc.get_operation().deref(ctx).get_result(0);
        let align = w as u32 * crate::types::scalar_byte_width(&elem_scalar_of(view_ty));
        let st = air::StoreOp::new(ctx, packed, vptr);
        air::set_align(st.get_operation(), ctx, align);
        rewriter.insert_operation(ctx, st.get_operation());
        Ok(())
    }
}
