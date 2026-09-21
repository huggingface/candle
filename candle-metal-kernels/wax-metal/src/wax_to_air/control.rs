//! Control flow. `wax.if`, `wax.for`, `wax.return`, and deferred back edge and body cleanup.

use super::*;
use crate::air_dialect::types as t;
use pliron::basic_block::BasicBlock;

/// Loop body with fragment placeholders to be stripped once conversion is done.
pub(crate) struct BodyCleanup {
    body: Ptr<BasicBlock>,
    hdr: Ptr<BasicBlock>,
    /// How many fragment placeholders the body still declares.
    n: usize,
}

/// Check if a value is in the constant pool.
pub(crate) fn is_const_pool_value(ctx: &Context, op: Ptr<Operation>) -> bool {
    use crate::air_dialect::ops as air;
    Operation::is_op::<air::ConstantOp>(op, ctx)
        || Operation::is_op::<air::UndefOp>(op, ctx)
        || Operation::is_op::<air::ZeroOp>(op, ctx)
        || Operation::is_op::<air::GlobalRefOp>(op, ctx)
}

// TODO: Move out of `control.rs` into `metadata.rs` or `utils.rs`.
/// Attach `!llvm.loop` unroll hints once bodies are lowered. The factor is a function of
/// AIR instruction count. Body must be branchless.
pub(crate) fn apply_loop_unroll(ctx: &mut Context, loops: &[(Ptr<BasicBlock>, Ptr<BasicBlock>)]) {
    use pliron::linked_list::ContainsLinkedList;
    for &(body, back) in loops {
        let ops: Vec<_> = body.deref(ctx).iter(ctx).collect();
        let Some(&last) = ops.last() else { continue };
        let straight = Operation::is_op::<air::BrOp>(last, ctx)
            && last.deref(ctx).successors().next() == Some(back);
        // Count relevant AIR instructions.
        let insts = ops
            .iter()
            .take(ops.len().saturating_sub(1))
            .filter(|&&o| !is_const_pool_value(ctx, o))
            .count();

        // The tuning override wins over the formula. Both only apply to a straight body.
        let count = match crate::types::loop_unroll_override() {
            Some(Some(n)) if straight && insts > 0 => n,
            Some(_) => continue,
            None => match crate::types::loop_unroll_factor(straight, insts) {
                Some(n) => n,
                None => continue,
            },
        };
        let Some(latch_term) = back.deref(ctx).iter(ctx).last() else {
            continue;
        };
        air::set_unroll(latch_term, ctx, count);
    }
}

/// Whether `block` already ends in a terminator.
fn block_is_terminated(ctx: &Context, block: Ptr<BasicBlock>) -> bool {
    block.deref(ctx).get_terminator(ctx).is_some()
}

impl WaxToAir {
    pub(crate) fn rewrite_return(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let ret = air::ReturnOp::new(ctx, None);
        rewriter.insert_operation(ctx, ret.get_operation());
        rewriter.replace_operation(ctx, op, ret.get_operation());
        self.converted += 1;
        Ok(())
    }

    /// Rewrite `wax.if` to `cond_br` over two arms that rejoin at a merge block.
    /// The arms' results become merge block arguments.
    ///
    /// Conceptually similar to MLIR `scf.if` -> `cf.cond_br`.
    ///
    /// Arm bodies are moved, not cloned. Ops are already queued, so a clone would re-convert
    /// the originals into a region about to be discarded.
    pub(crate) fn rewrite_if(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::{ContainsLinkedList, LinkedList};
        use pliron::r#type::Typed;

        let (n_results, n_regions, cond) = {
            let o = op.deref(ctx);
            (o.get_num_results(), o.num_regions(), o.operands().next())
        };
        let Some(cond) = cond else {
            self.bail(Opcode::If, "no condition operand");
            return Ok(());
        };
        // An `if` that produces values needs both arms to have something to merge.
        if n_results > 0 && n_regions < 2 {
            self.bail(Opcode::If, "value-producing if without an else arm");
            return Ok(());
        }
        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::If, "if is not inside a block");
            return Ok(());
        };
        // Lowering outer first. A nested `if` would be unlowered and should be a single block.
        let mut arms = Vec::with_capacity(n_regions);
        for i in 0..n_regions {
            let r = op.deref(ctx).get_region(i);
            let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
            if blocks.len() != 1 {
                self.bail(Opcode::If, "arm is not a single block");
                return Ok(());
            }
            arms.push(blocks[0]);
        }

        // Convert result types first.
        let mut merge_arg_tys = Vec::with_capacity(n_results);
        for i in 0..n_results {
            let t = op.deref(ctx).get_result(i).get_type(ctx);
            let Some(t) = air_ty_for(ctx, t) else {
                self.bail(Opcode::If, "result type has no air equivalent");
                return Ok(());
            };
            merge_arg_tys.push(t);
        }

        // Split block through the rewriter.
        let Some(region) = head.deref(ctx).get_container() else {
            self.bail(Opcode::If, "block is not inside a region");
            return Ok(());
        };
        let merge = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        for (i, t) in merge_arg_tys.iter().enumerate() {
            BasicBlock::insert_argument(merge, ctx, i, *t);
        }
        let merge_args: Vec<_> = (0..n_results)
            .map(|i| merge.deref(ctx).get_argument(i))
            .collect();

        // `split_block` moved the `if` itself into the merge block. Put it back so the head
        // ends with the `cond_br` that replaces it.
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockEnd(head));

        // Inline each arm's region between the head and the merge block.
        for i in 0..n_regions {
            let r = op.deref(ctx).get_region(i);
            rewriter.inline_region(ctx, r, BlockInsertionPoint::AtRegionEnd(region));
        }
        // Then merge.
        rewriter.move_block(ctx, merge, BlockInsertionPoint::AtRegionEnd(region));

        // An arm ends in `wax.yield`, which becomes the merge block branch with yielded values as
        // block arguments.
        for &arm in &arms {
            let last = arm.deref(ctx).iter(ctx).last();
            let (target_args, yield_op) = match last {
                Some(l) if opcode_of(ctx, l) == Some(Opcode::Yield) => {
                    let vals: Vec<_> = l.deref(ctx).operands().collect();
                    (vals, Some(l))
                }
                _ => (Vec::new(), None),
            };
            if target_args.len() != n_results {
                self.bail(Opcode::If, "arm yields the wrong number of values");
                return Ok(());
            }
            if let Some(y) = yield_op {
                rewriter.erase_operation(ctx, y);
            }
            // Only add `br` if arm is still open.
            if !block_is_terminated(ctx, arm) {
                let br = air::BrOp::new(ctx, merge, target_args);
                br.get_operation().insert_at_back(arm, ctx);
            }
        }

        // An `if` with no else arm falls straight through to the merge block.
        let else_target = if arms.len() > 1 { arms[1] } else { merge };
        let cbr = air::CondBrOp::new(ctx, cond, arms[0], vec![], else_target, vec![]);
        cbr.get_operation().insert_at_back(head, ctx);

        rewriter.replace_operation_with_values(ctx, op, merge_args);
        self.converted += 1;
        Ok(())
    }

    /// `wax.for` -> the four-block counted loop.
    ///
    /// ```text
    ///   head:  br hdr(lo, inits...)
    ///   hdr(iv, iter_values...): cond = iv slt hi; cond_br cond, body, exit
    ///   body:  ...; br back
    ///   back:  iv_next = iv + step; br hdr(iv_next, updates...)
    ///   exit:  the `for`'s results are hdr's iteration values
    /// ```
    ///
    /// The induction variable (IV) and each iteration value is a block argument of `hdr`.
    /// That is how a phi is spelled in a block-argument IR. `exit` needs no arguments of its
    /// own, being dominated by `hdr`. A fragment iteration value occupies `nr * nc`
    /// fragment registers there.
    pub(crate) fn rewrite_for(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::BinKind;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::{ContainsLinkedList, LinkedList};
        use pliron::r#type::Typed;
        use pliron_llvm::attributes::ICmpPredicateAttr;

        let (operands, n_iter_values) = {
            let o = op.deref(ctx);
            (o.operands().collect::<Vec<_>>(), o.get_num_results())
        };
        // operands are (lo, hi, step, inits...)
        if operands.len() < 3 + n_iter_values {
            self.bail(Opcode::For, "too few operands for lo/hi/step/inits");
            return Ok(());
        }
        let (lo, hi, step) = (operands[0], operands[1], operands[2]);
        let inits: Vec<_> = operands[3..3 + n_iter_values].to_vec();

        let body = {
            let r = op.deref(ctx).get_region(0);
            let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
            // The body must be a single block. Nested control flow is fine at this stage.
            if blocks.len() != 1 {
                self.bail(Opcode::For, "body is not a single block");
                return Ok(());
            }
            blocks[0]
        };
        let body_args: Vec<_> = body.deref(ctx).arguments().collect();
        if body_args.len() != 1 + n_iter_values {
            self.bail(
                Opcode::For,
                "body arguments do not match the iteration values",
            );
            return Ok(());
        }

        // The body's terminator supplies the next iteration's values. `yield` and
        // `continue` are equivalent here.
        let term = body.deref(ctx).iter(ctx).last();
        let (term, updates) = match term {
            Some(t)
                if matches!(
                    opcode_of(ctx, t),
                    Some(Opcode::Yield) | Some(Opcode::Continue)
                ) =>
            {
                let vals: Vec<_> = t.deref(ctx).operands().collect();
                (t, vals)
            }
            _ => {
                self.bail(Opcode::For, "body does not end in yield or continue");
                return Ok(());
            }
        };
        if updates.len() != n_iter_values {
            self.bail(
                Opcode::For,
                "body yields the wrong number of iteration values",
            );
            return Ok(());
        }

        // Emit allocate instruction for staging side of reuse cache plan. Must exist before
        // the split that moves this op.
        if let Some(&(pi, _)) = self.staging.cache_fors.get(&op) {
            self.ensure_cache_alloca(ctx, rewriter, pi)?;
        }

        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::For, "for is not inside a block");
            return Ok(());
        };

        // The `nr * nc` fragment registers only support f32, everything else is scalar.
        // TODO: Worth verifying this claim again.
        let frag64 = t::vec_ty(ctx, 64, t::f32_ty(ctx));
        let mut iter_value_frags: Vec<Option<(usize, usize)>> = Vec::with_capacity(n_iter_values);
        for i in 0..n_iter_values {
            let r = op.deref(ctx).get_result(i);
            if let Some((nr, nc)) = self.frag_dims(ctx, operands_info, r) {
                let ty = operands_info
                    .lookup_most_recent_type(r)
                    .unwrap_or_else(|| r.get_type(ctx));
                let Some(wax_ir::types::Type::Tile(t)) = wax_type_of(ctx, ty) else {
                    self.bail(Opcode::For, "fragment iteration value type unreadable");
                    return Ok(());
                };
                let wax_ir::types::TileElementType::Scalar(s) = &t.element_type else {
                    self.bail(Opcode::For, "fragment iteration value is not a scalar tile");
                    return Ok(());
                };
                if *s != wax_ir::types::ScalarType::F32 {
                    self.bail(Opcode::For, "mixed-precision fragment iteration value");
                    return Ok(());
                }
                iter_value_frags.push(Some((nr, nc)));
            } else {
                iter_value_frags.push(None);
            }
        }

        // Header argument types. The induction variable, and then one register per iteration value.
        // `nr * nc` for a fragment iteration value.
        let i32_ty = t::i32_ty(ctx);
        let mut hdr_arg_tys = vec![i32_ty];
        for (i, &frags) in iter_value_frags.iter().enumerate() {
            match frags {
                Some((nr, nc)) => hdr_arg_tys.extend(std::iter::repeat_n(frag64, nr * nc)),
                None => {
                    let t = op.deref(ctx).get_result(i).get_type(ctx);
                    let Some(t) = air_ty_for(ctx, t) else {
                        self.bail(Opcode::For, "iteration value type has no air equivalent");
                        return Ok(());
                    };
                    hdr_arg_tys.push(t);
                }
            }
        }
        // Where the register of each iteration value begins in the header.
        let mut iter_value_bases = Vec::with_capacity(n_iter_values);
        {
            let mut b = 1;
            for &frags in &iter_value_frags {
                iter_value_bases.push(b);
                b += match frags {
                    Some((nr, nc)) => nr * nc,
                    None => 1,
                };
            }
        }

        // Everything after the `for` becomes the exit block. Pass through the rewriter, so the
        // driver is told these blocks exist.
        let Some(region) = head.deref(ctx).get_container() else {
            self.bail(Opcode::For, "block is not inside a region");
            return Ok(());
        };
        // We move `exit` to the correct position later.
        let exit = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockEnd(head));

        // Blocks go to the end of the region in creation order.
        // `hdr`, `body`, `back` and `exit` are created up front and appended.
        let hdr = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionEnd(region),
            None,
            hdr_arg_tys,
        );
        rewriter.inline_region(
            ctx,
            op.deref(ctx).get_region(0),
            BlockInsertionPoint::AtRegionEnd(region),
        );
        let back =
            rewriter.create_block(ctx, BlockInsertionPoint::AtRegionEnd(region), None, vec![]);
        // Move `exit` to region end.
        rewriter.move_block(ctx, exit, BlockInsertionPoint::AtRegionEnd(region));

        let n_regs: usize = iter_value_frags
            .iter()
            .map(|f| match f {
                Some((nr, nc)) => nr * nc,
                None => 1,
            })
            .sum();
        let hdr_args: Vec<_> = (0..1 + n_regs)
            .map(|i| hdr.deref(ctx).get_argument(i))
            .collect();

        // Replace body arguments with those of the header.
        body_args[0].replace_all_uses_with(ctx, &hdr_args[0]);
        let mut remove_idx: Vec<usize> = vec![0];
        for i in 0..n_iter_values {
            let old = body_args[1 + i];
            match iter_value_frags[i] {
                None => {
                    old.replace_all_uses_with(ctx, &hdr_args[iter_value_bases[i]]);
                    remove_idx.push(1 + i);
                }
                Some((nr, nc)) => {
                    // Fragment iteration value needs special treatment.
                    let group: Vec<_> =
                        hdr_args[iter_value_bases[i]..iter_value_bases[i] + nr * nc].to_vec();
                    old.set_type(ctx, frag64);
                    self.frags.grids.insert(old, group);
                }
            }
        }
        for &idx in remove_idx.iter().rev() {
            BasicBlock::remove_argument(body, ctx, idx);
        }

        // head: br hdr(lo, inits...)
        let mut entry_args = vec![lo];
        for (i, &init) in inits.iter().enumerate() {
            match iter_value_frags[i] {
                None => entry_args.push(init),
                Some((nr, nc)) => {
                    let Some(grid) = self.grid_of(init).cloned() else {
                        self.bail(Opcode::For, "fragment iteration value init has no grid");
                        return Ok(());
                    };
                    if grid.len() != nr * nc {
                        self.bail(
                            Opcode::For,
                            "fragment iteration value init grid size disagrees",
                        );
                        return Ok(());
                    }
                    entry_args.extend(grid);
                }
            }
        }
        let br_hdr = air::BrOp::new(ctx, hdr, entry_args);
        br_hdr.get_operation().insert_at_back(head, ctx);

        // hdr: `cond = iv slt hi`, then `cond_br cond, body, exit`.
        let i1_ty = t::i1_ty(ctx);
        let cmp = air::icmp(ctx, ICmpPredicateAttr::SLT, i1_ty, hdr_args[0], hi);
        cmp.insert_at_back(hdr, ctx);
        let cond = cmp.deref(ctx).get_result(0);
        let body_feed: Vec<pliron::value::Value> = (0..n_iter_values)
            .filter(|&i| iter_value_frags[i].is_some())
            .map(|i| hdr_args[iter_value_bases[i]])
            .collect();
        let cbr = air::CondBrOp::new(ctx, cond, body, body_feed.clone(), exit, vec![]);
        cbr.get_operation().insert_at_back(hdr, ctx);

        // Defer fragment placeholder cleanup if still present.
        if !body_feed.is_empty() {
            self.control.body_cleanups.push(BodyCleanup {
                body,
                hdr,
                n: body_feed.len(),
            });
        }

        // body: make yield a fall-through to the back edge.
        rewriter.erase_operation(ctx, term);
        if !block_is_terminated(ctx, body) {
            let br_back = air::BrOp::new(ctx, back, vec![]);
            br_back.get_operation().insert_at_back(body, ctx);
        }

        // back: `iv_next = iv + step`, then `br hdr(iv_next, updates...)`
        let add = air::bin(ctx, BinKind::Add, i32_ty, hdr_args[0], step);
        add.insert_at_back(back, ctx);
        let iv_next = add.deref(ctx).get_result(0);
        let mut back_args = vec![iv_next];
        let mut any_deferred = false;
        for (i, &upd) in updates.iter().enumerate() {
            match iter_value_frags[i] {
                // If defining op converts rewrite single-fragment iteration value's to `<64 x float>`.
                None | Some((1, 1)) => back_args.push(upd),
                // Multi-fragment. Defer whole branch.
                Some((_, _)) => {
                    any_deferred = true;
                }
            }
        }
        if any_deferred {
            self.control.pending_back_edges.push(PendingBackEdge {
                back,
                hdr,
                iv_next,
                updates: updates.to_vec(),
                iter_value_frags: iter_value_frags.clone(),
            });
        } else {
            let back_edge = air::BrOp::new(ctx, hdr, back_args);
            back_edge.get_operation().insert_at_back(back, ctx);
        }

        self.control.loops.push((body, back));

        // Cache members index their slot with this loop's IV. Now that the loop exists that
        // IV is header argument 0.
        if let Some(&(pi, is_staging)) = self.staging.cache_fors.get(&op) {
            let p = &mut self.staging.cache_plans[pi];
            if is_staging {
                p.staging_iv = Some(hdr_args[0]);
            } else {
                p.consuming_iv = Some(hdr_args[0]);
            }
        }

        // Results are read from the exit block.
        // A fragment result stands for its register group, bound to the first register as the
        // representative. Same approach as AIR mma and fragment load.
        let mut reps = Vec::with_capacity(n_iter_values);
        for i in 0..n_iter_values {
            match iter_value_frags[i] {
                None => reps.push(hdr_args[iter_value_bases[i]]),
                Some((nr, nc)) => {
                    let group: Vec<_> =
                        hdr_args[iter_value_bases[i]..iter_value_bases[i] + nr * nc].to_vec();
                    let Some(&rep) = group.first() else {
                        self.bail(Opcode::For, "empty fragment iteration value group");
                        return Ok(());
                    };
                    let result_i = op.deref(ctx).get_result(i);
                    self.frags.grids.insert(result_i, group.clone());
                    self.frags.grids.insert(rep, group);
                    reps.push(rep);
                }
            }
        }
        rewriter.replace_operation_with_values(ctx, op, reps);
        self.converted += 1;
        Ok(())
    }

    /// Build the deferred branches from `rewrite_for`.
    ///
    /// One per loop that has a multi-fragment iteration value.
    pub(crate) fn finish_pending_back_edges(&mut self, ctx: &mut Context) {
        for l in std::mem::take(&mut self.control.pending_back_edges) {
            let mut args = vec![l.iv_next];
            let mut ok = true;
            for (i, &upd) in l.updates.iter().enumerate() {
                match l.iter_value_frags[i] {
                    None | Some((1, 1)) => args.push(upd),
                    Some((nr, nc)) => match self.grid_of(upd) {
                        Some(grid) if grid.len() == nr * nc => args.extend(grid.iter().copied()),
                        _ => {
                            ok = false;
                            break;
                        }
                    },
                }
            }
            if ok {
                let br = air::BrOp::new(ctx, l.hdr, args);
                br.get_operation().insert_at_back(l.back, ctx);
            }
        }
    }

    /// Clean up fragment placeholders from loop body.
    pub(crate) fn finish_body_cleanups(&mut self, ctx: &mut Context) {
        use pliron::builtin::op_interfaces::BranchOpInterface;
        use pliron::op::Op;
        for c in std::mem::take(&mut self.control.body_cleanups) {
            // Placeholders were rewired to use header representatives. If any still has a user it
            // means a consumer bailed. Skip and let the kernel fail the residual check.
            let bb = c.body.deref(ctx);
            let n_args = bb.get_num_arguments();
            let alive = (0..n_args).any(|i| bb.get_argument(i).num_uses(ctx) != 0);
            drop(bb);
            if n_args != c.n || alive {
                continue;
            }
            // The hdr's terminator is the cond_br that feeds the body. Drop the feed.
            let Some(term) = c.hdr.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            let succs: Vec<_> = term.deref(ctx).successors().collect();
            let Some(succ_idx) = succs.iter().position(|&s| s == c.body) else {
                continue;
            };
            let Some(cb) = air::CondBrOp::wrap_operation(term).downcast::<air::CondBrOp>() else {
                continue;
            };
            for i in (0..c.n).rev() {
                cb.remove_successor_operand(ctx, succ_idx, i);
            }
            // Remove arguments
            for i in (0..c.n).rev() {
                BasicBlock::remove_argument(c.body, ctx, i);
            }
        }
    }
}
