//! The control-flow arms: `wax.if`, `wax.for`, `wax.break`, and the deferred
//! latch and body cleanups.

use super::*;

/// A loop body that still carries fragment placeholders, to strip once conversion is done.
///
/// A placeholder exists only so the body's ops can resolve the carry's grid while they are
/// still wax ops. The CFG path's body block has NO arguments - the carries live in the
/// header - so the placeholders (and the branch feeds that keep them defined) must not
/// survive to the emission.
pub(crate) struct BodyCleanup {
    body: Ptr<pliron::basic_block::BasicBlock>,
    hdr: Ptr<pliron::basic_block::BasicBlock>,
    /// How many fragment placeholders the body still declares.
    n: usize,
}

/// Attach `!llvm.loop` unroll hints, once the loop bodies are lowered.
///
/// The factor is `160 / body_insts`, so it needs the AIR instruction count, which does not
/// exist while `wax.for` is being rewritten. Deferring keeps one policy
/// (`lower::loop_unroll_factor`) rather than a second estimate that would drift.
///
/// A body is only an unroll target if it is still STRAIGHT-LINE: control flow lowered inside
/// it (an `if`) splits the block, and the branch to the latch then lives in some later block.
/// Checking that the body still ends in the branch to its own latch is exactly the
/// `body_straight` test `walk.rs` makes.
pub(crate) fn apply_loop_unroll(
    ctx: &mut Context,
    loops: &[(
        Ptr<pliron::basic_block::BasicBlock>,
        Ptr<pliron::basic_block::BasicBlock>,
    )],
) {
    use pliron::linked_list::ContainsLinkedList;
    for &(body, back) in loops {
        let ops: Vec<_> = body.deref(ctx).iter(ctx).collect();
        let Some(&last) = ops.last() else { continue };
        let straight = Operation::get_op_dyn(last, ctx)
            .op_ref()
            .get_opid()
            .to_string()
            == "air.br"
            && last.deref(ctx).successors().next() == Some(back);
        // Count what becomes an INSTRUCTION. `block_inst_count` excludes the terminator,
        // which in pliron is an op in the block; it also never saw constants, because the CFG
        // builder carries those inline on the consuming operand (`mul i32 0, 32` is one
        // instruction). The air dialect makes them SSA ops that the emitter folds back into
        // immediates, so counting them inflates the body and pushes it past the 32-instruction
        // cutoff - which silently drops the unroll hint rather than changing the factor.
        let insts = ops
            .iter()
            .take(ops.len().saturating_sub(1))
            .filter(|&&o| {
                !matches!(
                    Operation::get_op_dyn(o, ctx)
                        .op_ref()
                        .get_opid()
                        .to_string()
                        .as_str(),
                    "air.const_bits" | "air.const_vec" | "air.undef" | "air.zero"
                )
            })
            .count();
        let Some(count) = crate::types::loop_unroll_factor(straight, insts) else {
            continue;
        };
        let Some(&latch_term) = back.deref(ctx).iter(ctx).collect::<Vec<_>>().last() else {
            continue;
        };
        air::set_unroll(latch_term, ctx, count);
    }
}

/// Whether `block` already ends in a terminator.
///
/// The CFG builder spells this `set_terminator_if_open`, and it matters wherever two arms of
/// the lowering both want to close a block: a `wax.break` inside an `if` arm terminates it,
/// and the arm's own fall-through to the merge block must then not be added. Two terminators
/// are the same LENGTH as one plus one unreachable branch, so this shows up as a content
/// difference at identical size.
pub(crate) fn block_is_terminated(ctx: &Context, block: Ptr<pliron::basic_block::BasicBlock>) -> bool {
    use pliron::linked_list::ContainsLinkedList;
    let Some(last) = block.deref(ctx).iter(ctx).last() else {
        return false;
    };
    matches!(
        Operation::get_op_dyn(last, ctx)
            .op_ref()
            .get_opid()
            .to_string()
            .as_str(),
        "air.br" | "air.cond_br" | "air.ret" | "air.unreachable"
    )
}

impl WaxToAir {
    /// `wax.if` -> `cond_br` over two arms that rejoin at a merge block.
    ///
    /// The structured-to-CFG shape MLIR's `SCFToControlFlow` uses, and the same one
    /// `walk.rs::translate_if` builds: split the containing block at the `if`, so the
    /// trailing operations become the merge block; move each arm's body into its own block;
    /// terminate the head with `cond_br`. Results become merge-block ARGUMENTS, which is how
    /// a block-argument IR spells a phi.
    ///
    /// The arms' bodies are MOVED, not cloned. The driver collected every nested operation
    /// into its worklist before calling any pattern, so those operations are already queued
    /// and still convert after the move - a clone would leave the originals queued too and
    /// convert them into a region that is about to be discarded.
    pub(crate) fn rewrite_if(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron::basic_block::BasicBlock;
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
        // A value-producing `if` needs both arms to have something to merge, which is the
        // same rule `translate_if` enforces.
        if n_results > 0 && n_regions < 2 {
            self.bail(Opcode::If, "value-producing if without an else arm");
            return Ok(());
        }
        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::If, "if is not inside a block");
            return Ok(());
        };
        let _region = match head.deref(ctx).get_container() {
            Some(r) => r,
            None => {
                self.bail(Opcode::If, "block is not inside a region");
                return Ok(());
            }
        };

        // Each arm must be a single block: a nested `if` that has already been lowered would
        // leave several, and rejoining those needs the arm's own exit block rather than its
        // entry. Refuse rather than branch to the wrong one.
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

        // Result types, converted, before anything moves.
        let mut merge_arg_tys = Vec::with_capacity(n_results);
        for i in 0..n_results {
            let t = op.deref(ctx).get_result(i).get_type(ctx);
            let Some(t) = self.air_ty(ctx, t) else {
                self.bail(Opcode::If, "result type has no air equivalent");
                return Ok(());
            };
            merge_arg_tys.push(t);
        }

        // Split through the REWRITER, not by hand. The rewriter notifies the conversion
        // driver's listener, which is what makes the driver treat the merge block as newly
        // inserted and convert its arguments; a block created straight off `BasicBlock::new`
        // is invisible to it, and these arguments are exactly the `if`'s result phis.
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

        // `split_block` moved the `if` itself into the merge block; put it back so the head
        // ends with the `cond_br` that replaces it.
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockEnd(head));

        // Inline each arm's region between the head and the merge block, again through the
        // rewriter so the arms' blocks are announced too.
        // Arms then merge, all at the region end: `walk.rs` creates then, else and merge in
        // that order and appends each, and block order is emission order.
        for i in 0..n_regions {
            let r = op.deref(ctx).get_region(i);
            rewriter.inline_region(ctx, r, BlockInsertionPoint::AtRegionEnd(region));
        }
        rewriter.move_block(ctx, merge, BlockInsertionPoint::AtRegionEnd(region));

        // An arm ends in `wax.yield`; that becomes the branch to the merge block, carrying
        // the yielded values as block arguments.
        for &arm in &arms {
            let last = arm.deref(ctx).iter(ctx).last();
            let (target_args, yield_op) = match last {
                Some(l) if try_opcode_of(ctx, l) == Some(Opcode::Yield) => {
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
            // Only if the arm is still open: a `break` inside it has already terminated it.
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
    /// `wax.for` -> the four-block counted loop the CFG path builds.
    ///
    /// ```text
    ///   head:  br hdr(lo, inits...)
    ///   hdr(iv, carries...):  cond = iv slt hi;  cond_br cond, body, exit
    ///   body:  ...;  br back
    ///   back:  iv_next = iv + step;  br hdr(iv_next, updates...)
    ///   exit:  the `for`'s results are hdr's carry arguments
    /// ```
    ///
    /// The induction variable and each carry are block ARGUMENTS of `hdr`, which is how a
    /// block-argument IR spells the phis `walk.rs` emits. `exit` needs no arguments of its
    /// own: it is dominated by `hdr`, so the carry arguments are in scope there, which is
    /// exactly what the CFG path means when it reads a carry phi from the exit block.
    ///
    /// One carry is one value here. `walk.rs` decomposes a carry into `nr * nc` fragment
    /// phis, but a fragment is what an MMA touches and this conversion refuses `MmaF` and the
    /// `sgmma_*` ops, so in any kernel it converts every carry is a single register.
    pub(crate) fn rewrite_for(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::air_dialect::attrs::{BinOpAttr, ICmpPredAttr};
        use pliron::basic_block::BasicBlock;
        use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
        use pliron::irbuild::rewriter::Rewriter;
        use pliron::linked_list::{ContainsLinkedList, LinkedList};
        use pliron::r#type::Typed;

        let (operands, n_carries) = {
            let o = op.deref(ctx);
            (o.operands().collect::<Vec<_>>(), o.get_num_results())
        };
        // operands are (lo, hi, step, inits...)
        if operands.len() < 3 + n_carries {
            self.bail(Opcode::For, "too few operands for lo/hi/step/inits");
            return Ok(());
        }
        let (lo, hi, step) = (operands[0], operands[1], operands[2]);
        let inits: Vec<_> = operands[3..3 + n_carries].to_vec();

        let body = {
            let r = op.deref(ctx).get_region(0);
            let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
            // A body that is already several blocks would need its EXIT block for the back
            // edge, not its entry. Nested control flow inside the body is still fine: the
            // driver rewrites this `for` while that `if` is a single unrewritten op.
            if blocks.len() != 1 {
                self.bail(Opcode::For, "body is not a single block");
                return Ok(());
            }
            blocks[0]
        };
        let body_args: Vec<_> = body.deref(ctx).arguments().collect();
        if body_args.len() != 1 + n_carries {
            self.bail(Opcode::For, "body arguments do not match the carries");
            return Ok(());
        }

        // The body's terminator supplies the next iteration's carries. `yield` and
        // `continue` mean the same thing here, as they do in `translate_for_body`.
        let term = body.deref(ctx).iter(ctx).last();
        let (term, updates) = match term {
            Some(t)
                if matches!(
                    try_opcode_of(ctx, t),
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
        if updates.len() != n_carries {
            self.bail(Opcode::For, "body yields the wrong number of carries");
            return Ok(());
        }

        // A loop the `cross_loop_reuse` pass has planned stages its loads into a private
        // `alloca` the partner loop reads back. The cache is created at the STAGING loop's
        // position - `walk.rs`'s place for it - whichever of the two loops the driver
        // reaches first, and it must exist before the split that moves this op.
        if let Some(&(pi, _)) = self.staging.cache_fors.get(&op) {
            self.ensure_cache_alloca(ctx, rewriter, pi)?;
        }

        let Some(head) = op.deref(ctx).get_container() else {
            self.bail(Opcode::For, "for is not inside a block");
            return Ok(());
        };

        // Per-carry register types, `carry_frag_tys` on the CFG path: a fragment carry is
        // `nr * nc` fragment registers (f32 only - mixed precision is refused there),
        // everything else is one scalar.
        let frag64 =
            crate::air_dialect::types::to_pliron(ctx, &LlType::vec(64, LlType::Float));
        let mut carry_frags: Vec<Option<(usize, usize)>> = Vec::with_capacity(n_carries);
        for i in 0..n_carries {
            let r = op.deref(ctx).get_result(i);
            if let Some((nr, nc)) = self.frag_dims(ctx, operands_info, r) {
                // `carry_frag_tys` refuses a non-f32 fragment carry; so do we.
                let ty = operands_info
                    .lookup_most_recent_type(r)
                    .unwrap_or_else(|| {
                        use pliron::r#type::Typed;
                        r.get_type(ctx)
                    });
                let Some(wax_ir::types::Type::Tile(t)) = wax_type_of(ctx, ty) else {
                    self.bail(Opcode::For, "fragment carry type unreadable");
                    return Ok(());
                };
                let wax_ir::types::TileElementType::Scalar(s) = &t.element_type else {
                    self.bail(Opcode::For, "fragment carry is not a scalar tile");
                    return Ok(());
                };
                if *s != wax_ir::types::ScalarType::F32 {
                    self.bail(Opcode::For, "mixed-precision fragment carry");
                    return Ok(());
                }
                carry_frags.push(Some((nr, nc)));
            } else {
                carry_frags.push(None);
            }
        }

        // Header argument types: the induction variable, then one register per carry -
        // `nr * nc` for a fragment carry.
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let mut hdr_arg_tys = vec![i32_ty];
        for (i, &carry) in carry_frags.iter().enumerate() {
            match carry {
                Some((nr, nc)) => hdr_arg_tys.extend(std::iter::repeat_n(frag64, nr * nc)),
                None => {
                    let t = op.deref(ctx).get_result(i).get_type(ctx);
                    let Some(t) = self.air_ty(ctx, t) else {
                        self.bail(Opcode::For, "carry type has no air equivalent");
                        return Ok(());
                    };
                    hdr_arg_tys.push(t);
                }
            }
        }
        // Where each carry's registers start in the header, in carry order.
        let mut carry_bases = Vec::with_capacity(n_carries);
        {
            let mut b = 1;
            for &carry in &carry_frags {
                carry_bases.push(b);
                b += match carry {
                    Some((nr, nc)) => nr * nc,
                    None => 1,
                };
            }
        }

        // Everything after the `for` becomes the exit block. Through the rewriter, so the
        // driver is told these blocks exist.
        let Some(region) = head.deref(ctx).get_container() else {
            self.bail(Opcode::For, "block is not inside a region");
            return Ok(());
        };
        let exit = rewriter.split_block(ctx, head, OpInsertionPoint::BeforeOperation(op), None);
        rewriter.move_operation(ctx, op, OpInsertionPoint::AtBlockEnd(head));

        // Blocks go to the END of the region, in creation order, because that is the order
        // the CFG builder produces and block order IS emission order. `walk.rs` creates
        // hdr, body, back, exit up front and appends each; inserting them adjacent to the
        // block they logically follow gives the same instructions in a different order, and
        // a different metallib.
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
        // `split_block` left the exit directly after the head; it is created fourth.
        rewriter.move_block(ctx, exit, BlockInsertionPoint::AtRegionEnd(region));

        let n_regs: usize = carry_frags
            .iter()
            .map(|f| match f {
                Some((nr, nc)) => nr * nc,
                None => 1,
            })
            .sum();
        let hdr_args: Vec<_> = (0..1 + n_regs)
            .map(|i| hdr.deref(ctx).get_argument(i))
            .collect();

        // The body's own arguments are replaced by the header's, then removed: the loop
        // carries live in exactly one place, and that place is `hdr`. A fragment carry is
        // the exception: its body argument STANDS FOR the grid, so it is retyped to one
        // fragment, registered in `grids`, and kept - the mma and the yield resolve it
        // through the table, and an argument no op references is legal in the emission.
        body_args[0].replace_all_uses_with(ctx, &hdr_args[0]);
        let mut remove_idx: Vec<usize> = vec![0];
        for i in 0..n_carries {
            let old = body_args[1 + i];
            match carry_frags[i] {
                None => {
                    old.replace_all_uses_with(ctx, &hdr_args[carry_bases[i]]);
                    remove_idx.push(1 + i);
                }
                Some((nr, nc)) => {
                    let group: Vec<_> =
                        hdr_args[carry_bases[i]..carry_bases[i] + nr * nc].to_vec();
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
            match carry_frags[i] {
                None => entry_args.push(init),
                Some((nr, nc)) => {
                    // The init defined before the loop is converted first, so its grid
                    // exists: a missing one is a bail, not a guess.
                    let Some(grid) = self.grid_of(init).cloned() else {
                        self.bail(Opcode::For, "fragment carry init has no grid");
                        return Ok(());
                    };
                    if grid.len() != nr * nc {
                        self.bail(Opcode::For, "fragment carry init grid size disagrees");
                        return Ok(());
                    }
                    entry_args.extend(grid);
                }
            }
        }
        let br_hdr = air::BrOp::new(ctx, hdr, entry_args);
        br_hdr.get_operation().insert_at_back(head, ctx);

        // hdr: cond = iv slt hi; cond_br cond, body, exit. The body's remaining arguments
        // are the fragment placeholders; the branch forwards each carry's first register
        // into its placeholder - the value is never read (consumers resolve the grid), but
        // the branch must feed every argument the block declares.
        let i1_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I1);
        let cmp = air::ICmpOp::new(ctx, i1_ty, hdr_args[0], hi);
        cmp.set_pred(ctx, ICmpPredAttr::Slt);
        cmp.get_operation().insert_at_back(hdr, ctx);
        let cond = cmp.get_operation().deref(ctx).get_result(0);
        let body_feed: Vec<pliron::value::Value> = (0..n_carries)
            .filter(|&i| carry_frags[i].is_some())
            .map(|i| hdr_args[carry_bases[i]])
            .collect();
        let cbr = air::CondBrOp::new(ctx, cond, body, body_feed.clone(), exit, vec![]);
        cbr.get_operation().insert_at_back(hdr, ctx);
        if !body_feed.is_empty() {
            self.control.body_cleanups.push(BodyCleanup {
                body,
                hdr,
                n: body_feed.len(),
            });
        }

        // body: the yield becomes a fall-through to the back edge. The update values stay
        // where they are defined; `back` is dominated by `body`, so it can name them.
        rewriter.erase_operation(ctx, term);
        if !block_is_terminated(ctx, body) {
            let br_back = air::BrOp::new(ctx, back, vec![]);
            br_back.get_operation().insert_at_back(body, ctx);
        }

        // back: iv_next = iv + step; br hdr(iv_next, updates...)
        let add = air::BinOp::new(ctx, i32_ty, hdr_args[0], step);
        add.set_op(ctx, BinOpAttr::Add);
        add.get_operation().insert_at_back(back, ctx);
        let iv_next = add.get_operation().deref(ctx).get_result(0);
        let mut back_args = vec![iv_next];
        let mut any_deferred = false;
        for (i, &upd) in updates.iter().enumerate() {
            match carry_frags[i] {
                // A single-fragment carry's update (the mma result) is re-wired to its
                // `<64 x float>` representative when the defining op converts, so the
                // branch can name the raw value now.
                None | Some((1, 1)) => back_args.push(upd),
                Some((_, _)) => {
                    // Multi-fragment: the update's grid does not exist until the body's mma
                    // is rewritten, which is AFTER this `for`. Defer the whole branch.
                    any_deferred = true;
                }
            }
        }
        if any_deferred {
            self.control.pending_latches.push(PendingLatch {
                back,
                hdr,
                iv_next,
                updates: updates.to_vec(),
                carry_frags: carry_frags.clone(),
            });
        } else {
            let br_latch = air::BrOp::new(ctx, hdr, back_args);
            br_latch.get_operation().insert_at_back(back, ctx);
        }

        self.control.loops.push((body, back));
        // The cache members index their slot with this loop's IV; the header argument 0 is
        // that IV now that the loop exists.
        if let Some(&(pi, is_staging)) = self.staging.cache_fors.get(&op) {
            let p = &mut self.staging.cache_plans[pi];
            if is_staging {
                p.staging_iv = Some(hdr_args[0]);
            } else {
                p.consuming_iv = Some(hdr_args[0]);
            }
        }
        // Recorded so a `wax.break` inside this body can branch straight to the exit. Only
        // for a loop with NO carries: the exit reads the header's carry arguments, and a
        // mid-body break cannot supply them - the same restriction `walk.rs` enforces.
        if n_carries == 0 {
            self.control.loop_exits.push(exit);
        }

        // The loop's results are the carry arguments, read from the exit block. A fragment
        // result stands for its register group, bound to the first register as the
        // representative - the same shape the mma and the fragment load use.
        let mut reps = Vec::with_capacity(n_carries);
        for i in 0..n_carries {
            match carry_frags[i] {
                None => reps.push(hdr_args[carry_bases[i]]),
                Some((nr, nc)) => {
                    let group: Vec<_> =
                        hdr_args[carry_bases[i]..carry_bases[i] + nr * nc].to_vec();
                    let Some(&rep) = group.first() else {
                        self.bail(Opcode::For, "empty fragment carry group");
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
    /// `wax.break` -> a branch to the innermost carry-free loop's exit.
    ///
    /// Carry-bearing loops are refused, as they are on the CFG path: the exit block reads the
    /// header's carry arguments and a mid-body break has no values to pass for them.
    pub(crate) fn rewrite_break(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        use pliron::linked_list::LinkedList;
        let Some(&exit) = self.control.loop_exits.last() else {
            self.bail(Opcode::Break, "break outside a carry-free loop");
            return Ok(());
        };
        let Some(block) = op.deref(ctx).get_container() else {
            self.bail(Opcode::Break, "break is not inside a block");
            return Ok(());
        };
        // A block that already has a terminator gets it REPLACED: a break always leaves, so
        // whatever fall-through was put there is unreachable.
        // The pointer is read out before erasing: `iter` holds a borrow that `erase` needs
        // mutably.
        let existing = {
            use pliron::linked_list::ContainsLinkedList;
            block.deref(ctx).iter(ctx).last()
        };
        if let Some(last) = existing
            && block_is_terminated(ctx, block)
        {
            rewriter.erase_operation(ctx, last);
        }
        let br = air::BrOp::new(ctx, exit, vec![]);
        br.get_operation().insert_at_back(block, ctx);
        rewriter.erase_operation(ctx, op);
        self.converted += 1;
        Ok(())
    }
    /// Build the branches `rewrite_for` deferred: one per loop whose back edge carries a
    /// multi-fragment update.
    ///
    /// Runs after the whole conversion, so the body's mma has registered the grid the
    /// back edge needs. If one is missing (the mma bailed) the branch is simply not
    /// built: the kernel fails the residual check either way, and a branch built on a
    /// guess would be worse than an incomplete one.
    pub(crate) fn finish_pending_latches(&mut self, ctx: &mut Context) {
        for l in std::mem::take(&mut self.control.pending_latches) {
            let mut args = vec![l.iv_next];
            let mut ok = true;
            for (i, &upd) in l.updates.iter().enumerate() {
                match l.carry_frags[i] {
                    None | Some((1, 1)) => args.push(upd),
                    Some((nr, nc)) => match self.grid_of(upd).cloned() {
                        Some(grid) if grid.len() == nr * nc => args.extend(grid),
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
    /// Strip the fragment placeholders a loop body still carries, and the branch feeds that
    /// keep them defined.
    ///
    /// Runs after the body's ops are converted: the placeholders' only users are the mma's
    /// operands, which are gone once the mma is rewritten. If a user remains (the mma bailed)
    /// the cleanup is skipped: the kernel fails the residual check anyway, and removing an
    /// argument that still has uses would leave a dangling reference.
    pub(crate) fn finish_body_cleanups(&mut self, ctx: &mut Context) {
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::op_interfaces::BranchOpInterface;
        use pliron::linked_list::ContainsLinkedList;
        use pliron::op::Op;
        for c in std::mem::take(&mut self.control.body_cleanups) {
            // The placeholders are the body's only remaining arguments. Their uses were
            // rewired to the header representatives when the loop was built, so they must
            // be dead now; if one still has users, a consumer bailed and the kernel fails
            // the residual check anyway - leave the IR alone then.
            let bb = c.body.deref(ctx);
            let n_args = bb.get_num_arguments();
            let alive = (0..n_args).any(|i| bb.get_argument(i).num_uses(ctx) != 0);
            drop(bb);
            if n_args != c.n || alive {
                continue;
            }
            // The hdr's terminator is the cond_br that feeds the body. Drop the feed.
            let hdr_ops: Vec<_> = c.hdr.deref(ctx).iter(ctx).collect();
            let Some(&term) = hdr_ops.last() else {
                continue;
            };
            let succs: Vec<_> = term.deref(ctx).successors().collect();
            let Some(succ_idx) = succs.iter().position(|&s| s == c.body) else {
                continue;
            };
            let cb = air::CondBrOp::wrap_operation(term)
                .downcast::<air::CondBrOp>()
                .expect("hdr terminator is a cond_br");
            for i in (0..c.n).rev() {
                cb.remove_successor_operand(ctx, succ_idx, i);
            }
            // Then the arguments themselves, highest index first.
            for i in (0..c.n).rev() {
                BasicBlock::remove_argument(c.body, ctx, i);
            }
        }
    }
}
