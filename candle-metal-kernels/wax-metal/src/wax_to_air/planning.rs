//! Pre-conversion planning: the implicit parameters and their promotion, and the
//! passes that read the still-wax function before the driver rewrites it in its own
//! order. Nothing in here emits instructions.

use super::*;

/// Add the parameters the AIR function has and the wax function does not.
///
/// `lower/walk.rs` appends these when it builds the CFG: `%_tp` and `%_tg` are the thread and
/// threadgroup position as `<3 x i32>`. They come last, after the buffer pointers and after any
/// `%_stride_arg_i` / `%_dim_arg_i` - order matters, because Metal binds by position.
///
/// Returns the block arguments for `(tp, tg)`.
///
/// **Only the positional case.** Dynamic strides and extents also add parameters, derived from
/// analysis of the wax module rather than from its signature (`find_dyn_shape_block_args`).
/// A kernel needing them is not handled here.
pub fn add_implicit_params(
    ctx: &mut Context,
    func: Ptr<Operation>,
    entry: Ptr<pliron::basic_block::BasicBlock>,
) -> (
    pliron::value::Value,
    pliron::value::Value,
    Vec<pliron::value::Value>,
) {
    use pliron::basic_block::BasicBlock;
    // Runtime extents a padded store needs become `i32 addrspace(2)*` parameters, loaded at
    // the prologue - the signature the CFG path calls `%_dim_arg_i`. Done BEFORE the scalar
    // parameters are dropped, because the promoted values ARE scalar parameters and would
    // otherwise be bound to constant 0.
    // Dynamic STRIDE arguments first: the CFG signature is buffers, dyn strides, dyn
    // extents, %_tp, %_tg, and the prologue loads follow the same order. Promoting them
    // before the extents keeps both.
    let stride_args = promoted_dyn_stride_args(ctx, func, entry);
    let extent_args = promoted_extent_args(ctx, func, entry);
    let mut ordered = stride_args.clone();
    ordered.extend(extent_args.iter().copied());
    let mut loaded = promote_scalar_params(ctx, entry, &ordered);
    let promoted = loaded.split_off(stride_args.len());
    drop_scalar_tile_params(ctx, entry);
    let vec3 = crate::air_dialect::types::to_pliron(ctx, &LlType::vec(3, LlType::I32));
    let at = entry.deref(ctx).get_num_arguments();
    BasicBlock::insert_argument(entry, ctx, at, vec3);
    BasicBlock::insert_argument(entry, ctx, at + 1, vec3);
    let b = entry.deref(ctx);
    (b.get_argument(at), b.get_argument(at + 1), promoted)
}

/// Remove the parameters the AIR signature does not carry.
///
/// A rank-0 scalar tile - a stride or extent passed by value - is not in the AIR signature at
/// all: `lower/walk.rs`'s `param_needs_load` marks it, and the CFG path resolves every use to
/// the constant 0. Leaving the block argument in place would put an extra parameter in the
/// emitted signature and shift every value number after it.
///
/// Uses are replaced by `air.const_bits 0`, which is what the CFG path binds them to.
pub(crate) fn drop_scalar_tile_params(ctx: &mut Context, entry: Ptr<pliron::basic_block::BasicBlock>) {
    use wax_ir::types::{TileElementType, Type as WT};
    use pliron::basic_block::BasicBlock;
    use pliron::r#type::Typed;

    // Back to front, so removing one does not shift the indices still to check.
    let n = entry.deref(ctx).get_num_arguments();
    for i in (0..n).rev() {
        let arg = entry.deref(ctx).get_argument(i);
        let is_scalar_tile = matches!(
            wax_type_of(ctx, arg.get_type(ctx)),
            Some(WT::Tile(t)) if t.shape.is_empty()
                && matches!(t.element_type, TileElementType::Scalar(_))
        );
        if !is_scalar_tile {
            continue;
        }
        let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
        let zero = air::ConstBitsOp::new(ctx, i32_ty, 0);
        let zero_op = zero.get_operation();
        // The constant must dominate every use, so it goes at the top of the entry block. The
        // first op is read out before inserting: `iter` holds a borrow that `insert_before`
        // would take again.
        let first = {
            use pliron::linked_list::ContainsLinkedList;
            entry.deref(ctx).iter(ctx).next()
        };
        match first {
            Some(f) => zero_op.insert_before(ctx, f),
            None => zero_op.insert_at_back(entry, ctx),
        }
        let zero_val = zero_op.deref(ctx).get_result(0);
        arg.replace_all_uses_with(ctx, &zero_val);
        BasicBlock::remove_argument(entry, ctx, i);
    }
}

/// The function's entry block.
pub(crate) fn entry_block(
    ctx: &Context,
    func: Ptr<Operation>,
) -> Option<Ptr<pliron::basic_block::BasicBlock>> {
    use pliron::linked_list::ContainsLinkedList;
    let r = func.deref(ctx).regions().next()?;
    r.deref(ctx).iter(ctx).next()
}

/// Retype the entry arguments named by `const_param_slots` as constant-space pointers.
///
/// `drop_scalar_tile_params` has already removed the by-value scalar parameters, so the
/// remaining arguments are exactly the buffer parameters in slot order and the slot index is
/// the argument index.
pub(crate) fn retype_const_params(
    ctx: &mut Context,
    func: Ptr<Operation>,
    entry: Ptr<pliron::basic_block::BasicBlock>,
) {
    use wax_ir::attr::Attribute as WA;
    use wax_ir::dialect::attr_mirror::WaxAttr;
    use pliron::r#type::Typed;
    let Ok(key) = "const_param_slots".try_into() else {
        return;
    };
    let slots: Vec<usize> = match func.deref(ctx).attributes.get::<WaxAttr>(&key) {
        Some(a) => match &a.0 {
            WA::DenseI32Array(v) => v.iter().map(|&x| x as usize).collect(),
            other => {
                why(&format!("const_param_slots wrong variant: {other:?}"));
                return;
            }
        },
        None => {
            why("const_param_slots attribute absent");
            return;
        }
    };
    why(&format!("const slots: {slots:?}"));
    for slot in slots {
        if slot >= entry.deref(ctx).get_num_arguments() {
            continue;
        }
        let v = entry.deref(ctx).get_argument(slot);
        let Some(wax) = wax_type_of(ctx, v.get_type(ctx)) else {
            continue;
        };
        let Some(LlType::Ptr { pointee, .. }) = air_type_for(&wax) else {
            continue;
        };
        let ty = crate::air_dialect::types::to_pliron(ctx, &LlType::const_ptr(*pointee));
        v.set_type(ctx, ty);
    }
}

/// The wax ops still present after conversion, by opcode.
///
/// This is the post-condition that matters: anything left is an op the air emitter would meet
/// and not understand. It is deliberately not the same thing as the notes the arms record,
/// which are a work list for a human and include ops that were later erased by their parent.
pub(crate) fn residual_wax_ops(ctx: &Context, func: Ptr<Operation>) -> Unconverted {
    let mut left = Unconverted::default();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if let Some(code) = try_opcode_of(ctx, op)
                && code != Opcode::Entry
            {
                left.note(code);
            }
        }
    });
    left
}

/// Block arguments carrying a runtime extent that a padded store needs.
///
/// Mirrors `lower::find_dyn_shape_block_args`, including its gate: promotion happens only
/// for a kernel that masks its STORE. `Tensor::partition` declares `padding::Zero` by
/// default, so nearly every view is nominally padded, and promoting on that alone would add
/// parameters to any kernel reading a runtime-shaped tensor - an ABI change visible only as a
/// wrong answer at dispatch. That gate cost two copies of `gemm` their results once.
///
/// Returned in block-argument order, which is the order they take in the signature.
pub(crate) fn promoted_extent_args(
    ctx: &Context,
    func: Ptr<Operation>,
    entry: Ptr<pliron::basic_block::BasicBlock>,
) -> Vec<pliron::value::Value> {
    use pliron::r#type::Typed;
    // Tensor views behind a padded store destination.
    let mut want: Vec<pliron::value::Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if try_opcode_of(ctx, op) != Some(Opcode::StoreViewTko) {
                continue;
            }
            let Some(view) = op.deref(ctx).operands().nth(1) else {
                continue;
            };
            let padded = matches!(
                wax_type_of(ctx, view.get_type(ctx)),
                Some(wax_ir::types::Type::PartitionView(pv)) if pv.padding_value.is_some()
            );
            if !padded {
                continue;
            }
            // The partition's first operand is the tensor view it came from.
            if let pliron::value::DefiningEntity::Op(def) = view.defining_entity()
                && try_opcode_of(ctx, def) == Some(Opcode::MakePartitionView)
                && let Some(tv) = def.deref(ctx).operands().next()
            {
                want.push(tv);
            }
        }
    });
    if want.is_empty() {
        return Vec::new();
    }

    let mut ordered: Vec<pliron::value::Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if try_opcode_of(ctx, op) != Some(Opcode::MakeTensorView) {
                continue;
            }
            let produces_wanted = {
                let o = op.deref(ctx);
                (0..o.get_num_results()).any(|i| want.contains(&o.get_result(i)))
            };
            if !produces_wanted {
                continue;
            }
            let (n_base, n_dyn_shape) = {
                let (b, s, _) = wax_operand_segments(ctx, op);
                (b, s)
            };
            if n_dyn_shape == 0 {
                continue;
            }
            let operands: Vec<_> = op.deref(ctx).operands().collect();
            for &sv in operands
                .iter()
                .skip(n_base)
                .take(n_dyn_shape.min(operands.len().saturating_sub(n_base)))
            {
                let ba = trace_through_assume(ctx, sv);
                if !ordered.contains(&ba) {
                    ordered.push(ba);
                }
            }
        }
    });

    // Signature order is block-argument order; anything not an entry argument sorts last,
    // matching the `u32::MAX` key the CFG path uses.
    order_as_entry_args(ctx, entry, &mut ordered);
    ordered
}

/// See through a `wax.assume` to the value it forwards.
pub(crate) fn trace_through_assume(ctx: &Context, v: pliron::value::Value) -> pliron::value::Value {
    match v.defining_entity() {
        pliron::value::DefiningEntity::Op(def)
            if try_opcode_of(ctx, def) == Some(Opcode::Assume) =>
        {
            def.deref(ctx).operands().next().unwrap_or(v)
        }
        _ => v,
    }
}

/// Turn each promoted runtime extent into a constant-buffer parameter loaded at the prologue.
///
/// The by-value scalar parameter is replaced by a pointer parameter plus an `air.load`, which
/// is what `setup_params` does for `%_dim_arg_i`. The pointer parameters are appended, so they
/// land after the buffers and before `%_tp`/`%_tg`, matching the CFG signature order.
/// The position of a value among the entry block's arguments; a non-argument
/// comes back as `u32::MAX`, which is also the key the CFG path sorts it under.
pub(crate) fn entry_arg_index(
    ctx: &Context,
    entry: Ptr<pliron::basic_block::BasicBlock>,
    v: &pliron::value::Value,
) -> u32 {
    let n = entry.deref(ctx).get_num_arguments();
    (0..n)
        .find(|&i| entry.deref(ctx).get_argument(i) == *v)
        .map(|i| i as u32)
        .unwrap_or(u32::MAX)
}

/// Signature order is block-argument order: sort the discovered values, then keep only the
/// ones that are entry arguments at all.
pub(crate) fn order_as_entry_args(
    ctx: &Context,
    entry: Ptr<pliron::basic_block::BasicBlock>,
    ordered: &mut Vec<pliron::value::Value>,
) {
    ordered.sort_by_key(|v| entry_arg_index(ctx, entry, v));
    ordered.retain(|v| entry_arg_index(ctx, entry, v) != u32::MAX);
}

/// The entry block arguments that feed a `MakeTensorView` DYNAMIC stride slot, in
/// block-argument order - the pliron mirror of `lower::find_dyn_stride_block_args`.
///
/// The CFG path carries exactly these as `i32 addrspace(2)*` parameters (loaded at the
/// prologue, ordered before the promoted extents), so the conversion must promote the
/// same set in the same order or the signatures - and the loads they drive - diverge.
pub(crate) fn promoted_dyn_stride_args(
    ctx: &Context,
    func: Ptr<Operation>,
    entry: Ptr<pliron::basic_block::BasicBlock>,
) -> Vec<pliron::value::Value> {
    let mut ordered: Vec<pliron::value::Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if try_opcode_of(ctx, op) != Some(Opcode::MakeTensorView) {
                continue;
            }
            let (n_base, n_dyn_shape, n_dyn_stride) = wax_operand_segments(ctx, op);
            if n_dyn_stride == 0 {
                continue;
            }
            let operands: Vec<_> = op.deref(ctx).operands().collect();
            let start = n_base + n_dyn_shape;
            for &sv in operands
                .iter()
                .skip(start)
                .take(n_dyn_stride.min(operands.len().saturating_sub(start)))
            {
                let ba = trace_through_assume(ctx, sv);
                if !ordered.contains(&ba) {
                    ordered.push(ba);
                }
            }
        }
    });
    order_as_entry_args(ctx, entry, &mut ordered);
    ordered
}

/// Turn each promoted runtime scalar (a dynamic stride extent) into a constant-buffer
/// parameter loaded at the prologue.
///
/// The by-value scalar parameter is replaced by a pointer parameter plus an `air.load`, which
/// is what `setup_params` does for the CFG path. The pointer parameters are appended in the
/// caller's order - that order IS the signature - and the prologue loads are inserted in
/// REVERSE so the stream order matches the parameter order.
pub(crate) fn promote_scalar_params(
    ctx: &mut Context,
    entry: Ptr<pliron::basic_block::BasicBlock>,
    ordered: &[pliron::value::Value],
) -> Vec<pliron::value::Value> {
    use pliron::basic_block::BasicBlock;
    use pliron::linked_list::ContainsLinkedList;
    why(&format!("promote: {} scalar arg(s)", ordered.len()));
    let promoted = ordered;
    if promoted.is_empty() {
        return Vec::new();
    }
    let mut loaded_vals: Vec<Option<pliron::value::Value>> = Vec::new();
    let ptr_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::const_ptr(LlType::I32));
    let i32_ty = crate::air_dialect::types::to_pliron(ctx, &LlType::I32);
    // Two passes. The parameters are APPENDED in order, because that order is the signature.
    // The prologue loads are then inserted in REVERSE, because each goes at the top of the
    // entry block to dominate its uses - inserting them forward puts each new load before the
    // previous one, emitting the loads in the opposite order to the parameters. Same kernel,
    // different metallib.
    let mut args = Vec::with_capacity(promoted.len());
    for &old in promoted {
        let at = entry.deref(ctx).get_num_arguments();
        BasicBlock::insert_argument(entry, ctx, at, ptr_ty);
        args.push((old, entry.deref(ctx).get_argument(at)));
    }
    loaded_vals.resize(args.len(), None);
    for (i, &(old, arg)) in args.iter().enumerate().rev() {
        let ld = air::LoadOp::new(ctx, i32_ty, arg);
        air::set_align(ld.get_operation(), ctx, 4);
        let first = entry.deref(ctx).iter(ctx).next();
        match first {
            Some(f) => ld.get_operation().insert_before(ctx, f),
            None => ld.get_operation().insert_at_back(entry, ctx),
        }
        let loaded = ld.get_operation().deref(ctx).get_result(0);
        old.replace_all_uses_with(ctx, &loaded);
        loaded_vals[i] = Some(loaded);
    }
    loaded_vals.into_iter().flatten().collect()
}

