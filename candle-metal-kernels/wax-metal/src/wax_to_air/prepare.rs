//! Preparatory steps applied to IR before conversion begins.
//! Add promoted dyn strides, extents, and `%_tp`/`%_tg` arguments. Remove redundant parameters.

use super::*;
use crate::air_dialect::types as t;
use pliron::basic_block::BasicBlock;
use pliron::irbuild::{inserter::IRInserter, listener::DummyListener};
use pliron::r#type::Typed;
use pliron::value::Value;

/// Add promoted dyn strides, extents, and `%_tp`/`%_tg` arguments.
pub fn add_implicit_params(
    ctx: &mut Context,
    func: Ptr<Operation>,
    entry: Ptr<BasicBlock>,
) -> (Value, Value, Vec<Value>) {
    let stride_args = promoted_dyn_stride_args(ctx, func, entry);
    let extent_args = promoted_extent_args(ctx, func, entry);
    let mut ordered = stride_args.clone();
    ordered.extend(extent_args.iter().copied());
    let mut loaded = promote_scalar_params(ctx, entry, &ordered);
    let promoted = loaded.split_off(stride_args.len());
    drop_scalar_tile_params(ctx, entry);
    let vec3 = t::vec_ty(ctx, 3, t::i32_ty(ctx));
    let at = entry.deref(ctx).get_num_arguments();
    BasicBlock::insert_argument(entry, ctx, at, vec3);
    BasicBlock::insert_argument(entry, ctx, at + 1, vec3);
    let b = entry.deref(ctx);
    (b.get_argument(at), b.get_argument(at + 1), promoted)
}

/// Remove redundant parameters.
pub(crate) fn drop_scalar_tile_params(ctx: &mut Context, entry: Ptr<BasicBlock>) {
    use wax_ir::types::TileElementType;

    // Reverse traversal so removals does not shift unprocessed items.
    let n = entry.deref(ctx).get_num_arguments();
    for i in (0..n).rev() {
        let arg = entry.deref(ctx).get_argument(i);
        let elem = match wax_type_of(ctx, arg.get_type(ctx)) {
            Some(WaxType::Tile(t)) if t.shape.is_empty() => match t.element_type {
                TileElementType::Scalar(s) => s,
                _ => continue,
            },
            _ => continue,
        };
        // Skip. `convert_type` will refuse by name.
        let Some(ty) = air_type_for(ctx, &WaxType::Scalar(elem.clone())) else {
            continue;
        };
        // Same.
        let Some(zero) = air::const_bits(ctx, ty, 0) else {
            continue;
        };
        let zero_op = zero.get_operation();
        // Place constant at block start.
        entry_start(entry).append_operation(ctx, zero_op);
        let zero_val = zero_op.deref(ctx).get_result(0);
        arg.replace_all_uses_with(ctx, &zero_val);
        BasicBlock::remove_argument(entry, ctx, i);
    }
}

fn entry_start(block: Ptr<BasicBlock>) -> IRInserter<DummyListener> {
    IRInserter::new_at_block_start(block)
}

/// The function's entry block.
pub(crate) fn entry_block(ctx: &Context, func: Ptr<Operation>) -> Option<Ptr<BasicBlock>> {
    use pliron::linked_list::ContainsLinkedList;
    let r = func.deref(ctx).regions().next()?;
    r.deref(ctx).iter(ctx).next()
}

/// The wax ops still present after conversion.
pub(crate) fn residual_wax_ops(ctx: &Context, func: Ptr<Operation>) -> Unconverted {
    let mut left = Unconverted::default();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if let Some(code) = opcode_of(ctx, op)
                && code != Opcode::Entry
            {
                left.note(code);
            }
        }
    });
    left
}

/// Block arguments with runtime extent.
pub(crate) fn promoted_extent_args(
    ctx: &Context,
    func: Ptr<Operation>,
    entry: Ptr<BasicBlock>,
) -> Vec<Value> {
    // Tensor views behind a padded store destination.
    let mut want: Vec<Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if opcode_of(ctx, op) != Some(Opcode::StoreViewTko) {
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
            // Partitions first operand is the tensor view it came from.
            if let pliron::value::DefiningEntity::Op(def) = view.defining_entity()
                && opcode_of(ctx, def) == Some(Opcode::MakePartitionView)
                && let Some(tv) = def.deref(ctx).operands().next()
            {
                want.push(tv);
            }
        }
    });
    if want.is_empty() {
        return Vec::new();
    }

    let mut ordered: Vec<Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if opcode_of(ctx, op) != Some(Opcode::MakeTensorView) {
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

    order_as_entry_args(ctx, entry, &mut ordered);
    ordered
}

/// See through a `wax.assume` to the value it forwards.
pub(crate) fn trace_through_assume(ctx: &Context, v: Value) -> Value {
    match v.defining_entity() {
        pliron::value::DefiningEntity::Op(def) if opcode_of(ctx, def) == Some(Opcode::Assume) => {
            def.deref(ctx).operands().next().unwrap_or(v)
        }
        _ => v,
    }
}

/// Position of a value in the entry block's arguments. `u32::MAX` means not found.
pub(crate) fn entry_arg_index(ctx: &Context, entry: Ptr<BasicBlock>, v: &Value) -> u32 {
    let n = entry.deref(ctx).get_num_arguments();
    (0..n)
        .find(|&i| entry.deref(ctx).get_argument(i) == *v)
        .map(|i| i as u32)
        .unwrap_or(u32::MAX)
}

/// Signature order is block argument order.
pub(crate) fn order_as_entry_args(ctx: &Context, entry: Ptr<BasicBlock>, ordered: &mut Vec<Value>) {
    ordered.sort_by_key(|v| entry_arg_index(ctx, entry, v));
    ordered.retain(|v| entry_arg_index(ctx, entry, v) != u32::MAX);
}

/// Dynamic strides extracted from `MakeTensorView`.
pub(crate) fn promoted_dyn_stride_args(
    ctx: &Context,
    func: Ptr<Operation>,
    entry: Ptr<BasicBlock>,
) -> Vec<Value> {
    let mut ordered: Vec<Value> = Vec::new();
    for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if opcode_of(ctx, op) != Some(Opcode::MakeTensorView) {
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

/// Turn each promoted runtime scalar (dynamic stride or extent) into a constant buffer parameter.
pub(crate) fn promote_scalar_params(
    ctx: &mut Context,
    entry: Ptr<BasicBlock>,
    params: &[Value],
) -> Vec<Value> {
    if params.is_empty() {
        return Vec::new();
    }
    let mut loaded_vals: Vec<Option<Value>> = Vec::new();
    let ptr_ty = t::const_ptr_ty(ctx, t::i32_ty(ctx));
    let i32_ty = t::i32_ty(ctx);

    // First add args in signature order.
    let mut args = Vec::with_capacity(params.len());
    for &old in params {
        let at = entry.deref(ctx).get_num_arguments();
        BasicBlock::insert_argument(entry, ctx, at, ptr_ty);
        args.push((old, entry.deref(ctx).get_argument(at)));
    }
    loaded_vals.resize(args.len(), None);

    // Then insert loads at the top of the block.
    for (i, &(old, arg)) in args.iter().enumerate().rev() {
        let ld = air::LoadOp::new(ctx, i32_ty, arg, 4);
        entry_start(entry).append_operation(ctx, ld.get_operation());
        let loaded = ld.get_operation().deref(ctx).get_result(0);
        old.replace_all_uses_with(ctx, &loaded);
        loaded_vals[i] = Some(loaded);
    }
    loaded_vals.into_iter().flatten().collect()
}
