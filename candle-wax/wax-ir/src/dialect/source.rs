//! Reading an op back out of the dialect.
use super::attrs::{BinaryAttr, CastAttr, CmpKindAttr, UnaryAttr};
use super::ops::*;
use crate::Opcode;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;

/// The operator attribute of a uniform family op.
fn family<T>(ctx: &Context, op: Ptr<Operation>) -> T
where
    T: pliron::attribute::Attribute + Clone,
{
    op.deref(ctx)
        .attributes
        .get::<T>(&ATTR_KEY_WAX_OP)
        .expect("family op has no operator attribute")
        .clone()
}

/// The opcode an op came from. Requires the original `Context`.
pub fn opcode_of(ctx: &Context, op: Ptr<Operation>) -> Opcode {
    {
        if Operation::is_op::<pliron::builtin::ops::FuncOp>(op, ctx) {
            return Opcode::Entry;
        }
        if Operation::is_op::<BinaryOp>(op, ctx) {
            return family::<BinaryAttr>(ctx, op).to_opcode();
        }
        if Operation::is_op::<UnaryOp>(op, ctx) {
            return family::<UnaryAttr>(ctx, op).to_opcode();
        }
        if Operation::is_op::<CastOp>(op, ctx) {
            return family::<CastAttr>(ctx, op).to_opcode();
        }
        if Operation::is_op::<CmpOp>(op, ctx) {
            return match family::<CmpKindAttr>(ctx, op) {
                CmpKindAttr::Float => Opcode::CmpF,
                CmpKindAttr::Int => Opcode::CmpI,
            };
        }
        for (is, code) in [
            (Operation::is_op::<FmaOp>(op, ctx), Opcode::Fma),
            (Operation::is_op::<SelectOp>(op, ctx), Opcode::Select),
            (Operation::is_op::<ConstantOp>(op, ctx), Opcode::Constant),
            (Operation::is_op::<BroadcastOp>(op, ctx), Opcode::Broadcast),
            (Operation::is_op::<ReshapeOp>(op, ctx), Opcode::Reshape),
            (Operation::is_op::<ExtractOp>(op, ctx), Opcode::Extract),
            (Operation::is_op::<OffsetOp>(op, ctx), Opcode::Offset),
            (Operation::is_op::<MmaFOp>(op, ctx), Opcode::MmaF),
            (Operation::is_op::<AssumeOp>(op, ctx), Opcode::Assume),
            (Operation::is_op::<AssertOp>(op, ctx), Opcode::Assert),
            (Operation::is_op::<MakeTokenOp>(op, ctx), Opcode::MakeToken),
            (
                Operation::is_op::<MakeTensorViewOp>(op, ctx),
                Opcode::MakeTensorView,
            ),
            (
                Operation::is_op::<MakePartitionViewOp>(op, ctx),
                Opcode::MakePartitionView,
            ),
            (Operation::is_op::<LoadViewOp>(op, ctx), Opcode::LoadViewTko),
            (
                Operation::is_op::<StoreViewOp>(op, ctx),
                Opcode::StoreViewTko,
            ),
            (Operation::is_op::<YieldOp>(op, ctx), Opcode::Yield),
            (Operation::is_op::<ContinueOp>(op, ctx), Opcode::Continue),
            (Operation::is_op::<ForOp>(op, ctx), Opcode::For),
            (Operation::is_op::<IfOp>(op, ctx), Opcode::If),
            (Operation::is_op::<ReturnOp>(op, ctx), Opcode::Return),
        ] {
            if is {
                return code;
            }
        }
        if let Some(code) = simple_to_opcode(ctx, op) {
            return code;
        }
        // TODO: Use proper error / Result instead
        panic!("no cutile opcode for this `wax` op");
    }
}

/// `op`'s operands.
pub fn operands_of(ctx: &Context, op: Ptr<Operation>) -> Vec<pliron::value::Value> {
    op.deref(ctx).operands().collect()
}

/// `op`'s results.
pub fn results_of(ctx: &Context, op: Ptr<Operation>) -> Vec<pliron::value::Value> {
    let o = op.deref(ctx);
    (0..o.get_num_results()).map(|i| o.get_result(i)).collect()
}

/// Attempt to get wax type of a pliron value.
pub fn value_type(ctx: &Context, v: pliron::value::Value) -> Option<crate::types::Type> {
    use pliron::r#type::Typed;
    crate::dialect::types::from_pliron(ctx, v.get_type(ctx))
}

/// Wax attributes of `op`. Pliron builtin's are not included.
pub fn attributes_of(ctx: &Context, op: Ptr<Operation>) -> Vec<(String, crate::attr::Attribute)> {
    use crate::dialect::attr_mirror::WaxAttr;
    op.deref(ctx)
        .attributes
        .0
        .iter()
        .filter_map(|(k, v)| {
            v.downcast_ref::<WaxAttr>()
                .map(|a| (k.to_string(), a.0.clone()))
        })
        .collect()
}

/// Op that defines `v`.
pub fn defining_op(v: pliron::value::Value) -> Option<Ptr<Operation>> {
    use pliron::value::DefiningEntity;
    match v.defining_entity() {
        DefiningEntity::Op(o) => Some(o),
        _ => None,
    }
}

/// Operations in `blk`.
pub fn ops_in(ctx: &Context, blk: Ptr<pliron::basic_block::BasicBlock>) -> Vec<Ptr<Operation>> {
    use pliron::linked_list::ContainsLinkedList;
    blk.deref(ctx).iter(ctx).collect()
}

/// Get i64 value of a `wax.constant`
///
/// Supports both `Attribute::Integer` and `Attribute::DenseElements`.
pub fn const_int_of(ctx: &Context, op: Ptr<Operation>) -> Option<i64> {
    if opcode_of(ctx, op) != Opcode::Constant {
        return None;
    }
    let a = op
        .deref(ctx)
        .attributes
        .get::<crate::dialect::attr_mirror::WaxAttr>(&crate::build::ATTR_VALUE)?
        .0
        .clone();
    match a {
        crate::attr::Attribute::DenseElements(de) => {
            let b = de.data.get(..4)?;
            Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        }
        crate::attr::Attribute::Integer(i, _) => Some(i),
        _ => None,
    }
}

/// Get i64 attribute of `op` by key.
pub fn attr_i64(ctx: &Context, op: Ptr<Operation>, key: &str) -> Option<i64> {
    attributes_of(ctx, op)
        .into_iter()
        .find_map(|(k, v)| match v {
            crate::attr::Attribute::Integer(i, _) if k == key => Some(i),
            _ => None,
        })
}

/// Every operation and block under `root`.
///
/// `root` is excluded since it is the pliron function, not wax with opcode etc.
pub fn ops_and_blocks_under(
    ctx: &Context,
    root: Ptr<Operation>,
) -> (
    Vec<Ptr<Operation>>,
    Vec<Ptr<pliron::basic_block::BasicBlock>>,
) {
    use pliron::graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op};

    let mut state: (
        Ptr<Operation>,
        Vec<Ptr<Operation>>,
        Vec<Ptr<pliron::basic_block::BasicBlock>>,
    ) = (root, Vec::new(), Vec::new());
    walk_op(
        ctx,
        &mut state,
        &WalkConfig::default(),
        root,
        |_ctx, (root, ops, blocks), node| match node {
            IRNode::Operation(op) if op != *root => ops.push(op),
            IRNode::BasicBlock(b) => blocks.push(b),
            _ => {}
        },
    );
    (state.1, state.2)
}
