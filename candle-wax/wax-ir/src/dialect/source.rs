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
        .get::<T>(&ATTR_KEY_WAX_OP.try_into().unwrap())
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
            (Operation::is_op::<BreakOp>(op, ctx), Opcode::Break),
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
