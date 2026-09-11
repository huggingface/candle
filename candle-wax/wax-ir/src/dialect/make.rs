//! Create `wax` dialect op from cutile opcode.
use crate::Attribute;
use crate::Opcode;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;

use super::attrs::{BinaryAttr, CastAttr, CmpKindAttr, UnaryAttr};
use super::ops::*;

#[derive(Debug, PartialEq)]
pub enum WaxBridgeErr {
    /// Opcode does not exixt in our dialect.
    UnmappedOpcode(String),
    /// Type does not exist in our dialect.
    UnmappedType(String),
    /// Incorrect arity (operand count) for this op.
    Arity { op: String, got: usize },
    /// An attribute variant with no mirror.
    Attr(String),
    /// Op produces fewer results than the cutile op definition.
    ResultCount { op: String, want: usize, got: usize },
    /// Op has a different region count than the cutile op definition.
    RegionCount { op: String, want: usize, got: usize },
}

// TODO: use thiserror
impl std::fmt::Display for WaxBridgeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaxBridgeErr::UnmappedOpcode(o) => write!(f, "no `wax` op for opcode `{o}`"),
            WaxBridgeErr::UnmappedType(t) => write!(f, "no `wax` type for `{t}`"),
            WaxBridgeErr::Arity { op, got } => {
                write!(f, "`{op}` received incorrect amount of operands: {got}")
            }
            WaxBridgeErr::Attr(a) => write!(f, "{a}"),
            WaxBridgeErr::ResultCount { op, want, got } => {
                write!(
                    f,
                    "`{op}` should produce {want} results, the `wax` op produces {got}"
                )
            }
            WaxBridgeErr::RegionCount { op, want, got } => {
                write!(
                    f,
                    "`{op}` should have {want} regions, the `wax` op has {got}"
                )
            }
        }
    }
}

/// Renders the attributes of an op.
fn render_attrs_of(attrs: &[(String, Attribute)]) -> String {
    attrs
        .iter()
        .map(|(k, v)| format!("{k}={v:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Get integer attribute from list by name.
fn int_attr_of(attrs: &[(String, Attribute)], name: &str) -> Option<i64> {
    attrs.iter().find_map(|(k, v)| match v {
        Attribute::Integer(i, _) if k == name => Some(*i),
        _ => None,
    })
}

/// Build the `tile` op for one opcode.
pub fn make_op(
    ctx: &mut Context,
    opcode: Opcode,
    res_tys: &[TypeHandle],
    opds: Vec<Value>,
    attrs: &[(String, Attribute)],
    n_regions: usize,
) -> Result<Ptr<Operation>, WaxBridgeErr> {
    // Single op + operator in typed attribute.
    if let Some(a) = BinaryAttr::from_opcode(opcode) {
        let ty = res_tys.first().cloned().expect("binary produces a value");
        let o = BinaryOp::new(ctx, ty, opds[0], opds[1]);
        set_binary(o.get_operation(), ctx, a);
        return Ok(o.get_operation());
    }
    if let Some(a) = UnaryAttr::from_opcode(opcode) {
        let ty = res_tys.first().cloned().expect("unary produces a value");
        let o = UnaryOp::new(ctx, ty, opds[0]);
        set_unary(o.get_operation(), ctx, a);
        return Ok(o.get_operation());
    }
    if let Some(a) = CastAttr::from_opcode(opcode) {
        let ty = res_tys.first().cloned().expect("cast produces a value");
        let o = CastOp::new(ctx, ty, opds[0]);
        set_cast(o.get_operation(), ctx, a);
        return Ok(o.get_operation());
    }

    let o: Ptr<Operation> = match opcode {
        Opcode::CmpF | Opcode::CmpI => {
            let ty = res_tys.first().cloned().expect("cmp produces a value");
            let kind = if opcode == Opcode::CmpF {
                CmpKindAttr::Float
            } else {
                CmpKindAttr::Int
            };
            let o = CmpOp::new(ctx, ty, opds[0], opds[1]);
            set_cmp(
                o.get_operation(),
                ctx,
                kind,
                int_attr_of(attrs, "predicate").unwrap_or(0),
            );
            o.get_operation()
        }
        Opcode::Fma => {
            let ty = res_tys.first().cloned().expect("fma produces a value");
            FmaOp::new(ctx, ty, opds[0], opds[1], opds[2]).get_operation()
        }
        Opcode::Select => {
            let ty = res_tys.first().cloned().expect("select produces a value");
            SelectOp::new(ctx, ty, opds[0], opds[1], opds[2]).get_operation()
        }
        Opcode::Constant => {
            let ty = res_tys.first().cloned().expect("constant produces a value");
            ConstantOp::new(ctx, ty, &render_attrs_of(attrs)).get_operation()
        }
        Opcode::Broadcast => {
            let ty = res_tys
                .first()
                .cloned()
                .expect("broadcast produces a value");
            BroadcastOp::new(ctx, ty, opds[0]).get_operation()
        }
        Opcode::Reshape => {
            let ty = res_tys.first().cloned().expect("reshape produces a value");
            ReshapeOp::new(ctx, ty, opds[0]).get_operation()
        }
        Opcode::Extract => {
            let ty = res_tys.first().cloned().expect("extract produces a value");
            ExtractOp::new_with(ctx, ty, opds).get_operation()
        }
        Opcode::Offset => {
            let ty = res_tys.first().cloned().expect("offset produces a value");
            OffsetOp::new(ctx, ty, opds[0], opds[1]).get_operation()
        }
        Opcode::MmaF => {
            let ty = res_tys.first().cloned().expect("mma produces a value");
            MmaFOp::new(ctx, ty, opds[0], opds[1], opds[2]).get_operation()
        }
        Opcode::MakePartitionView => {
            let ty = res_tys
                .first()
                .cloned()
                .expect("make_partition_view produces a value");
            MakePartitionViewOp::new(ctx, ty, opds[0]).get_operation()
        }
        Opcode::Assume => {
            let ty = res_tys
                .first()
                .cloned()
                .expect("assume forwards a refined value");
            AssumeOp::new(ctx, ty, opds[0]).get_operation()
        }
        Opcode::Assert => AssertOp::new(ctx, opds[0]).get_operation(),
        Opcode::MakeToken => {
            let ty = res_tys
                .first()
                .cloned()
                .expect("make_token produces a value");
            MakeTokenOp::new(ctx, ty).get_operation()
        }
        Opcode::MakeTensorView => {
            let ty = res_tys
                .first()
                .cloned()
                .expect("make_tensor_view produces a value");
            MakeTensorViewOp::new_with(ctx, ty, opds).get_operation()
        }
        Opcode::LoadViewTko => {
            let tys = res_tys.to_vec();
            let (view, idx) = opds.split_first().expect("load has a view operand");
            LoadViewOp::new(ctx, tys, *view, idx.to_vec()).get_operation()
        }
        // Control flow. We build through `Operation::new` directly instead of through the constructors
        // because we want to preserve the regions etc that cutile already carries.
        Opcode::For | Opcode::If => {
            let tys = res_tys.to_vec();
            let info = if opcode == Opcode::For {
                ForOp::get_concrete_op_info()
            } else {
                IfOp::get_concrete_op_info()
            };
            Operation::new(ctx, info, tys, opds, vec![], n_regions)
        }
        Opcode::StoreViewTko => {
            // Original order (value, view, indices..., token) is kept.
            let tys = res_tys.to_vec();
            StoreViewOp::new(ctx, tys, opds[0], opds[1], opds[2..].to_vec()).get_operation()
        }
        Opcode::Return => ReturnOp::new(ctx).get_operation(),
        Opcode::Yield => YieldOp::new(ctx, opds).get_operation(),
        Opcode::Break => BreakOp::new(ctx).get_operation(),
        Opcode::Continue => ContinueOp::new(ctx, opds).get_operation(),
        other => {
            // Simple ops (op in -> results out).
            let tys = res_tys.to_vec();
            match simple_from_opcode(ctx, other, tys, opds, n_regions) {
                Some(o) => o,
                None => return Err(WaxBridgeErr::UnmappedOpcode(format!("{other:?}"))),
            }
        }
    };
    Ok(o)
}
