//! Small shared utilities
use super::*;
use crate::air_dialect::attrs::BinKind;
use crate::air_dialect::ops as air;
use crate::air_dialect::types as t;
use pliron::identifier::Identifier;
use pliron::value::{DefiningEntity, Value};
use wax_ir::dialect::attr_mirror::WaxAttr;

/// The `air` type a wax tile value lowers to.
pub fn air_type_for(ctx: &Context, ty: &WaxType) -> Option<TypeHandle> {
    let scalar = |s: &ScalarType| crate::types::air_scalar(ctx, s);
    match ty {
        WaxType::Scalar(s) => scalar(s),
        WaxType::Tile(t) if matches!(t.element_type, TileElementType::Pointer(_)) => {
            let TileElementType::Pointer(p) = &t.element_type else {
                return None;
            };
            Some(t::device_ptr_ty(ctx, scalar(&p.pointee)?))
        }
        WaxType::Pointer(p) => Some(t::device_ptr_ty(ctx, scalar(&p.pointee)?)),
        WaxType::Tile(t) => {
            let TileElementType::Scalar(e) = &t.element_type else {
                return None;
            };
            scalar(e)
        }
        _ => None,
    }
}

/// Attempt to extract i64 value from `wax.constant`
pub(crate) fn wax_const_i64(ctx: &Context, v: Value) -> Option<i64> {
    use pliron::r#type::Typed;
    use wax_ir::dialect::ops::ConstantOp;
    let def = v.defining_op()?;
    if Operation::get_opid(def, ctx) != ConstantOp::get_opid_static() {
        return None;
    }
    let (ty, de) = {
        let o = def.deref(ctx);
        if o.get_num_results() != 1 {
            return None;
        }
        let ty = o.get_result(0).get_type(ctx);
        let de = o
            .attributes
            .get::<WaxAttr>(&keys::ATTR_VALUE)
            .and_then(|a| {
                let wax_ir::Attribute::DenseElements(de) = &a.0 else {
                    return None;
                };
                Some(de.clone())
            });
        (ty, de)
    };
    let wt = wax_type_of(ctx, ty)?;
    match crate::lower::dense_elements_value(&de?, &wt) {
        crate::lower::ImmVal::Int(i) => Some(i),
        crate::lower::ImmVal::Zero => Some(0),
        _ => None,
    }
}

/// Op defining `v`, if present. Block argument yields `None`.
pub(crate) fn def_op(v: Value) -> Option<Ptr<Operation>> {
    match v.defining_entity() {
        DefiningEntity::Op(o) => Some(o),
        _ => None,
    }
}

/// Op defining `op`'s first operand. One link up a view or wrapper chain.
pub(crate) fn first_operand_def(ctx: &Context, op: Ptr<Operation>) -> Option<Ptr<Operation>> {
    def_op(op.deref(ctx).operands().next()?)
}

/// Walk a view value back to the `make_tensor_view` it was built from, through the strided-view
/// and `assume` wrappers and the partition. `None` when the chain is not that shape.
pub(crate) fn tensor_view_def(ctx: &Context, view: Value) -> Option<Ptr<Operation>> {
    let mut cur = def_op(view)?;
    for _ in 0..4 {
        match try_opcode_of(ctx, cur) {
            Some(Opcode::MakeStridedView | Opcode::Assume) => cur = first_operand_def(ctx, cur)?,
            _ => break,
        }
    }
    if try_opcode_of(ctx, cur) != Some(Opcode::MakePartitionView) {
        return None;
    }
    let tv_op = first_operand_def(ctx, cur)?;
    (try_opcode_of(ctx, tv_op) == Some(Opcode::MakeTensorView)).then_some(tv_op)
}

/// Insert an operation at the current point and read its first result. The emit-then-read
/// shape every arm repeats; `emit_op` is the same for an emitter that hands back a typed op.
pub(crate) fn emit(ctx: &Context, rw: &mut DialectConversionRewriter, op: Ptr<Operation>) -> Value {
    rw.insert_operation(ctx, op);
    op.deref(ctx).get_result(0)
}

pub(crate) fn emit_op(ctx: &Context, rw: &mut DialectConversionRewriter, op: impl Op) -> Value {
    emit(ctx, rw, op.get_operation())
}

/// An i32 binop at the current insertion point.
pub(crate) fn air_bin_i32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    a: Value,
    b: Value,
    k: BinKind,
) -> Value {
    let i32_ty = t::i32_ty(ctx);
    let o = air::bin(ctx, k, i32_ty, a, b);
    emit(ctx, rw, o)
}

/// `@air.workgroup_barrier(2, 1)`: the two constants are pool entries, only the call is an
/// instruction.
pub(crate) fn air_wg_barrier(ctx: &mut Context, rw: &mut DialectConversionRewriter) {
    let two = air_const_i32(ctx, 2);
    let one = air_const_i32(ctx, 1);
    let b = air::call(ctx, crate::lower::AIR_WG_BARRIER, None, vec![two, one]);
    rw.insert_operation(ctx, b);
}

/// Call `f` with each block's op list, depth first.
pub(crate) fn for_each_block(
    ctx: &Context,
    op: Ptr<Operation>,
    f: &mut impl FnMut(&[Ptr<Operation>]),
) {
    use pliron::linked_list::ContainsLinkedList;
    for r in op.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            let ops: Vec<Ptr<Operation>> = b.deref(ctx).iter(ctx).collect();
            f(&ops);
            for &inner in &ops {
                for_each_block(ctx, inner, f);
            }
        }
    }
}

pub(crate) fn why(reason: &str) {
    if std::env::var_os("WAX_CONVERSION_WHY").is_some() {
        eprintln!("[wax] {reason}");
    }
}

/// `op`'s result types, in order. The list an arm pads its `undef` tokens against.
pub(crate) fn op_result_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    use pliron::r#type::Typed;
    let o = op.deref(ctx);
    (0..o.get_num_results())
        .map(|i| o.get_result(i).get_type(ctx))
        .collect()
}

/// A named wax integer attribute, whether it is stored as an integer or a bool.
pub(crate) fn wax_int_attr(ctx: &Context, op: Ptr<Operation>, key: &Identifier) -> Option<i64> {
    use wax_ir::attr::Attribute as WA;
    match &op.deref(ctx).attributes.get::<WaxAttr>(key)?.0 {
        WA::Integer(i, _) => Some(*i),
        WA::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}
