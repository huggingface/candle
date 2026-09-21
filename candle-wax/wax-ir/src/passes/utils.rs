//! Utils used by passes.
//! Op emission shorthand, type constructors, attribute encodings, etc.

use crate::attr::{Attribute, Bounded, DenseElements};
use crate::build::{BlockId, Module, OpId, Value};
use crate::builder::OpBuilder;
use crate::location::Location;
use crate::opcode::Opcode;
use crate::types::{
    FuncType, PaddingValue, PartitionViewType, PointerType, ScalarType, TensorViewType,
    TileElementType, TileType, Type,
};
use pliron::context::Context;
use pliron::irbuild::listener::DummyListener;

type Ins<'a> = &'a mut dyn pliron::irbuild::inserter::Inserter;

/// A function's entry block.
pub(crate) fn entry_block(ctx: &pliron::context::Context, func: OpId) -> Option<BlockId> {
    use pliron::linked_list::ContainsLinkedList;
    let r = func.deref(ctx).regions().next()?;
    r.deref(ctx).iter(ctx).next()
}

/// Copy a module into an owned one, for a pipeline that will rewrite it.
///
/// Not optional for a caller holding a cached module. A cached module is usually shared between
/// several realised kernel instantiations, for example a GEMM kernel is generic over the problem
/// shape (MxNxK), and the shape is not part of the cache key.
/// This means that if we rewrite the IR of one cached module this will corrupt the IR for all the
/// other kernel instantiations. Imagine a pass rewriting a GEMM module to GEMV in-place.
// TODO: Support multiple module entries
pub fn working_copy(module: &Module) -> Option<(Module, OpId)> {
    use pliron::builtin::op_interfaces::SingleBlockRegionInterface;
    use pliron::irbuild::cloning::{IrMapping, clone_operation_into_context};
    use pliron::irbuild::listener::DummyListener;
    use pliron::irbuild::rewriter::IRRewriter;

    let src = *module.entries().first()?;
    let mut copy = Module::new(module.name());
    let mut rewriter: IRRewriter<DummyListener> = IRRewriter::default();
    let func = clone_operation_into_context(
        src,
        &module.ctx,
        &mut copy.ctx,
        &mut rewriter,
        &mut IrMapping::default(),
    )
    .ok()?;

    // Add function to module
    copy.module_op().append_operation(&mut copy.ctx, func, 0);
    Some((copy, func))
}

/// A scalar tile type.
pub(crate) fn tile(shape: &[i64], elem: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(elem),
    })
}

/// Pointer to the provided scalar type as a tile.
pub(crate) fn ptr_tile(elem: &ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(
            PointerType {
                pointee: elem.clone(),
            }
            .into(),
        ),
    })
}

/// The scalar `i32` type.
pub(crate) fn i32() -> Type {
    Type::Scalar(ScalarType::I32)
}

/// Create partition view over `inner` with tile shape and padding.
pub(crate) fn pv(
    tile_shape: &[i32],
    inner: &TensorViewType,
    padding: Option<PaddingValue>,
) -> Type {
    Type::PartitionView(PartitionViewType {
        tile_shape: tile_shape.to_vec(),
        tensor_view: inner.clone(),
        dim_map: vec![0, 1],
        padding_value: padding,
    })
}

/// Set operand segment sizes on an op from the provided segments.
fn set_operand_segments(ctx: &pliron::context::Context, op: OpId, segments: &[&[Value]]) {
    let sizes: Vec<u32> = segments.iter().map(|s| s.len() as u32).collect();
    crate::builder::set_operand_segment_sizes(ctx, op, sizes);
}

/// Emit op using provided inserter.
pub(crate) fn emit_at(
    ctx: &mut pliron::context::Context,
    ins: Ins<'_>,
    op: Opcode,
    opds: &[Value],
    res: &[Type],
    attrs: &[(&str, Attribute)],
) -> pliron::result::Result<Vec<Value>> {
    let (id, vals) = OpBuilder::new(op, Location::Unknown)
        .operands(opds.iter().copied())
        .results(res.iter().cloned())
        .attrs(attrs.iter().map(|(k, v)| (k.to_string(), v.clone())))
        .build_in(ctx)?;
    ins.append_operation(ctx, id);
    Ok(vals)
}

/// [`emit_at`] for an op whose operands come in segments. Ref [`set_operand_segments`].
pub(crate) fn emit_at_segmented(
    ctx: &mut pliron::context::Context,
    ins: Ins<'_>,
    op: Opcode,
    segments: &[&[Value]],
    res: &[Type],
    attrs: &[(&str, Attribute)],
) -> pliron::result::Result<Vec<Value>> {
    let flat: Vec<Value> = segments.iter().flat_map(|s| s.iter().copied()).collect();
    let (id, vals) = OpBuilder::new(op, Location::Unknown)
        .operands(flat.iter().copied())
        .results(res.iter().cloned())
        .attrs(attrs.iter().map(|(k, v)| (k.to_string(), v.clone())))
        .build_in(ctx)?;
    set_operand_segments(ctx, id, segments);
    ins.append_operation(ctx, id);
    Ok(vals)
}

/// Assume bounded predicate.
pub(crate) fn bounded(lb: i64, ub: Option<i64>) -> Attribute {
    Attribute::Bounded(Bounded { lb: Some(lb), ub })
}

/// Create dense elements attribute from scalar type and raw bytes.
pub(crate) fn dens_scalar(elem: ScalarType, bytes: [u8; 4]) -> Attribute {
    Attribute::DenseElements(DenseElements {
        element_type: tile(&[], elem),
        shape: vec![],
        data: bytes.to_vec(),
    })
}

/// An i32 constant as dense element.
pub(crate) fn dens_i32(v: i64) -> Attribute {
    dens_scalar(ScalarType::I32, (v as u32).to_le_bytes())
}

/// At block end inserter.
pub(crate) fn at_end_of(block: BlockId) -> pliron::irbuild::inserter::IRInserter<DummyListener> {
    use pliron::irbuild::inserter::{IRInserter, OpInsertionPoint};
    IRInserter::new(OpInsertionPoint::AtBlockEnd(block))
}

/// Binary arithmetic
pub(crate) fn bin(
    ctx: &mut Context,
    ins: Ins<'_>,
    op: Opcode,
    a: Value,
    b: Value,
) -> pliron::result::Result<Value> {
    let ty =
        crate::dialect::source::value_type(ctx, a).expect("an arithmetic operand has a wax type");
    Ok(emit_at(ctx, ins, op, &[a, b], &[ty], &[])?[0])
}

/// A `wax.load_view`. Tile read from `view` at `[i0, i1]`, ordered by `tok`.
///
/// Operand segments are `(view, indices, token)` and the results `(tile, token)`.
// TODO: Return token as well.
pub(crate) fn load_view(
    ctx: &mut Context,
    ins: Ins<'_>,
    view: Value,
    i0: Value,
    i1: Value,
    tok: Value,
    tile_ty: &Type,
) -> pliron::result::Result<Value> {
    let attrs = [("memory_ordering_semantics", Attribute::i32(0))];
    Ok(emit_at_segmented(
        ctx,
        ins,
        Opcode::LoadViewTko,
        &[&[view], &[i0, i1], &[tok]],
        &[tile_ty.clone(), Type::Token],
        &attrs,
    )?[0])
}

/// A `wax.store_view`. `value` stored in `view` at `[i0, i1]`, ordered by `tok`.
///
/// Operand segments are `(value, view, indices, token)`.
// TODO: Use token (currently dropped).
pub(crate) fn store_view(
    ctx: &mut Context,
    ins: Ins<'_>,
    value: Value,
    view: Value,
    i0: Value,
    i1: Value,
    tok: Value,
) -> pliron::result::Result<()> {
    let attrs = [("memory_ordering_semantics", Attribute::i32(0))];
    emit_at_segmented(
        ctx,
        ins,
        Opcode::StoreViewTko,
        &[&[value], &[view], &[i0, i1], &[tok]],
        &[Type::Token],
        &attrs,
    )?;
    Ok(())
}

/// Floating-point binary arithmetic.
///
/// Float binary uses `rounding_mode`, hence separate fn from [`bin`].
/// Currently `rounding_mode` is always set to `0`.
// TODO: Investigate if support for alternative rounding modes is warranted.
pub(crate) fn binf(
    ctx: &mut Context,
    ins: Ins<'_>,
    op: Opcode,
    a: Value,
    b: Value,
) -> pliron::result::Result<Value> {
    let ty = crate::dialect::source::value_type(ctx, a).expect("a float operand has a wax type");
    let attrs = [("rounding_mode", Attribute::i32(0))];
    Ok(emit_at(ctx, ins, op, &[a, b], &[ty], &attrs)?[0])
}

/// Terminator. `wax.continue`/`wax.yield`/`wax.return`. No results.
pub(crate) fn terminator(
    ctx: &mut Context,
    ins: Ins<'_>,
    op: Opcode,
    opds: &[Value],
) -> pliron::result::Result<()> {
    emit_at(ctx, ins, op, opds, &[], &[])?;
    Ok(())
}

/// `wax.make_tensor_view`.
///
/// Operand segments are `(base, dynamic shape, dynamic strides)`.
/// Both shape and strides are empty here. They arrive at runtime as promoted parameters.
pub(crate) fn tensor_view(
    ctx: &mut Context,
    ins: Ins<'_>,
    base: Value,
    tv: &TensorViewType,
) -> pliron::result::Result<Value> {
    Ok(emit_at_segmented(
        ctx,
        ins,
        Opcode::MakeTensorView,
        &[&[base], &[], &[]],
        &[Type::TensorView(tv.clone())],
        &[],
    )?[0])
}

/// `wax.make_partition_view`.
pub(crate) fn partition_view(
    ctx: &mut Context,
    ins: Ins<'_>,
    view: Value,
    shape: &[i32],
    tv: &TensorViewType,
    pad: Option<PaddingValue>,
) -> pliron::result::Result<Value> {
    Ok(emit_at(
        ctx,
        ins,
        Opcode::MakePartitionView,
        &[view],
        &[pv(shape, tv, pad)],
        &[],
    )?[0])
}

/// `wax.ftof`. Converting `v` to `to`. Widen on load and narrow on store.
pub(crate) fn convert(
    ctx: &mut Context,
    ins: Ins<'_>,
    v: Value,
    to: &Type,
) -> pliron::result::Result<Value> {
    Ok(emit_at(ctx, ins, Opcode::FToF, &[v], std::slice::from_ref(to), &[])?[0])
}

/// i32 `wax.constant`.
pub(crate) fn const_i32(ctx: &mut Context, ins: Ins<'_>, v: i64) -> pliron::result::Result<Value> {
    Ok(emit_at(
        ctx,
        ins,
        Opcode::Constant,
        &[],
        &[i32()],
        &[("value", dens_i32(v))],
    )?[0])
}

pub(crate) fn const_scalar(
    ctx: &mut Context,
    ins: Ins<'_>,
    ty: &Type,
    elem: ScalarType,
    bytes: [u8; 4],
) -> pliron::result::Result<Value> {
    let attrs = [("value", dens_scalar(elem, bytes))];
    Ok(emit_at(
        ctx,
        ins,
        Opcode::Constant,
        &[],
        std::slice::from_ref(ty),
        &attrs,
    )?[0])
}

/// `wax.assume` that `v` is within provided bound.
pub(crate) fn assume_bounded(
    ctx: &mut Context,
    ins: Ins<'_>,
    v: Value,
    lb: i64,
    ub: Option<i64>,
) -> pliron::result::Result<Value> {
    let ty = crate::dialect::source::value_type(ctx, v).expect("an assume operand has a wax type");
    Ok(emit_at(
        ctx,
        ins,
        Opcode::Assume,
        &[v],
        &[ty],
        &[("predicate", bounded(lb, ub))],
    )?[0])
}

/// `wax.make_token`.
pub(crate) fn make_token(ctx: &mut Context, ins: Ins<'_>) -> pliron::result::Result<Value> {
    Ok(emit_at(ctx, ins, Opcode::MakeToken, &[], &[Type::Token], &[])?[0])
}

/// `wax.get_tile_block_id`. Returns (x,y,z).
pub(crate) fn tile_block_id(ctx: &mut Context, ins: Ins<'_>) -> pliron::result::Result<[Value; 3]> {
    let v = emit_at(
        ctx,
        ins,
        Opcode::GetTileBlockId,
        &[],
        &[i32(), i32(), i32()],
        &[],
    )?;
    Ok([v[0], v[1], v[2]])
}

/// Stamp function type attribute on op
pub(crate) fn stamp_in(ctx: &pliron::context::Context, f: OpId, params: &[Type]) {
    crate::builder::set_op_attr_in(
        ctx,
        f,
        &crate::build::ATTR_FUNCTION_TYPE,
        Attribute::Type(Type::Func(FuncType {
            inputs: params.to_vec(),
            results: vec![],
        })),
    );
}

/// Read wax attribute from op by key.
pub fn op_attr_in(
    ctx: &pliron::context::Context,
    op: crate::build::OpId,
    key: &pliron::identifier::Identifier,
) -> Option<crate::attr::Attribute> {
    use crate::dialect::attr_mirror::WaxAttr;
    op.deref(ctx)
        .attributes
        .get::<WaxAttr>(key)
        .map(|a| a.0.clone())
}
