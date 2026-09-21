//! Small shared utilities: the type mapping, constant and binop emitters, and the
//! opcode-reading helpers the arms use.

use super::*;

/// The `air` type a wax tile VALUE lowers to.
///
/// Verified against emitted AIR, not inferred: a `[1,32] f32` tile loads as `float`, one scalar
/// per lane, because a tile is distributed across the threadgroup rather than materialised
/// whole. `types::ll_type`'s `[N x T]` spelling is the kernel-PARAMETER ABI and is unreachable
/// for values - which is why `LlType` has no array variant at all.
///
/// The mapping is nonetheless a pure function of the wax type, which is what makes it
/// expressible as pliron's `convert_type`:
///
/// - a rank-0 tile is already a register value -> its element type
/// - an 8x8 tile is one SGMMA fragment -> `<64 x float>`
/// - any other tile is distributed -> its element type, one lane's worth
///
/// What is NOT expressible there is how many lanes participate or how the address is formed.
/// That is the op expansion, and it is why this prototype converts no tile-shaped op.
pub fn air_type_for(ty: &WaxType) -> Option<LlType> {
    let scalar = |s: &ScalarType| -> Option<LlType> {
        Some(match s {
            ScalarType::I1 => LlType::I1,
            ScalarType::I8 => LlType::I8,
            ScalarType::I16 => LlType::I16,
            ScalarType::I32 => LlType::I32,
            ScalarType::I64 => LlType::I64,
            ScalarType::F16 => LlType::Half,
            ScalarType::BF16 => LlType::BFloat,
            ScalarType::F32 => LlType::Float,
            ScalarType::F64 => LlType::Double,
            _ => return None,
        })
    };
    match ty {
        WaxType::Scalar(s) => scalar(s),
        // A pointer-element tile is a buffer parameter: it lowers to the device pointer it
        // holds. `const_params` can move one to addrspace(2); that is a compile option the
        // conversion does not see, so only the default is handled here.
        WaxType::Tile(t) if matches!(t.element_type, TileElementType::Pointer(_)) => {
            let TileElementType::Pointer(p) = &t.element_type else {
                return None;
            };
            Some(LlType::device_ptr(scalar(&p.pointee)?))
        }
        // A bare pointer value (a raw pointer parameter): the device pointer itself.
        WaxType::Pointer(p) => Some(LlType::device_ptr(scalar(&p.pointee)?)),
        WaxType::Tile(t) => {
            let TileElementType::Scalar(e) = &t.element_type else {
                return None;
            };
            // NO shape-based fragment guess. `passes::fragment_set` exists precisely because
            // fragment-ness is not a property of the shape: `[32, 32]` is equally a
            // threadgroup-spanning tile with one element per thread, and a fragment is
            // what an MMA touches. This conversion refuses `MmaF` and the `sgmma_*` ops,
            // which are the only seeds of that analysis, so in any kernel it converts
            // nothing is a fragment and every tile is distributed - one lane's element.
            // Guessing `<64 x float>` from the shape loaded a whole fragment where the
            // backend loads a scalar.
            scalar(e)
        }
        _ => None,
    }
}

/// The integer value of a value defined by a `wax.constant`, reading the same dense-elements
/// attribute the CFG decodes, so the two routes see the same constants.
pub(crate) fn wax_const_i64(ctx: &Context, v: pliron::value::Value) -> Option<i64> {
    use pliron::r#type::Typed;
    let def = v.defining_op()?;
    if Operation::get_op_dyn(def, ctx)
        .op_ref()
        .get_opid()
        .to_string()
        != "wax.constant"
    {
        return None;
    }
    let (ty, de) = {
        let o = def.deref(ctx);
        if o.get_num_results() != 1 {
            return None;
        }
        let ty = o.get_result(0).get_type(ctx);
        let de = o.attributes.0.iter().find_map(|(k, a)| {
            let a = a.downcast_ref::<wax_ir::dialect::attr_mirror::WaxAttr>()?;
            if k.to_string() != "value" {
                return None;
            }
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

/// An i32 binop at the current insertion point.
pub(crate) fn air_bin_i32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    i32_ty: TypeHandle,
    a: pliron::value::Value,
    b: pliron::value::Value,
    k: crate::air_dialect::attrs::BinOpAttr,
) -> pliron::value::Value {
    let o = air::BinOp::new(ctx, i32_ty, a, b);
    o.set_op(ctx, k);
    rw.insert_operation(ctx, o.get_operation());
    o.get_operation().deref(ctx).get_result(0)
}

/// `@air.workgroup_barrier(2, 1)`: the two constants are pool entries, only the call is an
/// instruction.
pub(crate) fn air_wg_barrier(ctx: &mut Context, rw: &mut DialectConversionRewriter, i32_ty: TypeHandle) {
    let two = air_const_i32(ctx, i32_ty, 2);
    let one = air_const_i32(ctx, i32_ty, 1);
    let b = air::CallOp::new(ctx, crate::lower::AIR_WG_BARRIER, None, vec![two, one]);
    rw.insert_operation(ctx, b.get_operation());
}

/// Call `f` with each block's op list, depth first.
pub(crate) fn for_each_block(ctx: &Context, op: Ptr<Operation>, f: &mut impl FnMut(&[Ptr<Operation>])) {
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

/// Reason a restriction fired, for the corpus driver. Env-gated: the point is to turn "this
/// kernel did not convert" into "this is the one restriction stopping it".
pub(crate) fn why(reason: &str) {
    if std::env::var_os("WAX_CONVERSION_WHY").is_some() {
        eprintln!("[conv-why] {reason}");
    }
}

/// The SSA value carrying dim `d`'s runtime extent, if it was promoted to a parameter.
///
/// Walks the view chain the way `find_shape_val_for_pv` does: through any strided-view or
/// `assume` wrapper to the partition, then to its tensor view, then into the `dyn_shape`
/// operand group indexed by how many dynamic dims precede this one.
///
/// `op`'s result types, in order - the list an arm pads its `undef` tokens against.
///
/// `op`'s result types, in order - the list an arm pads its `undef` tokens against.
pub(crate) fn op_result_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    use pliron::r#type::Typed;
    let o = op.deref(ctx);
    (0..o.get_num_results())
        .map(|i| o.get_result(i).get_type(ctx))
        .collect()
}

