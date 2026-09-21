use wax_ir::builder::{OpBuilder, append_op};
use wax_ir::{Attribute, Location, Module, Opcode, ScalarType, TileElementType, TileType, Type};

fn scalar(e: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(e),
    })
}

fn binary_module(name: &str, code: Opcode, e: ScalarType, signed: Option<i64>) -> Module {
    let mut m = Module::new(name);
    let ty = scalar(e);
    let (entry, block, args) = m
        .begin_function(name, &[ty.clone(), ty.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let mut b = OpBuilder::new(code, Location::Unknown)
        .operands([args[0], args[1]])
        .result(ty);
    if let Some(s) = signed {
        b = b.attr("signedness", Attribute::i32(s));
    }
    let (id, _) = b.build(&mut m).expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn air_op_names(m: &mut Module) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let mut out = Vec::new();
    let ctx = &m.ctx;
    for r in func.deref(ctx).regions() {
        for blk in r.deref(ctx).iter(ctx) {
            for op in blk.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                out.push(
                    pliron::operation::Operation::get_op_dyn(op, ctx)
                        .op_ref()
                        .get_opid()
                        .to_string(),
                );
            }
        }
    }
    out
}

#[track_caller]
fn assert_lowers_to(mut m: Module, expected: &str) {
    let names = air_op_names(&mut m);
    assert!(
        names.iter().any(|n| n == expected),
        "expected `{expected}`, got {names:?}"
    );
}

#[test]
fn xor_lowers_to_llvm_xor() {
    assert_lowers_to(
        binary_module("xor_i32", Opcode::XOrI, ScalarType::I32, None),
        "llvm.xor",
    );
}

#[test]
fn shift_left_lowers_to_llvm_shl() {
    assert_lowers_to(
        binary_module("shl_i32", Opcode::ShLI, ScalarType::I32, None),
        "llvm.shl",
    );
}

#[test]
fn signed_right_shift_is_arithmetic_and_unsigned_is_logical() {
    assert_lowers_to(
        binary_module("ashr_i32", Opcode::ShRI, ScalarType::I32, Some(1)),
        "llvm.ashr",
    );
    assert_lowers_to(
        binary_module("lshr_i32", Opcode::ShRI, ScalarType::I32, Some(0)),
        "llvm.lshr",
    );
}

fn cast_module(
    name: &str,
    code: Opcode,
    a: ScalarType,
    r: ScalarType,
    signed: Option<i64>,
) -> Module {
    let mut m = Module::new(name);
    let at = scalar(a.clone());
    let (entry, block, args) = m
        .begin_function(name, &[at.clone(), at.clone()])
        .expect("builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let add_code = if a.is_float() {
        Opcode::AddF
    } else {
        Opcode::AddI
    };
    let (add, add_res) = OpBuilder::new(add_code, Location::Unknown)
        .operands([args[0], args[1]])
        .result(at)
        .build(&mut m)
        .expect("add builds");
    append_op(&mut m, block, add);
    let mut b = OpBuilder::new(code, Location::Unknown)
        .operands([add_res[0]])
        .result(scalar(r));
    if let Some(s) = signed {
        b = b.attr("signedness", Attribute::i32(s));
    }
    let (id, _) = b.build(&mut m).expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn unary_module(
    name: &str,
    code: Opcode,
    a: ScalarType,
    r: ScalarType,
    signed: Option<i64>,
) -> Module {
    let mut m = Module::new(name);
    let (entry, block, args) = m.begin_function(name, &[scalar(a)]).expect("builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let mut b = OpBuilder::new(code, Location::Unknown)
        .operands([args[0]])
        .result(scalar(r));
    if let Some(s) = signed {
        b = b.attr("signedness", Attribute::i32(s));
    }
    let (id, _) = b.build(&mut m).expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

#[test]
fn sign_extend() {
    assert_lowers_to(
        cast_module(
            "sext",
            Opcode::ExtI,
            ScalarType::I32,
            ScalarType::I64,
            Some(1),
        ),
        "llvm.sext",
    );
}

#[test]
fn zero_extend() {
    assert_lowers_to(
        cast_module(
            "zext",
            Opcode::ExtI,
            ScalarType::I32,
            ScalarType::I64,
            Some(0),
        ),
        "llvm.zext",
    );
}

#[test]
fn truncate() {
    assert_lowers_to(
        cast_module(
            "trunc",
            Opcode::TruncI,
            ScalarType::I64,
            ScalarType::I32,
            None,
        ),
        "llvm.trunc",
    );
}

#[test]
fn bitcast_between_same_width_types() {
    assert_lowers_to(
        cast_module(
            "bitcast",
            Opcode::Bitcast,
            ScalarType::I32,
            ScalarType::F32,
            None,
        ),
        "llvm.bitcast",
    );
}

#[test]
fn integer_negate() {
    assert_lowers_to(
        unary_module("negi", Opcode::NegI, ScalarType::I32, ScalarType::I32, None),
        "llvm.sub",
    );
}

#[test]
fn integer_max_is_a_compare_and_select() {
    let names = air_op_names(&mut binary_module(
        "maxi",
        Opcode::MaxI,
        ScalarType::I32,
        Some(1),
    ));
    assert!(names.iter().any(|n| n == "llvm.icmp"), "{names:?}");
    assert!(names.iter().any(|n| n == "llvm.select"), "{names:?}");
}

#[test]
fn float_abs_and_fma() {
    assert_lowers_to(
        unary_module("absf", Opcode::AbsF, ScalarType::F32, ScalarType::F32, None),
        "llvm.call_intrinsic",
    );
}

fn icmp_predicates(m: &mut Module) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                let dynop = pliron::operation::Operation::get_op_dyn(op, ctx);
                if dynop.op_ref().get_opid().to_string() != "llvm.icmp" {
                    continue;
                }
                let Some(icmp) = pliron_llvm::ops::ICmpOp::wrap_operation(op)
                    .downcast::<pliron_llvm::ops::ICmpOp>()
                else {
                    continue;
                };
                out.push(format!("{:?}", icmp.predicate(ctx)));
            }
        }
    }
    out
}

#[test]
fn integer_max_signedness_rides_the_compare_predicate() {
    let signed = icmp_predicates(&mut binary_module(
        "smax",
        Opcode::MaxI,
        ScalarType::I32,
        Some(1),
    ));
    assert!(
        signed
            .iter()
            .any(|p| p.contains("SGT") || p.contains("SGE")),
        "signed max should compare signed, got {signed:?}"
    );
    let unsigned = icmp_predicates(&mut binary_module(
        "umax",
        Opcode::MaxI,
        ScalarType::I32,
        Some(0),
    ));
    assert!(
        unsigned
            .iter()
            .any(|p| p.contains("UGT") || p.contains("UGE")),
        "unsigned max should compare unsigned, got {unsigned:?}"
    );
}

fn ternary_module(name: &str, code: Opcode, e: ScalarType) -> Module {
    let mut m = Module::new(name);
    let ty = scalar(e);
    let (entry, block, args) = m
        .begin_function(name, &[ty.clone(), ty.clone(), ty.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let (id, _) = OpBuilder::new(code, Location::Unknown)
        .operands([args[0], args[1], args[2]])
        .result(ty)
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

#[test]
fn fma_lowers_to_a_multiply_and_an_add() {
    let names = air_op_names(&mut ternary_module("fma", Opcode::Fma, ScalarType::F32));
    assert!(names.iter().any(|n| n == "llvm.fmul"), "{names:?}");
    assert!(names.iter().any(|n| n == "llvm.fadd"), "{names:?}");
}

#[test]
fn integer_arithmetic_and_bitwise() {
    let i = ScalarType::I32;
    for (code, want) in [
        (Opcode::AndI, "llvm.and"),
        (Opcode::OrI, "llvm.or"),
        (Opcode::AddI, "llvm.add"),
        (Opcode::SubI, "llvm.sub"),
        (Opcode::MulI, "llvm.mul"),
    ] {
        assert_lowers_to(binary_module("k", code, i.clone(), None), want);
    }
}

#[test]
fn integer_divide_and_remainder_honour_signedness() {
    let i = ScalarType::I32;
    for (code, signed, want) in [
        (Opcode::DivI, 1, "llvm.sdiv"),
        (Opcode::DivI, 0, "llvm.udiv"),
        (Opcode::RemI, 1, "llvm.srem"),
        (Opcode::RemI, 0, "llvm.urem"),
    ] {
        assert_lowers_to(binary_module("k", code, i.clone(), Some(signed)), want);
    }
}

#[test]
fn float_arithmetic() {
    let f = ScalarType::F32;
    for (code, want) in [
        (Opcode::AddF, "llvm.fadd"),
        (Opcode::SubF, "llvm.fsub"),
        (Opcode::MulF, "llvm.fmul"),
        (Opcode::DivF, "llvm.fdiv"),
    ] {
        assert_lowers_to(binary_module("k", code, f.clone(), None), want);
    }
}

fn intrinsic_names(m: &mut Module) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                let Some(c) = wax_metal::air_dialect::ops::CallIntrinsicOp::wrap_operation(op)
                    .downcast::<wax_metal::air_dialect::ops::CallIntrinsicOp>()
                else {
                    continue;
                };
                if let Some(n) = c.get_attr_llvm_intrinsic_name(ctx) {
                    out.push(n.clone().into());
                }
            }
        }
    }
    out
}

#[track_caller]
fn assert_calls(mut m: Module, expected: &str) {
    let got = intrinsic_names(&mut m);
    assert!(
        got.iter().any(|n| n.starts_with(expected)),
        "expected a call to `{expected}`, got {got:?}"
    );
}

#[test]
fn float_intrinsics_calls_correctly() {
    let f = ScalarType::F32;
    for (code, expected) in [
        (Opcode::Sin, "@air.sin"),
        (Opcode::Cos, "@air.cos"),
        (Opcode::Sqrt, "@air.sqrt"),
        (Opcode::Log, "@air.log"),
        (Opcode::Log2, "@air.log2"),
        (Opcode::Exp2, "@air.exp2"),
        (Opcode::Exp, "@air.fast_exp"),
        (Opcode::AbsF, "@air.fabs"),
    ] {
        assert_calls(
            unary_module("k", code, f.clone(), f.clone(), None),
            expected,
        );
    }
    assert_calls(binary_module("k", Opcode::MaxF, f, None), "@air.fast_fmax");
}

fn cmp_module(name: &str, code: Opcode, e: ScalarType, pred: i64, signed: i64) -> Module {
    let mut m = Module::new(name);
    let ty = scalar(e);
    let (entry, block, args) = m
        .begin_function(name, &[ty.clone(), ty.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let (id, _) = OpBuilder::new(code, Location::Unknown)
        .operands([args[0], args[1]])
        .result(scalar(ScalarType::I1))
        .attr("comparison_predicate", Attribute::i32(pred))
        .attr("signedness", Attribute::i32(signed))
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

#[test]
fn integer_compare_maps_predicate_and_signedness() {
    for (pred, signed, expected) in [
        (0, 1, "EQ"),
        (1, 1, "NE"),
        (2, 1, "SLT"),
        (2, 0, "ULT"),
        (4, 1, "SGT"),
        (4, 0, "UGT"),
    ] {
        let result = icmp_predicates(&mut cmp_module(
            "k",
            Opcode::CmpI,
            ScalarType::I32,
            pred,
            signed,
        ));
        assert!(
            result.iter().any(|p| p == expected),
            "predicate {pred}/signed {signed}: expected {expected}, got {result:?}"
        );
    }
}

#[test]
fn numeric_conversions_between_int_and_float() {
    for (code, from, to, expected) in [
        (
            Opcode::IToF,
            ScalarType::I32,
            ScalarType::F32,
            "llvm.sitofp",
        ),
        (
            Opcode::FToI,
            ScalarType::F32,
            ScalarType::I32,
            "llvm.fptosi",
        ),
    ] {
        assert_lowers_to(cast_module("k", code, from, to, Some(1)), expected);
    }
}

#[test]
fn integer_min_is_a_compare_and_select() {
    let names = air_op_names(&mut binary_module(
        "mini",
        Opcode::MinI,
        ScalarType::I32,
        Some(1),
    ));
    assert!(names.iter().any(|n| n == "llvm.icmp"), "{names:?}");
    assert!(names.iter().any(|n| n == "llvm.select"), "{names:?}");
}

#[test]
fn float_min_is_a_compare_and_select() {
    let names = air_op_names(&mut binary_module(
        "minf",
        Opcode::MinF,
        ScalarType::F32,
        None,
    ));
    assert!(names.iter().any(|n| n == "llvm.fcmp"), "{names:?}");
    assert!(names.iter().any(|n| n == "llvm.select"), "{names:?}");
}

fn fcmp_predicates(m: &mut Module) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                let dynop = pliron::operation::Operation::get_op_dyn(op, ctx);
                if dynop.op_ref().get_opid().to_string() != "llvm.fcmp" {
                    continue;
                }
                let Some(c) = pliron_llvm::ops::FCmpOp::wrap_operation(op)
                    .downcast::<pliron_llvm::ops::FCmpOp>()
                else {
                    continue;
                };
                out.push(format!("{:?}", c.predicate(ctx)));
            }
        }
    }
    out
}

#[test]
fn float_compare_maps_its_predicate() {
    for (pred, expected) in [(0, "OEQ"), (1, "ONE"), (2, "OLT"), (4, "OGT")] {
        let result = fcmp_predicates(&mut cmp_module("k", Opcode::CmpF, ScalarType::F32, pred, 1));
        assert!(
            result.iter().any(|p| p == expected),
            "predicate {pred}: expected {expected}, got {result:?}"
        );
    }
}

#[test]
fn float_to_float_narrows_and_widens_by_width() {
    assert_lowers_to(
        cast_module(
            "narrow",
            Opcode::FToF,
            ScalarType::F32,
            ScalarType::F16,
            None,
        ),
        "llvm.fptrunc",
    );
    assert_lowers_to(
        cast_module(
            "widen",
            Opcode::FToF,
            ScalarType::F16,
            ScalarType::F32,
            None,
        ),
        "llvm.fpext",
    );
}

#[test]
fn rsqrt_is_a_sqrt_and_a_divide() {
    let mut m = unary_module("rs", Opcode::Rsqrt, ScalarType::F32, ScalarType::F32, None);
    let names = air_op_names(&mut m);
    assert!(names.iter().any(|n| n == "llvm.fdiv"), "{names:?}");
    let mut m = unary_module("rs", Opcode::Rsqrt, ScalarType::F32, ScalarType::F32, None);
    assert!(
        intrinsic_names(&mut m)
            .iter()
            .any(|n| n.starts_with("@air.sqrt")),
        "expected a sqrt call"
    );
}

#[test]
fn select_lowers_to_llvm_select() {
    let mut m = Module::new("sel");
    let f = scalar(ScalarType::F32);
    let i1 = scalar(ScalarType::I1);
    let (entry, block, args) = m
        .begin_function("sel", &[i1, f.clone(), f.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String("sel".to_string()),
    );
    let (id, _) = OpBuilder::new(Opcode::Select, Location::Unknown)
        .operands([args[0], args[1], args[2]])
        .result(f)
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    assert_lowers_to(m, "llvm.select");
}

#[test]
fn asserts_and_tokens_are_erased() {
    let mut m = Module::new("tok");
    let (entry, block, _args) = m.begin_function("tok", &[]).expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String("tok".to_string()),
    );
    let (tk, _) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
        .result(Type::Token)
        .build(&mut m)
        .expect("token builds");
    append_op(&mut m, block, tk);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    let names = air_op_names(&mut m);
    assert!(
        !names.iter().any(|n| n.contains("token")),
        "a token survived into the emitted air: {names:?}"
    );
    assert_eq!(
        names,
        vec!["llvm.return".to_string()],
        "a token with no consumer should leave nothing behind"
    );
}

fn rmw_module(name: &str, elem: ScalarType, mode: i64) -> Module {
    use wax_ir::PointerType;
    let mut m = Module::new(name);
    let ptr = Type::Pointer(PointerType {
        pointee: elem.clone(),
    });
    let ty = scalar(elem);
    let (entry, block, args) = m
        .begin_function(name, &[ptr, ty.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let (id, _) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
        .operands([args[0], args[1]])
        .result(ty)
        .result(Type::Token)
        .attr("mode", Attribute::i32(mode))
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn cas_module(name: &str, elem: ScalarType) -> Module {
    use wax_ir::PointerType;
    let mut m = Module::new(name);
    let ptr = Type::Pointer(PointerType {
        pointee: elem.clone(),
    });
    let ty = scalar(elem);
    let (entry, block, args) = m
        .begin_function(name, &[ptr, ty.clone(), ty.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let (id, _) = OpBuilder::new(Opcode::AtomicCAS, Location::Unknown)
        .operands([args[0], args[1], args[2]])
        .result(ty)
        .result(Type::Token)
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn scan_module(name: &str, n: i64, reverse: bool) -> Module {
    use wax_ir::DenseElements;
    let mut m = Module::new(name);
    let tile = Type::Tile(TileType {
        shape: vec![1, n],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let (entry, block, _) = m.begin_function(name, &[]).expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let (cid, cres) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(tile.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: Type::Scalar(ScalarType::F32),
                shape: vec![1, n],
                data: vec![0u8; n as usize * 4],
            }),
        )
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, cid);
    let scal = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let (region, body, bargs) =
        wax_ir::builder::build_single_block_region(&mut m, &[scal.clone(), scal.clone()])
            .expect("body region builds");
    let (add, add_res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
        .operands([bargs[0], bargs[1]])
        .result(scal.clone())
        .build(&mut m)
        .expect("combine builds");
    append_op(&mut m, body, add);
    let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
        .operands([add_res[0]])
        .build(&mut m)
        .expect("yield builds");
    append_op(&mut m, body, y);
    let mut b = OpBuilder::new(Opcode::Scan, Location::Unknown)
        .operands([cres[0]])
        .result(tile)
        .region(region);
    if reverse {
        b = b.attr("reverse", Attribute::Bool(true));
    }
    let (id, _) = b.build(&mut m).expect("op builds");
    append_op(&mut m, block, id);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn const_module(name: &str, value: f32) -> Module {
    use wax_ir::DenseElements;
    let mut m = Module::new(name);
    let tile = Type::Tile(TileType {
        shape: vec![1, 32],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let (entry, block, _) = m.begin_function(name, &[]).expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );
    let mut data = Vec::new();
    for _ in 0..32 {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let (cid, cres) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(tile.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: Type::Scalar(ScalarType::F32),
                shape: vec![1, 32],
                data,
            }),
        )
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, cid);
    let (nid, _) = OpBuilder::new(Opcode::NegF, Location::Unknown)
        .operands([cres[0]])
        .result(tile)
        .build(&mut m)
        .expect("op builds");
    append_op(&mut m, block, nid);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("return builds");
    append_op(&mut m, block, ret);
    m
}

fn constant_values(m: &mut Module) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                if pliron::operation::Operation::get_op_dyn(op, ctx)
                    .op_ref()
                    .get_opid()
                    .to_string()
                    != "llvm.constant"
                {
                    continue;
                }
                if let Some(c) = pliron_llvm::ops::ConstantOp::wrap_operation(op)
                    .downcast::<pliron_llvm::ops::ConstantOp>()
                {
                    out.push(c.get_value(ctx).disp(ctx).to_string());
                }
            }
        }
    }
    out
}

#[test]
fn float_negate_lowers_to_llvm_fneg() {
    assert_lowers_to(
        unary_module("k", Opcode::NegF, ScalarType::F32, ScalarType::F32, None),
        "llvm.fneg",
    );
}

#[test]
fn atan2_calls_the_air_intrinsic() {
    assert_calls(
        binary_module("k", Opcode::Atan2, ScalarType::F32, None),
        "@air.fast_atan2",
    );
}

#[test]
fn atomic_rmw_names_its_operation_and_address_space() {
    for (elem, mode, expected) in [
        (ScalarType::I32, 3, "@air.atomic.global.add.s.i32"),
        (ScalarType::I32, 2, "@air.atomic.global.xor.s.i32"),
        (ScalarType::F32, 4, "@air.atomic.global.add.f32"),
    ] {
        let res = intrinsic_names(&mut rmw_module("k", elem.clone(), mode));
        assert_eq!(res, vec![expected.to_string()], "rmw {elem:?} mode {mode}");
    }
}

#[test]
fn atomic_compare_and_swap_calls_cmpxchg() {
    for (elem, expected) in [
        (ScalarType::I32, "@air.atomic.global.cmpxchg.weak.i32"),
        (ScalarType::F32, "@air.atomic.global.cmpxchg.weak.f32"),
    ] {
        let res = intrinsic_names(&mut cas_module("k", elem.clone()));
        assert_eq!(res, vec![expected.to_string()], "cas {elem:?}");
    }
}

#[test]
fn scan_within_one_simd_group_is_the_shuffle_walk_alone() {
    let res = intrinsic_names(&mut scan_module("k", 32, false));
    assert_eq!(
        res.iter()
            .filter(|n| n.starts_with("@air.simd_shuffle_up"))
            .count(),
        5,
        "expected log2(32) shuffles, got {res:?}"
    );
    assert!(
        !res.iter().any(|n| n.contains("barrier")),
        "32-wide scan must not use threadgroup memory: {res:?}"
    );
    let names = air_op_names(&mut scan_module("k", 32, false));
    assert!(
        !names.iter().any(|n| n == "air.global_ref"),
        "32-wide scan allocated threadgroup scratch: {names:?}"
    );
}

#[test]
fn scan_past_one_simd_group_adds_a_threadgroup_round_trip() {
    let res = intrinsic_names(&mut scan_module("k", 64, false));
    assert_eq!(
        res.iter()
            .filter(|n| n.as_str() == "@air.wg.barrier")
            .count(),
        2,
        "publish and consume are each behind a barrier: {res:?}"
    );
    let names = air_op_names(&mut scan_module("k", 64, false));
    assert!(
        names.iter().any(|n| n == "air.global_ref"),
        "no threadgroup scratch: {names:?}"
    );
    for expected in ["air.gep_array", "air.store", "air.load", "llvm.cond_br"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing `{expected}`: {names:?}"
        );
    }
}

#[test]
fn reversed_scan_shuffles_the_other_way() {
    let fwd = intrinsic_names(&mut scan_module("k", 32, false));
    let rev = intrinsic_names(&mut scan_module("k", 32, true));
    assert!(
        fwd.iter().all(|n| n.starts_with("@air.simd_shuffle_up")),
        "{fwd:?}"
    );
    assert!(
        rev.iter().all(|n| n.starts_with("@air.simd_shuffle_down")),
        "{rev:?}"
    );
    assert_eq!(
        air_op_names(&mut scan_module("k", 32, false)),
        air_op_names(&mut scan_module("k", 32, true))
    );
}

#[test]
fn tile_constant_carries_its_value() {
    let res = constant_values(&mut const_module("k", 0.5));
    assert!(
        res.iter().any(|v| v.contains("0.5")),
        "expected 0.5 constant, got {res:?}"
    );
}
