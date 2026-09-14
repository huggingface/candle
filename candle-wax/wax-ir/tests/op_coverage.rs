use pliron::context::Context;
use wax_ir::Opcode;
use wax_ir::dialect::ops::{simple_from_opcode, simple_to_opcode};

/// Opcodes the `wax` dialect does not support yet.
const NO_DIALECT_OP: &[Opcode] = &[
    Opcode::Module,
    Opcode::Global,
    Opcode::Print,
    Opcode::Loop,
    Opcode::JoinTokens,
    Opcode::Cat,
    Opcode::Unpack,
    Opcode::MmaFScaled,
    Opcode::AtomicRedViewTko,
    Opcode::Alloca,
    Opcode::AbsI,
    Opcode::Ceil,
    Opcode::CosH,
    Opcode::Floor,
    Opcode::GetGlobal,
    Opcode::MmaI,
    Opcode::MulhiI,
    Opcode::Permute,
    Opcode::Pow,
    Opcode::PtrToInt,
    Opcode::PtrToPtr,
    Opcode::RemF,
    Opcode::SinH,
    Opcode::StorePtrTko,
    Opcode::Tan,
    Opcode::TanH,
];

/// Ops taht are built through a named arm (not the simple op table).
const STRUCTURED: &[Opcode] = &[
    Opcode::Entry,
    Opcode::For,
    Opcode::If,
    Opcode::Yield,
    Opcode::Break,
    Opcode::Continue,
    Opcode::Return,
    Opcode::Assert,
    Opcode::Assume,
    Opcode::Constant,
    Opcode::Select,
    Opcode::Fma,
    Opcode::Broadcast,
    Opcode::Reshape,
    Opcode::Extract,
    Opcode::Offset,
    Opcode::MmaF,
    Opcode::MakeToken,
    Opcode::MakeTensorView,
    Opcode::MakePartitionView,
    Opcode::LoadViewTko,
    Opcode::StoreViewTko,
    Opcode::CmpF,
    Opcode::CmpI,
];

/// A dialect op carrying a typed enum
fn is_family(op: Opcode) -> bool {
    use wax_ir::dialect::attrs::{BinaryAttr, CastAttr, UnaryAttr};
    BinaryAttr::from_opcode(op).is_some()
        || UnaryAttr::from_opcode(op).is_some()
        || CastAttr::from_opcode(op).is_some()
}

#[test]
fn all_supported_opcodes_have_a_wax_op() {
    let ctx = &mut Context::new();
    let mut missing = Vec::new();

    for op in Opcode::ALL {
        if NO_DIALECT_OP.contains(&op) || STRUCTURED.contains(&op) || is_family(op) {
            continue;
        }
        if simple_from_opcode(ctx, op, vec![], vec![], 0).is_none() {
            missing.push(format!("{op:?}"));
        }
    }

    assert!(
        missing.is_empty(),
        "the `wax` dialect cannot express {} opcode(s): {}",
        missing.len(),
        missing.join(", ")
    );
}

#[test]
fn simple_ops_round_trip() {
    let ctx = &mut Context::new();
    for op in Opcode::ALL {
        if let Some(built) = simple_from_opcode(ctx, op, vec![], vec![], 0) {
            assert_eq!(
                simple_to_opcode(ctx, built),
                Some(op),
                "{op:?} did not read back as itself"
            );
        }
    }
}
