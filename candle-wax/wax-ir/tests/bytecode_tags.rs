//! Verify our type vocabulary against the cuda-tile spec.
//!
//! The tags from `BytecodeTypeOpcodes.td` and `BytecodeAttrOpcodes.td` in NVIDIA/cuda-tile
//! are marked as "FROZEN for backward compatibility".
//!
//! We interpret this as meaning no types will ever disappear, but should there be emerge new type
//! that is sufficiently useful there could be additions with a new release.
//!
//! If the `TYPE_SPEC/ATTR_SPEC` needs to be updated we can extract the values from cuda-tile:
//!     grep -E "TypeTag<"     include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td
//!     grep -E "BytecodeAttrTag<" include/cuda_tile/Dialect/CudaTile/IR/BytecodeAttrOpcodes.td

use wax_ir::{Attribute, ScalarType, Type};

/// The type tag spec
const TYPE_SPEC: &[(&str, u8)] = &[
    ("Int1", 0),
    ("Int8", 1),
    ("Int16", 2),
    ("Int32", 3),
    ("Int64", 4),
    ("Float16", 5),
    ("BFloat16", 6),
    ("Float32", 7),
    ("TFloat32", 8),
    ("Float64", 9),
    ("Float8E4M3FN", 10),
    ("Float8E5M2", 11),
    ("PointerType", 12),
    ("TileType", 13),
    ("TensorViewType", 14),
    ("PartitionViewType", 15),
    ("FunctionType", 16),
    ("TokenType", 17),
    ("Float8E8M0FNU", 18),
    ("Float4E2M1FN", 19),
    ("GatherScatterViewType", 20),
    ("StridedViewType", 21),
    ("Int4", 22),
];

/// All scalar types carry their respective tag.
#[test]
fn scalar_tags_match_the_spec() {
    let pairs: &[(ScalarType, &str)] = &[
        (ScalarType::I1, "Int1"),
        (ScalarType::I4, "Int4"),
        (ScalarType::I8, "Int8"),
        (ScalarType::I16, "Int16"),
        (ScalarType::I32, "Int32"),
        (ScalarType::I64, "Int64"),
        (ScalarType::F16, "Float16"),
        (ScalarType::BF16, "BFloat16"),
        (ScalarType::F32, "Float32"),
        (ScalarType::TF32, "TFloat32"),
        (ScalarType::F64, "Float64"),
        (ScalarType::F8E4M3FN, "Float8E4M3FN"),
        (ScalarType::F8E5M2, "Float8E5M2"),
        (ScalarType::F8E8M0FNU, "Float8E8M0FNU"),
        (ScalarType::F4E2M1FN, "Float4E2M1FN"),
    ];
    for (ty, name) in pairs {
        let want = TYPE_SPEC
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not in the spec table"))
            .1;
        assert_eq!(
            ty.bytecode_tag(),
            Some(want),
            "{ty:?} must carry the frozen tag for {name}"
        );
    }
    // Accounting
    assert_eq!(pairs.len(), 15);
    assert_eq!(TYPE_SPEC.len(), 23);
}

/// All composite types carry their respective tag.
#[test]
fn composite_tags_match_the_spec() {
    let tv = wax_ir::TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![1],
        strides: vec![1],
    };
    let cases: Vec<(Type, &str)> = vec![
        (
            Type::Pointer(wax_ir::PointerType {
                pointee: ScalarType::F32,
            }),
            "PointerType",
        ),
        (
            Type::Tile(wax_ir::TileType {
                shape: vec![1],
                element_type: wax_ir::TileElementType::Scalar(ScalarType::F32),
            }),
            "TileType",
        ),
        (Type::TensorView(tv.clone()), "TensorViewType"),
        (
            Type::PartitionView(wax_ir::PartitionViewType {
                tile_shape: vec![1],
                tensor_view: tv.clone(),
                dim_map: vec![0],
                padding_value: None,
            }),
            "PartitionViewType",
        ),
        (Type::Token, "TokenType"),
    ];
    for (ty, name) in cases {
        let want = TYPE_SPEC.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(ty.bytecode_tag(), Some(want), "{name}");
    }
}

/// `External` is our escape hatch for types not supported by cutile.
/// It has no cutile encoding, and should not have one for us either.
#[test]
fn an_external_dtype_has_no_spec_tag() {
    #[derive(Debug)]
    struct Fp6;
    impl wax_ir::ScalarT for Fp6 {
        fn name(&self) -> &str {
            "fp6"
        }
        fn storage_type(&self) -> ScalarType {
            ScalarType::I8
        }
        fn compute_type(&self) -> ScalarType {
            ScalarType::F16
        }
        fn byte_width(&self) -> u32 {
            1
        }
        fn is_float(&self) -> bool {
            true
        }
    }
    let ext = ScalarType::External(std::sync::Arc::new(Fp6));
    assert_eq!(ext.bytecode_tag(), None);
    assert_eq!(Type::Scalar(ext).bytecode_tag(), None);
}

#[test]
fn spec_tags_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for (name, tag) in TYPE_SPEC {
        assert!(seen.insert(tag), "tag {tag} appears twice (at {name})");
    }
}

/// The attribute tag spec.
const ATTR_SPEC: &[(&str, u8)] = &[
    ("Integer", 1),
    ("Float", 2),
    ("Bool", 3),
    ("Type", 4),
    ("String", 5),
    ("Array", 6),
    ("DenseElements", 7),
    ("DivBy", 8),
    ("SameElements", 9),
    ("Dictionary", 10),
    ("OptimizationHints", 11),
    ("Bounded", 12),
];

/// All attributes carry their respective tag.
#[test]
fn attribute_tags_match_the_spec() {
    let ty = Type::Token;
    let cases: Vec<(Attribute, &str)> = vec![
        (Attribute::Integer(1, ty.clone()), "Integer"),
        (
            Attribute::Float(wax_ir::FloatBits::new(1.0), ty.clone()),
            "Float",
        ),
        (Attribute::Bool(true), "Bool"),
        (Attribute::Type(ty.clone()), "Type"),
        (Attribute::String("x".into()), "String"),
        (Attribute::Array(vec![]), "Array"),
        (
            Attribute::DenseElements(wax_ir::DenseElements {
                element_type: Type::Scalar(ScalarType::F32),
                shape: vec![0],
                data: vec![],
            }),
            "DenseElements",
        ),
        (Attribute::Dictionary(vec![]), "Dictionary"),
    ];
    for (a, name) in &cases {
        let want = ATTR_SPEC
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not in the attribute spec table"))
            .1;
        assert_eq!(a.bytecode_tag(), Some(want), "{name}");
    }
    assert_eq!(ATTR_SPEC.len(), 12, "the spec freezes 12 attribute tags");
}

/// `DenseI32Array` is MLIR's builtin `DenseI32ArrayAttr`, not a cuda-tile attribute.
#[test]
fn dense_i32_array_has_no_spec_tag() {
    assert_eq!(Attribute::DenseI32Array(vec![1, 2, 3]).bytecode_tag(), None);
    assert!(
        !ATTR_SPEC.iter().any(|(n, _)| *n == "DenseI32Array"),
        "the spec table must not list an MLIR builtin"
    );
}

#[test]
fn attribute_spec_tags_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for (name, tag) in ATTR_SPEC {
        assert!(
            seen.insert(tag),
            "attribute tag {tag} appears twice (at {name})"
        );
    }
}
