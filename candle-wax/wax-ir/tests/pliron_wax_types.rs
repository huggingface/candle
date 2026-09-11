use pliron::context::Context;
use wax_ir::dialect::types::{from_pliron, to_pliron};
use wax_ir::{
    FuncType, PaddingValue, PartitionViewType, PointerType, ScalarType, TensorViewType,
    TileElementType, TileType, Type,
};

fn tv(elem: ScalarType, shape: Vec<i64>, strides: Vec<i64>) -> TensorViewType {
    TensorViewType {
        element_type: elem,
        shape,
        strides,
    }
}

fn cases() -> Vec<Type> {
    let mut v = vec![
        Type::Token,
        Type::Scalar(ScalarType::F32),
        Type::Scalar(ScalarType::F16),
        Type::Scalar(ScalarType::BF16),
        Type::Scalar(ScalarType::TF32),
        Type::Scalar(ScalarType::F8E4M3FN),
        Type::Scalar(ScalarType::F8E5M2),
        Type::Scalar(ScalarType::I1),
        Type::Scalar(ScalarType::I64),
        Type::Pointer(PointerType {
            pointee: ScalarType::F32,
        }),
        Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::I32),
        }),
        Type::Tile(TileType {
            shape: vec![1, 32],
            element_type: TileElementType::Scalar(ScalarType::F32),
        }),
        Type::Tile(TileType {
            shape: vec![8, 8],
            element_type: TileElementType::Scalar(ScalarType::BF16),
        }),
        Type::TensorView(tv(ScalarType::F32, vec![1, 32], vec![32, 1])),
        // Dynamic extents (runtime shaped tensor).
        Type::TensorView(tv(
            ScalarType::F32,
            vec![wax_ir::DYNAMIC, wax_ir::DYNAMIC],
            vec![32, 1],
        )),
    ];
    for pad in [None, Some(PaddingValue::Zero), Some(PaddingValue::NegInf)] {
        v.push(Type::PartitionView(PartitionViewType {
            tile_shape: vec![1, 32],
            tensor_view: tv(ScalarType::F32, vec![1, 32], vec![32, 1]),
            dim_map: vec![0, 1],
            padding_value: pad,
        }));
    }
    v
}

#[test]
fn all_types_round_trip() {
    let ctx = &mut Context::new();
    for t in cases() {
        let h = to_pliron(ctx, &t).unwrap_or_else(|| panic!("{t:?} did not map"));
        let back = from_pliron(ctx, h).unwrap_or_else(|| panic!("{t:?} did not map back"));
        assert_eq!(back, t, "{t:?} changed across the dialect");
    }
    println!("{} types round-tripped exactly", cases().len());
}

/// A function type is module structure, not a value type.
#[test]
fn function_type_does_not_map() {
    let ctx = &mut Context::new();
    let f = Type::Func(FuncType {
        inputs: vec![Type::Scalar(ScalarType::F32)],
        results: vec![],
    });
    assert!(to_pliron(ctx, &f).is_none());
}

#[test]
fn identical_types_are_uniqued() {
    let ctx = &mut Context::new();
    let t = Type::Tile(TileType {
        shape: vec![1, 32],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let a = to_pliron(ctx, &t).unwrap();
    let b = to_pliron(ctx, &t.clone()).unwrap();
    assert_eq!(a, b, "the same type interned twice");

    let other = Type::Tile(TileType {
        shape: vec![1, 64],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    assert_ne!(
        a,
        to_pliron(ctx, &other).unwrap(),
        "different shapes collided"
    );
}
