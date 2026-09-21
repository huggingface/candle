use wax_ir::dialect::text::{parse, print};
use wax_ir::types::{
    DYNAMIC, FuncType, GatherScatterViewType, PaddingValue, PartitionViewType, PointerType,
    ScalarType, StridedViewType, TensorViewType, TileElementType, TileType, Type,
};

fn tv() -> TensorViewType {
    TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC, 64],
        strides: vec![4096, 1],
    }
}

fn tile(e: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(e),
    })
}

fn types() -> Vec<Type> {
    let mut v: Vec<Type> = Vec::new();
    // All scalars.
    for s in [
        ScalarType::I1,
        ScalarType::I4,
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::F16,
        ScalarType::BF16,
        ScalarType::F32,
        ScalarType::TF32,
        ScalarType::F64,
        ScalarType::F8E4M3FN,
        ScalarType::F8E5M2,
        ScalarType::F8E8M0FNU,
        ScalarType::F4E2M1FN,
    ] {
        v.push(Type::Scalar(s.clone()));
        v.push(Type::Pointer(PointerType { pointee: s.clone() }));
        v.push(Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(s.clone()),
        }));
        v.push(Type::Tile(TileType {
            shape: vec![32, 8],
            element_type: TileElementType::Scalar(s.clone()),
        }));
        v.push(Type::Tile(TileType {
            shape: vec![DYNAMIC],
            element_type: TileElementType::Pointer(Box::new(PointerType { pointee: s })),
        }));
    }
    v.push(Type::Token);
    v.push(Type::TensorView(tv()));
    for pad in [None, Some(PaddingValue::Zero), Some(PaddingValue::NegInf)] {
        for dim_map in [vec![], vec![0, 1]] {
            v.push(Type::PartitionView(PartitionViewType {
                tile_shape: vec![32, 32],
                tensor_view: tv(),
                dim_map: dim_map.clone(),
                padding_value: pad,
            }));
            v.push(Type::StridedView(StridedViewType {
                tile_shape: vec![1, 1],
                traversal_strides: vec![1024, 1],
                tensor_view: tv(),
                dim_map,
                padding_value: pad,
            }));
        }
        v.push(Type::GatherScatterView(GatherScatterViewType {
            tile_shape: vec![16, 16],
            tensor_view: tv(),
            sparse_dim: 1,
            padding_value: pad,
        }));
    }
    // Rank 0 everywhere, the case a shape printer gets wrong.
    v.push(Type::StridedView(StridedViewType {
        tile_shape: vec![],
        traversal_strides: vec![],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![],
            strides: vec![],
        },
        dim_map: vec![],
        padding_value: None,
    }));
    v.push(Type::Func(FuncType {
        inputs: vec![Type::Scalar(ScalarType::I32), Type::TensorView(tv())],
        results: vec![],
    }));
    v.push(Type::Func(FuncType {
        inputs: vec![tile(ScalarType::I32)],
        results: vec![tile(ScalarType::F32)],
    }));
    v.push(Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    }));
    v
}

#[test]
fn types_round_trip() {
    for t in types() {
        let text = print(&t);
        match parse(&text) {
            Some(back) => assert_eq!(back, t, "{text} parsed, but incorrectly"),
            None => panic!("{text} printed, but not parsed"),
        }
        assert_eq!(print(&parse(&text).unwrap()), text, "round trip failed");
    }
}

#[test]
fn dynamic_dim_prints_question_mark() {
    let t = Type::Tile(TileType {
        shape: vec![DYNAMIC, 64],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    assert_eq!(print(&t), "tile<?x64xf32>");
}

#[test]
fn scalars_are_lowercase() {
    assert_eq!(print(&Type::Scalar(ScalarType::F32)), "f32");
    assert_eq!(print(&Type::Scalar(ScalarType::BF16)), "bf16");
    assert_eq!(print(&Type::Scalar(ScalarType::F8E4M3FN)), "f8e4m3fn");
}

#[test]
fn type_prefixes_are_handled_correctly() {
    assert_eq!(parse("i16"), Some(Type::Scalar(ScalarType::I16)));
    assert_eq!(parse("i1"), Some(Type::Scalar(ScalarType::I1)));
    assert_eq!(
        parse("f8e8m0fnu"),
        Some(Type::Scalar(ScalarType::F8E8M0FNU))
    );
}

#[test]
fn malformed_type_is_not_parsed() {
    for malformed in [
        "tile<i32>x",
        "tile<1.5xf32>",
        "ptr<nope>",
        "partition_view<>",
        "",
        "tile<",
    ] {
        assert_eq!(
            parse(malformed),
            None,
            "{malformed} is malformed and should not parse."
        );
    }
}
