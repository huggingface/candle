#![cfg(feature = "llvm-check")]

use wax_ir::builder::{OpBuilder, append_op, build_block};
use wax_ir::{
    Attribute, DenseElements, Location, Module, Opcode, ScalarType, TileElementType, TileType, Type,
};

/// Simple reduce kernel.
fn reduce_kernel(name: &str, elem: ScalarType, combine: Opcode) -> Module {
    let bytes = match elem {
        ScalarType::F16 | ScalarType::BF16 => 2,
        _ => 4,
    };
    let n = 32i64;
    let mut m = Module::new(name);
    let tile = Type::Tile(TileType {
        shape: vec![n],
        element_type: TileElementType::Scalar(elem.clone()),
    });
    let scalar = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(elem.clone()),
    });
    let (entry, block, _) = m.begin_function(name, &[]).expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );

    let (c_id, c_res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(tile.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: Type::Scalar(elem.clone()),
                shape: vec![n],
                data: vec![0u8; n as usize * bytes],
            }),
        )
        .build(&mut m)
        .expect("fixture op builds");
    append_op(&mut m, block, c_id);

    let (cb, cargs) =
        build_block(&mut m, &[scalar.clone(), scalar.clone()]).expect("the block builds");
    let (_, comb) = OpBuilder::new(combine, Location::Unknown)
        .operands([cargs[0], cargs[1]])
        .result(scalar.clone())
        .build_and_append(&mut m, cb)
        .expect("fixture op builds");
    OpBuilder::new(Opcode::Yield, Location::Unknown)
        .operands([comb[0]])
        .build_and_append(&mut m, cb)
        .expect("fixture op builds");

    let (red, _) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
        .operand(c_res[0])
        .result(scalar)
        .attr("dim", Attribute::i32(0))
        .region(vec![cb])
        .build(&mut m)
        .expect("fixture op builds");
    append_op(&mut m, block, red);

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("fixture op builds");
    append_op(&mut m, block, ret);
    m
}

#[test]
fn combine_op_names_the_simd_intrinsic() {
    for (combine, base) in [
        (Opcode::AddF, "simd_sum"),
        (Opcode::MulF, "simd_product"),
        (Opcode::MaxF, "simd_max"),
        (Opcode::MinF, "simd_min"),
    ] {
        for (elem, suf) in [(ScalarType::F32, "f32"), (ScalarType::F16, "f16")] {
            let want = format!("air.{base}.{suf}");
            let mut m = reduce_kernel("reduce_probe", elem, combine);
            let bytes = wax_metal::lower_to_metallib(&mut m)
                .unwrap_or_else(|e| panic!("{want}: did not lower: {e}"));
            let bc =
                wax_metal::metallib::bitcode_of(&bytes).unwrap_or_else(|e| panic!("{want}: {e}"));
            let p = wax_llvm_utils::parse(bc).unwrap_or_else(|e| panic!("{want}: {e}"));
            assert!(
                p.declarations().any(|f| f.name == want),
                "expected `{want}`, got {:?}",
                p.declarations().map(|f| f.name.clone()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn simd_reduce_intrinsic_is_convergent() {
    // Intrinsic must be `convergent` so that it is not moved across control flow during optim.
    let mut m = reduce_kernel("reduce_probe", ScalarType::F32, Opcode::AddF);
    let bytes = wax_metal::lower_to_metallib(&mut m).expect("lowers");
    let bc = wax_metal::metallib::bitcode_of(&bytes).expect("a metallib");
    let p = wax_llvm_utils::parse(bc).expect("LLVM reads it");
    let f = p
        .declarations()
        .find(|f| f.name == "air.simd_sum.f32")
        .expect("the reduce declares its intrinsic");
    assert!(f.has_attr("convergent"), "`{}` is not convergent", f.name);
}

#[test]
fn entry_defined_air_intrinsics_declared() {
    // Entry functions should be `define`, air intrinsics should be `declare` (prototypes).
    let mut m = reduce_kernel("reduce_probe", ScalarType::F32, Opcode::AddF);
    let bytes = wax_metal::lower_to_metallib(&mut m).expect("lowers");
    let bc = wax_metal::metallib::bitcode_of(&bytes).expect("a metallib");
    let p = wax_llvm_utils::parse(bc).expect("LLVM reads it");

    assert!(
        p.definitions().any(|f| f.name == "reduce_probe"),
        "entry is not a definition: {:?}",
        p.definitions().map(|f| f.name.clone()).collect::<Vec<_>>()
    );
    let defined_intrinsics: Vec<_> = p
        .definitions()
        .filter(|f| f.name.starts_with("air."))
        .map(|f| f.name.clone())
        .collect();
    assert!(
        defined_intrinsics.is_empty(),
        "air intrinsic must be declared, never defined: {defined_intrinsics:?}"
    );
    assert!(
        p.declarations().any(|f| f.name.starts_with("air.")),
        "no air intrinsic was declared at all"
    );
}
