fn dsl_module(rows: i32, cols: i32, splits: i32, dtype: &str) -> wax_ir::Module {
    dsl_entry(rows, cols, splits, dtype, wax_kernels::splitk_reduce::ENTRY)
}

fn dsl_entry(rows: i32, cols: i32, splits: i32, dtype: &str, entry: &str) -> wax_ir::Module {
    let ldc = cols;
    wax_frontend::KernelCompiler::new(
        wax_kernels::splitk_reduce::module_ast,
        wax_kernels::splitk_reduce::MODULE_NAME,
        entry,
    )
    .generics(vec![
        dtype.to_string(),
        rows.to_string(),
        cols.to_string(),
        splits.to_string(),
        (splits * rows).to_string(),
    ])
    .strides(&[("out", &[ldc, 1]), ("part", &[cols, 1])])
    .compile()
    .unwrap_or_else(|e| panic!("{entry}: frontend: {e}"))
    .to_wax_module(entry)
    .unwrap_or_else(|e| panic!("{entry}: wax module: {e}"))
}

fn stamped_module(rows: i64, cols: i64, splits: i32, ldc: i64) -> wax_ir::Module {
    let mut m = wax_ir::Module::new("splitk_reduce");
    let module_op = m.module_op();
    let _ = wax_ir::passes::split_k::build(
        &mut m.ctx,
        module_op,
        "splitk_reduce",
        "splitk_reduce",
        &wax_ir::types::ScalarType::F32,
        rows,
        cols,
        splits,
        ldc,
        32,
    );
    m
}

#[test]
fn module_compiles_and_lowers() {
    for &(rows, cols, splits) in &[(1i32, 64i32, 4i32), (8, 64, 4), (8, 100, 2)] {
        let mut m = dsl_module(rows, cols, splits, "f32");
        wax_metal::lower_to_metallib(&mut m)
            .unwrap_or_else(|e| panic!("{rows}x{cols} splits={splits}: {e}"));
    }
}

#[test]
fn module_promotes_the_destination_extents() {
    let promoted = wax_metal::lower::promoted_scalars(&dsl_module(8, 64, 4, "f32"));
    assert_eq!(
        promoted.len(),
        2,
        "padded mutable store must promote two extents, got {promoted:?}"
    );
    assert!(
        promoted.iter().all(|p| p.param == 0),
        "promoted extents should all be outputs', got {promoted:?}"
    );
    let stamped = wax_metal::lower::promoted_scalars(&stamped_module(8, 64, 4, 64));
    assert!(stamped.is_empty(), "stamped module promoted: {stamped:?}");
}

#[test]
fn dsl_and_direct_build_produces_identical_modules() {
    for &(rows, cols, splits) in &[(1i32, 64i32, 4i32), (8, 64, 4), (8, 100, 2)] {
        let dsl = wax_metal::lower_to_metallib(&mut dsl_module(rows, cols, splits, "f32")).is_ok();
        let stamped = wax_metal::lower_to_metallib(&mut stamped_module(
            rows as i64,
            cols as i64,
            splits,
            cols as i64,
        ))
        .is_ok();
        assert_eq!(
            dsl, stamped,
            "{rows}x{cols} splits={splits}: dsl={dsl} stamped={stamped}"
        );
    }
}

#[test]
fn packed_entry_promotes_nothing() {
    let mut m = dsl_entry(8, 64, 4, "f32", wax_kernels::splitk_reduce::ENTRY_PACKED);
    let promoted = wax_metal::lower::promoted_scalars(&m);
    assert!(
        promoted.is_empty(),
        "plain store should bake extents, promoted {promoted:?}"
    );
    wax_metal::lower_to_metallib(&mut m).expect("packed lowers");
}
