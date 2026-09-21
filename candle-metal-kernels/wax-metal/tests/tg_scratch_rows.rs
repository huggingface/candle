mod common;

const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

    #[wax::entry()]
    pub fn col_reduce<const BN: i32>(
        out: &mut Tensor<f32, { [-1, 32] }>,
        x: &Tensor<f32, { [-1, 32] }>,
    ) {
        let wide = const_shape![BN, 32];
        let chunk = const_shape![1, 32];
        let xp = x.partition(wide);
        let t: Tile<f32, { [BN, 32] }> = xp.load([0i32, 0i32]);
        let mx: Tile<f32, { [32] }> = reduce_max(t, 0i32);
        let mxw: Tile<f32, { [BN, 32] }> = mx.reshape(chunk).broadcast(wide);
        let d: Tile<f32, { [BN, 32] }> = t - mxw;
        let s1: Tile<f32, { [32] }> = reduce_sum(d, 0i32);
        let a: Tile<f32, { [BN, 32] }> = d * mxw;
        let b: Tile<f32, { [BN, 32] }> = d + mxw;
        let s2: Tile<f32, { [32] }> = reduce_sum(a, 0i32);
        let s3: Tile<f32, { [32] }> = reduce_sum(b, 0i32);
        let r: Tile<f32, { [32] }> = s1 + s2 + s3;
        let o: Tile<f32, { [1, 32] }> = r.reshape(chunk);
        let mut op: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(chunk) };
        unsafe { op.store(o, [0i32, 0i32]) };
    }
}
"#;

fn col_reduce() -> wax_ir::Module {
    common::module_from_src(
        SRC,
        "tests/tg_scratch_rows.rs",
        "kernels",
        "col_reduce",
        &["32"],
        &[("out", &[32, 1]), ("x", &[32, 1])],
    )
}

fn with_rows(m: &mut wax_ir::Module, rows: i32) {
    let f = *m.entries().first().expect("one entry function");
    wax_ir::builder::set_op_attr_in(
        &m.ctx,
        f,
        &pliron::ident!("tg_scratch_rows"),
        wax_ir::attr::Attribute::DenseI32Array(vec![rows]),
    );
}

#[test]
fn attribute_binds_and_128_rows_is_default() {
    let default = wax_metal::lower_to_metallib(&mut col_reduce()).expect("col_reduce lowers");

    let mut m = col_reduce();
    with_rows(&mut m, 128);
    let same = wax_metal::lower_to_metallib(&mut m).expect("lowers with attribute");
    assert_eq!(same, default, "128 rows is the default");

    let mut m = col_reduce();
    with_rows(&mut m, 1);
    let one_row = wax_metal::lower_to_metallib(&mut m).expect("lowers with the attribute");
    assert_ne!(one_row, default, "attribute did not change output");
}

#[test]
fn smaller_scratch_costs_instructions() {
    let mut small = col_reduce();
    with_rows(&mut small, 1);
    let small = wax_metal::lower_to_metallib(&mut small).unwrap();

    let mut big = col_reduce();
    with_rows(&mut big, 128);
    let big = wax_metal::lower_to_metallib(&mut big).unwrap();

    assert!(
        small.len() > big.len(),
        "expected the 1-row build to need more code, got {} vs {}",
        small.len(),
        big.len()
    );
}
