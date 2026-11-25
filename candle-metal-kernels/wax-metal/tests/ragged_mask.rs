mod common;

const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

    #[wax::entry()]
    pub fn gemm_padded<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
        const KB: i32,
    >(
        z: &mut Tensor<E, { [M, N] }>,
        x: &Tensor<E, { [M, K] }>,
        y: &Tensor<E, { [K, N] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let xp = x.partition(const_shape![BM, BK]);
        let yp = y.partition(const_shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..KB {
            acc = mmaf(xp.load([pid.0, i]), yp.load([i, pid.1]), acc);
        }
        let ztok: Token = get_tensor_token(z);
        let mut zp: PartitionMut<E, { [BM, BN] }> =
            unsafe { make_partition_view_mut(z, const_shape![BM, BN], padding::Zero, ztok) };
        let narrowed: Tile<E, { [BM, BN] }> = convert_tile(acc);
        unsafe { zp.store(narrowed, [pid.0, pid.1]) };
    }
}
"#;

fn lower(generics: &'static [&'static str]) -> Vec<u8> {
    let mut module = common::module_from_src(
        SRC,
        "tests/ragged_mask.rs",
        "kernels",
        "gemm_padded",
        generics,
        &[("z", &[64, 1]), ("x", &[64, 1]), ("y", &[64, 1])],
    );
    let profile = wax_metal::TargetProfile::new(wax_metal::Platform::MacOS, 26, 0);
    wax_metal::lower::lower_to_metallib_for(&mut module, &profile)
        .expect("lowers")
        .metallib
}

#[test]
fn mask_is_applied_correctly() {
    let a = lower(&["f32", "16", "16", "8", "32", "64", "64", "8"]);
    let b = lower(&["f32", "16", "16", "8", "64", "64", "64", "8"]);
    assert_eq!(a.len(), b.len(), "an exact multiple must not gain a mask");

    let ragged = lower(&["f32", "16", "16", "8", "7", "64", "64", "8"]);
    assert_ne!(b.len(), ragged.len(), "ragged extent must emit mask");
}
