mod common;

const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

    #[wax::entry()]
    pub fn gemm_plain<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let xp = x.partition(const_shape![BM, BK]);
        let yp = y.partition(const_shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            acc = mmaf(xp.load([pid.0, i]), yp.load([i, pid.1]), acc);
        }
        let ztok: Token = get_tensor_token(z);
        let mut zp: PartitionMut<E, { [BM, BN] }> =
            unsafe { make_partition_view_mut(z, const_shape![BM, BN], padding::None, ztok) };
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        unsafe { zp.store(out, [pid.0, pid.1]) };
    }

    #[wax::entry()]
    pub fn gemm_padded<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let xp = x.partition(const_shape![BM, BK]);
        let yp = y.partition(const_shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            acc = mmaf(xp.load([pid.0, i]), yp.load([i, pid.1]), acc);
        }
        let ztok: Token = get_tensor_token(z);
        let mut zp: PartitionMut<E, { [BM, BN] }> =
            unsafe { make_partition_view_mut(z, const_shape![BM, BN], padding::Zero, ztok) };
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        unsafe { zp.store(out, [pid.0, pid.1]) };
    }
}
"#;

fn gemm_module(name: &str) -> wax_ir::Module {
    common::module_from_src(
        SRC,
        "tests/promoted_scalars.rs",
        "kernels",
        name,
        &["f32", "32", "32", "8", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    )
}
#[test]
fn plain_store_promotes_nothing() {
    let m = gemm_module("gemm_plain");
    assert_eq!(wax_metal::lower::promoted_scalars(&m), vec![]);
}

#[test]
fn padded_store_promotes_output_extents() {
    let m = gemm_module("gemm_padded");
    let promoted = wax_metal::lower::promoted_scalars(&m);
    for (i, p) in promoted.iter().enumerate() {
        assert_ne!(p.param, usize::MAX, "slot {i} has no parameter");
        assert_ne!(p.dim, usize::MAX, "slot {i} has no dimension");
    }
    assert_eq!(
        promoted,
        vec![
            wax_metal::lower::PromotedScalar { param: 0, dim: 0 },
            wax_metal::lower::PromotedScalar { param: 0, dim: 1 },
        ]
    );
    assert_eq!(promoted.len(), wax_metal::lower::promoted_scalar_count(&m));
}
