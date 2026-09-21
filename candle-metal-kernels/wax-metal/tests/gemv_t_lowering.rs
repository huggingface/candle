mod common;

use common::profile;

use wax_ir::passes::launch::{Tile, TileBlock};

const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

    fn store_current_block_2d<E: ElementType, const S: [i32; 2]>(
        y: &mut Tensor<E, S>,
        result: Tile<E, S>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
    }

    #[wax::entry()]
    pub fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mmaf(tile_x, tile_y, tile_z);
        }
        store_current_block_2d(z, tile_z);
    }
}
"#;

fn gemv_module() -> wax_ir::Module {
    common::module_from_src(
        SRC,
        "tests/gemv_t_lowering.rs",
        "kernels",
        "gemm",
        &["f32", "1", "32", "1", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    )
}

#[test]
fn gemm_at_bm_bk_1_lowers() {
    let mut m = gemv_module();
    wax_metal::lower::lower_to_metallib_for(&mut m, &profile())
        .expect("BM=BK=1 per lane fma must lower.");
}

#[test]
fn pass_realises_the_split_k_form() {
    let m = gemv_module(); // K = 64
    let (sm, s) = common::specialise(&m, 1, 4096, 8, 2).expect("K = 64 divides 8*2");
    assert_eq!(sm.entries().len(), 2, "primary and reduce in one module");
    assert_eq!(
        s.launch.as_ref().expect("realised").block,
        TileBlock { x: 128, y: 1, z: 2 },
        "(N/32, 1, S)"
    );
    let launch = s.launch.as_ref().expect("realised");
    assert_eq!(launch.tile, Tile(vec![8, 32]));
    let reduce = s.reduce.as_ref().expect("reduce");
    assert_eq!(reduce.block, TileBlock { x: 128, y: 1, z: 1 });
    assert_eq!(reduce.tile, Tile(vec![1, 32]));
    assert!(reduce.entry.contains("reduce"), "{}", reduce.entry);
    assert_eq!(reduce.scratch_elems, 2 * 4096);
    let (_, three) = common::specialise(&m, 1, 4096, 8, 3)
        .expect("K = 64 not divisable by 8*3 = 24. Downgrades.");
    assert!(
        three.declined == Some(wax_metal::gemm::SPLIT_INDIVISIBLE),
        "split should downgrade, got {three:?}"
    );
    assert!(three.reduce.is_none(), "downgrade has no reduce");
    assert_eq!(
        three.launch.as_ref().expect("realised").block.z,
        1,
        "downgrade launches single plane"
    );
}
