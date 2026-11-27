mod common;

use common::profile;

use wax_metal::gemm::{GemmSpec, Plan, lower_gemm};

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

fn blocked(
    generics: &[&str],
    strides: &'static [(&'static str, &'static [i32])],
) -> wax_ir::Module {
    common::module_from_src(
        SRC,
        "tests/gemm_lowering.rs",
        "kernels",
        "gemm",
        generics,
        strides,
    )
}

fn gemm_module() -> wax_ir::Module {
    blocked(
        &["f32", "32", "32", "8", "64"],
        &[("z", &[4096, 1]), ("x", &[64, 1]), ("y", &[4096, 1])],
    )
}

fn gemm_module_f16() -> wax_ir::Module {
    blocked(
        &["f16", "32", "32", "8", "256"],
        &[("z", &[4096, 1]), ("x", &[256, 1]), ("y", &[4096, 1])],
    )
}

fn spec(m: usize, n: usize, k: usize) -> GemmSpec {
    GemmSpec {
        trans_b: false,
        m,
        n,
        k,
        elem_bytes: 4,
        ldc: n,
        batch: 1,
        a_batch: 0,
        b_batch: 0,
    }
}

#[test]
fn large_square_routes_to_mpp_and_substitutes() {
    let m = gemm_module();
    let got = lower_gemm(&m, &spec(1024, 1024, 64), &profile()).expect("lowers");
    assert!(
        matches!(got.wanted, Plan::Mpp { .. }),
        "expected MPP for a large square, got {:?} ({:?})",
        got.wanted,
        got.reason
    );
    assert_eq!(
        got.fell_back, None,
        "MPP was chosen but not emitted: {:?}",
        got.fell_back
    );
    assert!(
        got.kernel.metallib.len() > 512,
        "implausibly small metallib: {} bytes",
        got.kernel.metallib.len()
    );
    assert_eq!(got.kernel.launch.threadgroup, (32, 1, 1));
}

#[test]
fn sgmma_shapes_lowers_kernel_as_written() {
    let m = gemm_module();
    let got = lower_gemm(&m, &spec(1024, 8, 64), &profile()).expect("lowers");
    assert!(
        matches!(got.wanted, Plan::Lower { sg: 0, splits: 1 }),
        "expected the tile loop for thin N, got {:?} ({:?})",
        got.wanted,
        got.reason
    );
    assert_eq!(got.fell_back, None, "the tile loop is not a fall-back here");
    assert_eq!(
        got.kernel.launch.grid,
        (1024usize.div_ceil(32), 8usize.div_ceil(32), 1)
    );
    assert_eq!(
        got.kernel.launch.threadgroup,
        (32, 1, 1),
        "a fragment kernel dispatches one SIMD group"
    );
}

#[test]
fn gemv_t_lowers_correctly() {
    let gemv_blocked = blocked(
        &["f32", "1", "32", "1", "64"],
        &[("z", &[4096, 1]), ("x", &[64, 1]), ("y", &[4096, 1])],
    );
    for (name, m) in [
        ("gemm blocked", gemm_module()),
        ("gemv blocked", gemv_blocked),
    ] {
        let result = lower_gemm(&m, &spec(1, 4096, 64), &profile()).expect("lowers");
        assert!(
            matches!(result.wanted, Plan::Lower { sg: 8, splits: 1 }),
            "{name}: expected GemvT {{ sg: 8 }} for 1x4096x64 f32, got {:?} ({:?})",
            result.wanted,
            result.reason
        );
        assert_eq!(result.fell_back, None,);
        assert_eq!(result.kernel.launch.grid, (4096usize / 32, 1, 1),);
        assert_eq!(result.kernel.launch.threadgroup, (8 * 32, 1, 1),);
        assert_eq!(result.kernel.scalar_args, vec![]);
    }
}

#[test]
fn mid_band_should_lower_to_gemv() {
    let m16 = gemm_module_f16();
    let result = lower_gemm(
        &m16,
        &GemmSpec {
            trans_b: false,
            m: 1,
            n: 4096,
            k: 256,
            elem_bytes: 2,
            ldc: 4096,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        },
        &profile(),
    )
    .expect("lowers");
    assert!(
        matches!(result.wanted, Plan::Lower { .. }),
        "expected GEMV for 1x4096x256 f16, got {:?} ({:?})",
        result.wanted,
        result.reason
    );
    assert_eq!(result.fell_back, None);
    assert_eq!(result.kernel.scratch_bytes, 0);
}

#[test]
fn unrealisable_gemm_shape_falls_back_correctly() {
    let m48 = blocked(
        &["f16", "32", "32", "8", "48"],
        &[("z", &[512, 1]), ("x", &[48, 1]), ("y", &[512, 1])],
    );
    let result = lower_gemm(
        &m48,
        &GemmSpec {
            trans_b: false,
            m: 1,
            n: 512,
            k: 48,
            elem_bytes: 2,
            ldc: 512,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        },
        &profile(),
    )
    .expect("lowers");
    assert!(
        matches!(result.wanted, Plan::Lower { .. }),
        "expected the GemvT for 1x512x48 f16, got {:?} ({:?})",
        result.wanted,
        result.reason
    );
    assert!(result.fell_back.is_some(), "GEMV specialisation declined");

    let m = gemm_module(); // K = 64
    use common::specialise;
    assert!(
        specialise(&m, 2, 4096, 8, 1).is_none(),
        "m > 1 is not the GEMV family"
    );
    assert!(
        specialise(&m, 1, 4096, 48, 1).is_none(),
        "K = 64 does not divide the SIMD-group count"
    );
}

fn gemm_module_k4096() -> wax_ir::Module {
    blocked(
        &["f16", "32", "32", "8", "4096"],
        &[("z", &[512, 1]), ("x", &[4096, 1]), ("y", &[512, 1])],
    )
}

#[test]
fn split_k_realises_scratch_and_second_dispatch() {
    let m = gemm_module_k4096();
    let result = lower_gemm(
        &m,
        &GemmSpec {
            trans_b: false,
            m: 1,
            n: 512,
            k: 4096,
            elem_bytes: 2,
            ldc: 512,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        },
        &profile(),
    )
    .expect("lowers");
    assert!(
        matches!(result.wanted, Plan::Lower { sg: 32, splits: 16 }),
        "expected GemvTSplitK for 1x512x4096 f16, got {:?} ({:?})",
        result.wanted,
        result.reason
    );
    assert_eq!(result.fell_back, None);
    assert_eq!(result.kernel.launch.grid, (512 / 32, 1, 16));
    assert_eq!(result.kernel.launch.threadgroup, (32 * 32, 1, 1));
    assert_eq!(result.kernel.scratch_bytes, 16 * 512 * 4);

    let second = result.kernel.second.as_ref().expect("reduce");
    assert_eq!(second.launch.grid, (512 / 32, 1, 1));
    assert_eq!(second.launch.threadgroup, (32, 1, 1));
    assert!(second.entry.contains("reduce"), "{}", second.entry);
}

#[test]
fn unsplittable_split_k_downgrades_to_single_pass() {
    let m = gemm_module_k4096();
    let result = lower_gemm(
        &m,
        &GemmSpec {
            trans_b: false,
            m: 1,
            n: 96,
            k: 4096,
            elem_bytes: 2,
            ldc: 96,
            batch: 1,
            a_batch: 0,
            b_batch: 0,
        },
        &profile(),
    )
    .expect("lowers");
    assert!(
        matches!(result.wanted, Plan::Lower { sg: 32, splits: 3 }),
        "expected the split-K for 1x96x4096 f16, got {:?} ({:?})",
        result.wanted,
        result.reason
    );
    assert!(result.fell_back.is_some());
    assert!(result.kernel.second.is_none());
    assert_eq!(result.kernel.launch.grid, (96 / 32, 1, 1));
    assert_eq!(result.kernel.launch.threadgroup, (32 * 32, 1, 1));
    assert_eq!(result.kernel.scratch_bytes, 0);
}

#[test]
fn relowering_is_identical() {
    let m = gemm_module();
    for (m_dim, n_dim) in [
        (1024usize, 1024usize),
        (1024usize, 8usize),
        (1usize, 4096usize),
    ] {
        let a = lower_gemm(&m, &spec(m_dim, n_dim, 64), &profile()).expect("first");
        let b = lower_gemm(&m, &spec(m_dim, n_dim, 64), &profile()).expect("second");
        assert_eq!(
            a.kernel.metallib, b.kernel.metallib,
            "{}x{}: bytes differ ({:?} wanted)",
            m_dim, n_dim, a.wanted
        );
        assert_eq!(
            a.kernel.launch, b.kernel.launch,
            "{}x{}: launch differs",
            m_dim, n_dim
        );
    }
}

#[test]
fn padded_gemm_lowers_correctly() {
    let m = wax_frontend::KernelCompiler::new(
        wax_kernels::gemm::module_ast,
        wax_kernels::gemm::MODULE_NAME,
        wax_kernels::gemm::ENTRY_PADDED,
    )
    .generics(vec![
        "f16".into(),
        "16".into(),
        "32".into(),
        "8".into(),
        "2048".into(),
    ])
    .strides(&[("z", &[4096i32, 1]), ("x", &[2048, 1]), ("y", &[4096, 1])])
    .compile()
    .expect("padded entry compiles")
    .to_wax_module(wax_kernels::gemm::ENTRY_PADDED)
    .expect("module");
    let spec = GemmSpec {
        trans_b: false,
        m: 300,
        n: 4096,
        k: 2048,
        elem_bytes: 2,
        ldc: 4096,
        batch: 1,
        a_batch: 0,
        b_batch: 0,
    };
    let result = lower_gemm(&m, &spec, &profile()).expect("lowers");
    assert!(
        matches!(result.wanted, Plan::Mpp { .. }),
        "expected MPP for large ragged, got {:?} ({:?})",
        result.wanted,
        result.reason
    );
    assert!(result.fell_back.is_none());
}
