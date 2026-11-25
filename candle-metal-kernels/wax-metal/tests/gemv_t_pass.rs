mod common;

use common::profile;
const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

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
        z.store(tile_z);
    }

    #[wax::entry()]
    pub fn gemm_convert<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
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
        let out: Tile<E, { [BM, BN] }> = convert_tile(tile_z);
        z.store(out);
    }
}
"#;

fn from_src(
    entry: &str,
    g: &[&str],
    strides: &'static [(&'static str, &'static [i32])],
) -> wax_ir::Module {
    common::module_from_src(SRC, "tests/gemv_t_pass.rs", "kernels", entry, g, strides)
}

fn gemv_module(g: &[&str]) -> wax_ir::Module {
    from_src(
        "gemm",
        g,
        &[("z", &[4096, 1]), ("x", &[64, 1]), ("y", &[4096, 1])],
    )
}
fn convert_module(g: &[&str]) -> wax_ir::Module {
    from_src(
        "gemm_convert",
        g,
        &[("z", &[4096, 1]), ("x", &[64, 1]), ("y", &[4096, 1])],
    )
}
const CONVERT: [(&str, &[&str]); 3] = [
    ("f32", &["f32", "1", "32", "1", "64"]),
    ("f16", &["f16", "1", "32", "1", "64"]),
    ("bf16", &["bf16", "1", "32", "1", "64"]),
];

const G: [(&str, &[&str]); 3] = [
    ("f32", &["f32", "1", "32", "1", "64"]),
    ("f16", &["f16", "1", "32", "1", "64"]),
    ("bf16", &["bf16", "1", "32", "1", "64"]),
];

fn padded_module(e: &str, trans_b: bool) -> wax_ir::Module {
    let strides: Vec<(&str, &[i32])> = vec![
        ("z", &[4096, 1]),
        ("x", &[64, 1]),
        ("y", if trans_b { &[1, 64] } else { &[4096, 1] }),
    ];
    wax_frontend::KernelCompiler::new(
        wax_kernels::gemm::module_ast,
        wax_kernels::gemm::MODULE_NAME,
        wax_kernels::gemm::ENTRY_PADDED,
    )
    .generics(vec![
        e.into(),
        "32".into(),
        "32".into(),
        "8".into(),
        "64".into(),
    ])
    .strides(&strides)
    .compile()
    .expect("compiles")
    .to_wax_module(wax_kernels::gemm::ENTRY_PADDED)
    .expect("module")
}

fn corpus() -> Vec<(&'static str, wax_ir::Module, (i64, i32, i32))> {
    let f = |n: i64, sg: i32, splits: i32| (n, sg, splits);
    vec![
        ("sg8_f32", gemv_module(G[0].1), f(4096, 8, 1)),
        ("sg32_f32", gemv_module(G[0].1), f(4096, 32, 1)),
        ("padded_f32", padded_module("f32", false), f(4096, 8, 1)),
        ("padded_f16", padded_module("f16", false), f(4096, 8, 1)),
        (
            "padded_bf16_transb",
            padded_module("bf16", true),
            f(4096, 8, 1),
        ),
        (
            "padded_f16_transb",
            padded_module("f16", true),
            f(4096, 32, 1),
        ),
        (
            "padded_f32_transb",
            padded_module("f32", true),
            f(4096, 16, 1),
        ),
        ("split2_f32", convert_module(CONVERT[0].1), f(4096, 8, 2)),
        ("split8_f16", convert_module(CONVERT[1].1), f(4096, 8, 8)),
        ("split16_bf16_deep_k", deep_k_module("bf16"), f(512, 32, 16)),
    ]
}

fn deep_k_module(e: &'static str) -> wax_ir::Module {
    let g: &'static [&'static str] = match e {
        "f16" => &["f16", "1", "32", "1", "8192"],
        _ => &["bf16", "1", "32", "1", "8192"],
    };
    from_src(
        "gemm_convert",
        g,
        &[("z", &[512, 1]), ("x", &[8192, 1]), ("y", &[512, 1])],
    )
}

fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens/gemv_t")
        .join(format!("{name}.metallib"))
}

#[test]
fn passes_match_goldens() {
    let mut failed = Vec::new();
    for (name, m, facts) in corpus() {
        let Some((mut sm, _)) = common::specialise(&m, 1, facts.0, facts.1, facts.2) else {
            failed.push(format!("{name}: the passes declined"));
            continue;
        };
        let result = lower_both(&mut sm);
        if let Ok(dir) = std::env::var("WAX_GOLDEN_OUT") {
            let _ = std::fs::write(format!("{dir}/{name}.metallib"), &result);
        }
        match std::fs::read(golden_path(name)) {
            Ok(want) if want == result => {}
            Ok(want) => failed.push(format!(
                "{name}: {} bytes, golden {} bytes, first difference at {:?}",
                result.len(),
                want.len(),
                result.iter().zip(want.iter()).position(|(a, b)| a != b)
            )),
            Err(e) => failed.push(format!("{name}: no golden ({e})")),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

fn lower_both(m: &mut wax_ir::Module) -> Vec<u8> {
    let units: Vec<_> = m
        .entries()
        .iter()
        .map(|&f| wax_metal::wax_to_air::EntryUnit {
            func: f,
            kernel_symbol: wax_metal::wax_to_air::kernel_symbol(m, f).expect("named"),
            args: wax_metal::lower::kernel_args_for_module(m, f).expect("args"),
        })
        .collect();
    wax_metal::lower::lower_to_metallib_entries(m, &units, &profile())
        .expect("lowers")
        .metallib
}

#[test]
fn gemv_t_realisation_bakes_every_extent() {
    for e in ["f32", "f16", "bf16"] {
        for trans_b in [false, true] {
            let m = padded_module(e, trans_b);
            assert!(
                !wax_metal::lower::promoted_scalars(&m).is_empty(),
                "padded GEMM promotes extents"
            );
            let (sm, _) = common::specialise(&m, 1, 4096, 8, 1).expect("rewrites");
            assert_eq!(wax_metal::lower::promoted_scalars(&sm), vec![]);
            let func = *sm.entries().first().expect("entry");
            let args = wax_metal::lower::kernel_args_for_module(&sm, func).expect("args");
            assert_eq!(args.len(), 5, "{e} trans_b={trans_b}: {args:?}");
        }
    }
}

#[test]
fn indivisible_split_downgrades() {
    let m = gemv_module(G[0].1);
    let (_, s) = common::specialise(&m, 1, 4096, 8, 3).expect("downgrades");
    assert!(s.reduce.is_none());
    assert_eq!(
        s.launch.as_ref().expect("realised").block.z,
        1,
        "downgrade launches single plane"
    );
}

#[test]
fn narrow_output_is_declined() {
    let m = gemv_module(G[0].1);
    for n in [1i64, 8, 16, 31] {
        assert!(
            common::specialise(&m, 1, n, 32, 1).is_none(),
            "n = {n} < 32 lanes"
        );
    }
    assert!(common::specialise(&m, 1, 32, 32, 1).is_some());
}

#[test]
fn as_written_runs_the_pipeline_and_realises_nothing() {
    use wax_ir::passes::launch::Launch;
    use wax_metal::gemm::{GemmSpec, realise};

    let m = gemv_module(G[0].1);
    let spec = GemmSpec {
        trans_b: false,
        m: 1,
        n: 4096,
        k: 0,
        elem_bytes: 4,
        ldc: 4096,
    };

    let as_written = realise(&m, 0, 1, &spec).expect("pipeline runs");
    assert!(
        as_written.rewritten.launch.is_none() && as_written.rewritten.declined.is_none(),
        "got {:?}",
        as_written.rewritten
    );
    assert!(Launch::of(&as_written.module.ctx, as_written.func).is_none());
    assert_eq!(marked_only_hash(&m), module_hash(&as_written.module));

    let realised = realise(&m, 8, 1, &spec).expect("pipeline runs");
    assert!(
        realised.rewritten.launch.is_some() && realised.rewritten.declined.is_none(),
        "got {:?}",
        realised.rewritten
    );
    assert_ne!(marked_only_hash(&m), module_hash(&realised.module));
}

fn module_hash(m: &wax_ir::Module) -> u64 {
    use pliron::irbuild::equivalence::{IGNORE_LOC_NAMES, operation_hash};
    use pliron::op::Op;
    operation_hash(&m.ctx, m.module_op().get_operation(), &IGNORE_LOC_NAMES)
}

fn marked_only_hash(src: &wax_ir::Module) -> u64 {
    use pliron::pass::{AnalysisManager, Pass, Passes};
    use wax_ir::passes::gemm_analysis::GemmMarkPass;
    let (mut owned, func) = wax_ir::passes::utils::working_copy(src).expect("an entry function");
    let mut passes = Passes::default();
    passes.add_pass(GemmMarkPass::default());
    passes
        .run(func, &mut owned.ctx, &mut AnalysisManager::default())
        .expect("the mark pass runs");
    module_hash(&owned)
}

#[test]
fn reduce_implies_a_realisation() {
    let mut split_cases = 0;
    for (name, m, facts) in corpus() {
        let Some((_, r)) = common::specialise(&m, 1, facts.0, facts.1, facts.2) else {
            continue;
        };
        assert!(
            r.reduce.is_none() || r.launch.is_some(),
            "{name}: reduce without realisation"
        );
        assert!(
            r.declined.is_none() || r.reduce.is_none(),
            "{name}: declined yet has reduce"
        );
        if r.reduce.is_some() {
            split_cases += 1;
        }
    }
    assert!(split_cases >= 3, "corpus missing splits, saw {split_cases}");
}

#[test]
fn multi_entry_module_is_rejected() {
    let one = gemv_module(G[0].1);
    let root = *one.entries().first().expect("entry");
    let mut two = wax_ir::Module::new("two");
    let first = two.clone_function_from(&one, root, "first").expect("clone");
    two.mark_entry(first);
    let second = two
        .clone_function_from(&one, root, "second")
        .expect("clone");
    two.mark_entry(second);
    assert_eq!(two.entries().len(), 2);

    let spec = wax_metal::gemm::GemmSpec {
        m: 1,
        n: 4096,
        k: 0,
        elem_bytes: 4,
        trans_b: false,
        ldc: 4096,
    };
    match wax_metal::gemm::realise(&two, 8, 1, &spec) {
        Err(e) => assert!(
            format!("{e:?}").contains("exactly one entry"),
            "unexpected error: {e:?}"
        ),
        Ok(_) => panic!("two entries must be rejected"),
    }

    assert!(
        wax_metal::gemm::realise(&one, 8, 1, &spec)
            .expect("one entry runs")
            .rewritten
            .launch
            .is_some()
    );
}
