use pliron::pass::Pass;
use wax_frontend::KernelCompiler;
use wax_ir::attr::Attribute;
use wax_ir::passes::gemm_analysis::{
    ATTR_SPLIT_REDUCTION, ATTR_SPLIT_REDUCTION_K, GemmAnalysis, GemmMarkPass, stamped_loops,
};
use wax_ir::passes::gemv_t::{Facts, GemvTPass};
use wax_ir::passes::launch::Launch;

#[allow(unused_unsafe)]
#[wax_macro::module]
pub mod kernels {
    use cutile::core::*;

    /// Canonical blocked GEMM. `mmaf` loop over `K / BK`.
    #[wax_macro::entry()]
    pub fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            acc = mmaf(tile_x, tile_y, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out);
    }

    /// Not a GEMM at all
    #[wax_macro::entry()]
    pub fn scale<E: ElementType, const N: i32>(
        out: &mut Tensor<E, { [1, N] }>,
        x: &Tensor<E, { [1, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        unsafe { po.store(v + v, [0i32, 0i32]) };
    }
}

const MODULE_NAME: &str = "kernels";

fn module(entry: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> wax_ir::Module {
    KernelCompiler::new(kernels::__module_ast_self, MODULE_NAME, entry)
        .generics(generics.iter().map(|s| s.to_string()).collect())
        .strides(strides)
        .compile()
        .unwrap_or_else(|e| panic!("{entry}: frontend: {e}"))
        .to_wax_module(entry)
        .unwrap_or_else(|e| panic!("{entry}: wax module: {e}"))
}

fn gemm_module(k: i32, bk: i32) -> wax_ir::Module {
    module(
        "gemm",
        &["f32", "32", "32", &bk.to_string(), &k.to_string()],
        &[("z", &[32, 1]), ("x", &[k, 1]), ("y", &[32, 1])],
    )
}

/// The GEMV blocking: BM = 1. Not in the bench corpus, so the recogniser's golden list in
/// `wax-metal` never sees this instantiation.
fn gemv_module() -> wax_ir::Module {
    module(
        "gemm",
        &["f32", "1", "32", "1", "64"],
        &[("z", &[32, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    )
}

#[test]
fn gemm_analysis_stamps_split_k_correctly() {
    let m = gemm_module(64, 8);
    let func = *m.entries().first().expect("one entry function");
    let mut analyses = wax_ir::passes::AnalysisManager::default();

    // ForOp before mark pass runs.
    let for_op = analyses
        .get_analysis::<GemmAnalysis>(func, &m.ctx)
        .expect("analysis computed")
        .pattern
        .as_ref()
        .expect("GEMM pattern recognized")
        .for_op;

    // No attributes present.
    assert_eq!(m.op_attr(for_op, &ATTR_SPLIT_REDUCTION), None);
    assert_eq!(m.op_attr(for_op, &ATTR_SPLIT_REDUCTION_K), None);

    // Run mark pass.
    let mut m = m;
    let mut pass = GemmMarkPass::default();
    let res = pass
        .run(func, &mut m.ctx, &mut analyses)
        .expect("pass runs");

    // Ensure pass ran.
    assert_eq!(pass.marked, 1);
    assert_eq!(res.ir_changed, pliron::irbuild::IRStatus::Changed);

    // Attributes present.
    assert_eq!(
        m.op_attr(for_op, &ATTR_SPLIT_REDUCTION).unwrap(),
        Attribute::Bool(true),
    );
    assert_eq!(
        m.op_attr(for_op, &ATTR_SPLIT_REDUCTION_K).unwrap(),
        Attribute::i32(64),
    );

    // Assert stamp is only on loop, not function itself.
    assert!(m.op_attr(func, &ATTR_SPLIT_REDUCTION).is_none());
}

#[test]
fn non_gemm_unaffected() {
    let mut m = module(
        "scale",
        &["f32", "32"],
        &[("out", &[32, 1]), ("x", &[32, 1])],
    );
    let func = *m.entries().first().expect("one entry function");
    let mut analyses = wax_ir::passes::AnalysisManager::default();

    let pattern = analyses
        .get_analysis::<GemmAnalysis>(func, &m.ctx)
        .expect("analysis computed")
        .pattern
        .clone();
    // Pattern not recognized.
    assert!(pattern.is_err());

    // Nothing is marked.
    let mut pass = GemmMarkPass::default();
    let res = pass
        .run(func, &mut m.ctx, &mut analyses)
        .expect("the pass runs");

    assert_eq!(pass.marked, 0);
    assert_eq!(res.ir_changed, pliron::irbuild::IRStatus::Unchanged);
}

#[test]
fn gemv_recognized_correctly() {
    let m = gemv_module();
    let func = *m.entries().first().expect("entry");
    let mut analyses = wax_ir::passes::AnalysisManager::default();

    let p = analyses
        .get_analysis::<GemmAnalysis>(func, &m.ctx)
        .expect("analysis computed")
        .pattern
        .clone()
        .expect("gemv recognized as gemm pattern");

    assert_eq!((p.bm, p.bn, p.bk), (1, 32, 1));
    assert_eq!(p.k(), 64);
}

#[test]
fn split_k_stamp_survives_gemv_rewrite() {
    let mut m = gemv_module();
    let func = *m.entries().first().expect("entry");
    let mut analyses = wax_ir::passes::AnalysisManager::default();
    let marked = GemmMarkPass::default()
        .run(func, &mut m.ctx, &mut analyses)
        .expect("the mark pass runs");
    analyses.retain_preserved(&marked);

    let mut pass = GemvTPass::new(Facts {
        m: 1,
        n: 4096,
        sg: 8,
        lanes: 32,
    });
    let res = pass
        .run(func, &mut m.ctx, &mut analyses)
        .expect("the gemv pass runs");

    assert_eq!(res.ir_changed, pliron::irbuild::IRStatus::Changed);
    assert!(
        Launch::of(&m.ctx, func).is_some(),
        "Pass rewrote and stamped launch geometry"
    );

    // Gemm analysis should be invalidated
    analyses.retain_preserved(&res);
    assert!(analyses.try_get_analysis::<GemmAnalysis>(func).is_none());

    // Ensure split k stamp is still present in rewritten IR.
    let stamped = stamped_loops(&m.ctx, func);
    assert_eq!(stamped.len(), 1);
    assert_eq!(
        m.op_attr(stamped[0], &ATTR_SPLIT_REDUCTION_K).unwrap(),
        Attribute::i32(64)
    );
}
