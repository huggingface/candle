mod common;
use common::all_ops;

use wax_frontend::{KernelCompiler, ast};

/// Two loops reading the same view. Should trigger cross loop reuse.
fn kernel_ast() -> ast::Module {
    const SRC: &str = r#"
mod kernels {
    use cutile::core::*;
    #[wax::entry()]
    fn two_pass<const N: i32, const CHUNKS: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
        x: &Tensor<f32, { [CHUNKS, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut acc = px.load([0i32, 0i32]);
        for i in 0i32..CHUNKS {
            acc = acc + px.load([i, 0i32]);
        }
        for i in 0i32..CHUNKS {
            acc = acc + px.load([i, 0i32]);
        }
        let mut po: PartitionMut<f32, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        unsafe { po.store(acc, [0i32, 0i32]) };
    }
}
"#;
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let mut m = ast::Module::with_span_base(
        "kernels",
        item,
        ast::SpanBase::new("tests/passes_run.rs".into(), 1, 1),
    );
    m.set_absolute_path("wax_frontend::kernels".to_string());
    m
}

#[test]
fn cross_loop_reuse_annotations_reach_the_pliron_module() {
    let c = KernelCompiler::new(kernel_ast, "kernels", "two_pass")
        .generics(vec!["32".into(), "4".into()])
        .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
        .compile()
        .expect("frontend compiles two_pass");

    let m = c.to_wax_module("two_pass").expect("module builds");
    let f = *m.entries().first().expect("one function");

    let mut analyses = wax_ir::passes::AnalysisManager::default();
    let _ = wax_ir::passes::cross_loop_reuse::annotate_with(&mut analyses, &m.ctx, f);
    use wax_ir::source::WaxSource;
    let annotated: Vec<String> = all_ops(&m)
        .iter()
        .flat_map(|o| m.attributes(*o).into_iter().map(|(k, _)| k))
        .filter(|k| k.starts_with("wax_cache_"))
        .collect();

    assert!(
        !annotated.is_empty(),
        "no `wax_cache_*` attribute in the module: pass did not run"
    );
}
