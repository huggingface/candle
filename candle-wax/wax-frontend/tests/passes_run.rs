//! The IR-level passes must actually run on what a backend receives.
//!
//! `Compiled::ir` used to be materialised straight from `cutile_ir::Module`, bypassing the
//! dialect and therefore every pass. Nothing failed: kernels compiled, ran and produced correct
//! results, just without the optimisations. That is the failure mode worth a test - correct
//! output is not evidence the pass ran.

use wax_frontend::{KernelCompiler, ast};

/// Two loops reading the same view, which is what cross-loop reuse looks for: the second loop
/// can replay what the first read from thread-private registers instead of global memory.
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
fn cross_loop_reuse_annotations_reach_the_materialised_ir() {
    let c = KernelCompiler::new(kernel_ast, "kernels", "two_pass")
        .generics(vec!["32".into(), "4".into()])
        .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
        .compile()
        .expect("frontend compiles two_pass");

    // The pass marks the producing loop and the consuming loop. Finding either in the flat IR
    // proves the pass ran AND that materialisation carried its conclusion across.
    let annotated: Vec<&str> =
        c.ir.ops
            .iter()
            .flat_map(|o| o.attributes.iter().map(|(k, _)| k.as_str()))
            .filter(|k| k.starts_with("wax_cache_"))
            .collect();

    assert!(
        !annotated.is_empty(),
        "no `wax_cache_*` attribute in the materialised IR: the pass did not run, or its \
         conclusions were dropped on the way to the flat form"
    );
}
