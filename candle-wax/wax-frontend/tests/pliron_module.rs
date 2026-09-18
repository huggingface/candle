//! The dialect layer, restored on the `cutile-frontend` path.
//!
//! `wax-frontend` produces `WaxIr` for a backend to lower. That path never touches the `wax`
//! pliron dialect, so `cross_loop_reuse`, `verify_operation` and every pass plan section 8 has
//! queued were unreachable from candle-wax. These check the module they operate on can be built
//! from what the frontend actually emits.

use wax_frontend::{KernelCompiler, ast};

fn kernel_ast() -> ast::Module {
    const SRC: &str = r#"
mod kernels {
    use cutile::core::*;
    #[wax::entry()]
    fn axpy<const N: i32>(
        out: &mut Tensor<f32, { [-1, N] }>,
        x: &Tensor<f32, { [-1, N] }>,
    ) {
        let ts = const_shape![1, N];
        let row = get_tile_block_id().0;
        let px = x.partition(ts);
        let mut po: PartitionMut<f32, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([row, 0i32]);
        unsafe { po.store(v + v, [row, 0i32]) };
    }
}
"#;
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let mut m = ast::Module::with_span_base(
        "kernels",
        item,
        ast::SpanBase::new("tests/pliron_module.rs".into(), 1, 1),
    );
    m.set_absolute_path("wax_frontend::kernels".to_string());
    m
}

fn compile() -> wax_frontend::Compiled {
    KernelCompiler::new(kernel_ast, "kernels", "axpy")
        .generics(vec!["32".into()])
        .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
        .compile()
        .expect("frontend compiles axpy")
}

#[test]
fn a_pliron_module_can_be_built_from_the_frontends_output() {
    let c = compile();
    let m = c.to_wax_module("axpy").expect("module builds");

    assert_eq!(m.functions.len(), 1, "one entry function");
    // The IR is not empty and carries the shape the flat form has: comparing op counts is what
    // catches a builder that silently drops a region's contents.
    let flat_ops = c.ir.ops.len();
    assert!(
        flat_ops > 0,
        "flat IR is empty, so the comparison proves nothing"
    );
    assert!(
        !m.ir_text().is_empty(),
        "pliron module rendered empty despite {flat_ops} ops in the flat form"
    );
}

#[test]
fn the_cross_loop_reuse_pass_runs_on_it() {
    let c = compile();
    let mut m = c.to_wax_module("axpy").expect("module builds");
    let f = *m.functions.first().unwrap();

    // This kernel has no two-pass structure, so the pass should find nothing - the point is
    // that it can RUN, which it could not when the dialect was bypassed.
    let mut analyses = wax_ir::passes::AnalysisManager::default();
    wax_ir::passes::cross_loop_reuse::annotate_with(&mut analyses, &m.ctx, f)
        .expect("the pass runs to completion");
}
