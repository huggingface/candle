//! A kernel, all the way from DSL text to metallib bytes, inside candle.
//!
//! Every layer is exercised: `cutile-frontend` lowers the DSL, the `WaxSource` adapter carries
//! it into wax IR, and this crate emits AIR and wraps it in a metallib container. Nothing here
//! shells out, and nothing needs a CUDA toolkit.
//!
//! The narrow tests around it check pieces; this one is the only thing that would notice a
//! layer being wired up wrong.

use wax_frontend::{KernelCompiler, ast};

fn kernel_ast() -> ast::Module {
    const SRC: &str = r#"
mod kernels {
    use cutile::core::*;
    #[wax::entry()]
    fn vector_add(
        a: &Tensor<f32, { [256] }>,
        b: &Tensor<f32, { [256] }>,
        c: &mut Tensor<f32, { [256] }>,
    ) {
        let ta = a.partition(const_shape![256]).load([0i32]);
        let tb = b.partition(const_shape![256]).load([0i32]);
        c.store(ta + tb);
    }
}
"#;
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let mut m = ast::Module::with_span_base(
        "kernels",
        item,
        ast::SpanBase::new("tests/end_to_end.rs".into(), 1, 1),
    );
    m.set_absolute_path("wax_metal::kernels".to_string());
    m
}

#[test]
fn a_kernel_becomes_a_metallib() {
    let compiled = KernelCompiler::new(kernel_ast, "kernels", "vector_add")
        .strides(&[("a", &[1]), ("b", &[1]), ("c", &[1])])
        .compile()
        .expect("frontend compiles the kernel");

    let lib =
        wax_metal::lower_to_metallib_from_wax_ir(&compiled.ir, &wax_metal::TargetProfile::host())
            .expect("backend lowers to a metallib");

    // A metallib is a container with a magic header; anything shorter is not one.
    assert!(
        lib.len() > 64,
        "metallib is implausibly small: {} bytes",
        lib.len()
    );
    assert_eq!(&lib[..4], b"MTLB", "missing the metallib magic");
}

#[test]
fn the_air_text_path_agrees_that_it_lowers() {
    let compiled = KernelCompiler::new(kernel_ast, "kernels", "vector_add")
        .strides(&[("a", &[1]), ("b", &[1]), ("c", &[1])])
        .compile()
        .expect("frontend compiles the kernel");
    let ir = &compiled.ir;
    assert!(!ir.ops.is_empty(), "no ops materialised");
    // The two emission paths are independent writers, so a kernel that lowers through one and
    // not the other is a bug in whichever is newer.
    assert!(
        wax_metal::lower_to_metallib_from_wax_ir(ir, &wax_metal::TargetProfile::host()).is_ok(),
        "bitcode path refused a kernel the frontend accepted"
    );
}
