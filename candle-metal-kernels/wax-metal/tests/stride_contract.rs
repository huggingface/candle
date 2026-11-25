mod common;

use wax_frontend::{KernelCompiler, ast};

const SRC: &str = r#"
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn add(
        out: &mut Tensor<f32, { [2, 128] }>,
        x: &Tensor<f32, { [2, 128] }>,
        y: &Tensor<f32, { [2, 128] }>,
    ) {
        let ts = const_shape![2, 128];
        out.store(x.partition(ts).load([0i32, 0i32]) + y.partition(ts).load([0i32, 0i32]));
    }

    #[wax_macro::entry()]
    pub fn add_dyn<const K: i32>(
        out: &mut Tensor<f32, { [2, K] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let ts = const_shape![2, K];
        out.store(x.partition(ts).load([0i32, 0i32]) + y.partition(ts).load([0i32, 0i32]));
    }
}
"#;

fn compile(entry: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> Result<(), String> {
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let span_base = ast::SpanBase::new("tests/stride_contract.rs".into(), 1, 1);
    KernelCompiler::new(
        move || ast::Module::with_span_base("kernels", item.clone(), span_base.clone()),
        "kernels",
        entry,
    )
    .generics(generics.iter().map(|g| g.to_string()).collect())
    .strides(strides)
    .compile()?
    .to_wax_module(entry)
    .map(|_| ())
}

#[test]
fn dense_strides_are_accepted() {
    compile(
        "add",
        &[],
        &[("out", &[128, 1]), ("x", &[128, 1]), ("y", &[128, 1])],
    )
    .expect("dense row-major strides describe distinct elements");
}

#[test]
fn padded_strides_are_accepted() {
    compile(
        "add",
        &[],
        &[("out", &[256, 1]), ("x", &[256, 1]), ("y", &[256, 1])],
    )
    .expect("a view into a wider tensor is injective, just not dense");
}

#[test]
fn aliasing_strides_are_rejected() {
    let err = compile(
        "add",
        &[],
        &[("out", &[1, 1]), ("x", &[1, 1]), ("y", &[1, 1])],
    )
    .expect_err("stride 1 on both axes maps (0,1) and (1,0) to the same element");
    assert!(
        err.contains("aliases") && err.contains("axis"),
        "diagnostic should name the aliasing axis, got: {err}"
    );
}

#[test]
fn dynamic_extents_defer_to_launch() {
    compile(
        "add_dyn",
        &["64"],
        &[("out", &[64, 1]), ("x", &[64, 1]), ("y", &[32, 1])],
    )
    .expect("an unresolved extent is a launch-time obligation, not a compile-time failure");
}
