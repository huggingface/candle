mod common;

use wax_metal::LowerError;

const SRC: &str = r#"
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn add2048(
        out: &mut Tensor<f32, { [1, 2048] }>,
        x: &Tensor<f32, { [1, 2048] }>,
        y: &Tensor<f32, { [1, 2048] }>,
    ) {
        let ts = const_shape![1, 2048];
        out.store(x.partition(ts).load([0i32, 0i32]) + y.partition(ts).load([0i32, 0i32]));
    }

    #[wax_macro::entry()]
    pub fn add1024(
        out: &mut Tensor<f32, { [1, 1024] }>,
        x: &Tensor<f32, { [1, 1024] }>,
        y: &Tensor<f32, { [1, 1024] }>,
    ) {
        let ts = const_shape![1, 1024];
        out.store(x.partition(ts).load([0i32, 0i32]) + y.partition(ts).load([0i32, 0i32]));
    }

    #[wax_macro::entry()]
    pub fn add256(
        out: &mut Tensor<f32, { [2, 128] }>,
        x: &Tensor<f32, { [2, 128] }>,
        y: &Tensor<f32, { [2, 128] }>,
    ) {
        let ts = const_shape![2, 128];
        out.store(x.partition(ts).load([0i32, 0i32]) + y.partition(ts).load([0i32, 0i32]));
    }
}
"#;

fn lower(
    entry: &str,
    strides: &'static [(&'static str, &'static [i32])],
) -> Result<wax_metal::lower::Lowered, LowerError> {
    let mut m = common::module_from_src(
        SRC,
        "tests/launch_geometry.rs",
        "kernels",
        entry,
        &[],
        strides,
    );
    wax_metal::lower::lower_to_metallib_for(&mut m, &common::profile())
}

const S2048: &[(&str, &[i32])] = &[("out", &[2048, 1]), ("x", &[2048, 1]), ("y", &[2048, 1])];
const S1024: &[(&str, &[i32])] = &[("out", &[1024, 1]), ("x", &[1024, 1]), ("y", &[1024, 1])];
const S128: &[(&str, &[i32])] = &[("out", &[128, 1]), ("x", &[128, 1]), ("y", &[128, 1])];

#[test]
fn tile_over_the_threadgroup_ceiling_is_refused() {
    let err = lower("add2048", S2048).expect_err("2048 threads exceed a threadgroup");
    assert!(
        matches!(err, LowerError::ExceedsTarget(_)),
        "wrong variant: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("2048") && msg.contains("1024"),
        "diagnostic should name both counts, got: {msg}"
    );
}

#[test]
fn tile_at_the_threadgroup_ceiling_is_accepted() {
    let got = lower("add1024", S1024).expect("1024 threads is exactly one threadgroup");
    assert_eq!(got.entry_threads(0), Some(1024));
}

#[test]
fn lowering_reports_the_thread_count() {
    let got = lower("add256", S128).expect("lowers");
    assert_eq!(got.entry_threads(0), Some(256));
    assert!(!got.metallib.is_empty());
}
