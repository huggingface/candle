use candle_core::{DType, Device, Tensor};

// A sequential in-dtype sum of f16/bf16 saturates on a long axis; sum/mean must
// accumulate in f32 so the result stays correct (and matches the Metal backend).
fn check(dtype: DType) {
    let n = 4096usize;
    let t = Tensor::ones((n,), dtype, &Device::Cpu).unwrap();
    let s = t
        .sum_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(s, n as f32, "{dtype:?} sum of {n} ones = {s}");
    let m = t
        .mean_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((m - 1.0).abs() < 1e-3, "{dtype:?} mean of ones = {m}");
}

#[test]
fn f16_sum_mean_no_saturation() {
    check(DType::F16);
}

#[test]
fn bf16_sum_mean_no_saturation() {
    check(DType::BF16);
}
