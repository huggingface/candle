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

// Summing over a non-trailing axis (or a strided view) can't use the contiguous
// vec_reduce_sum fast path, so it used to fall back to a plain in-dtype `+=` chain
// that still saturates for f16/bf16 even though the trailing-axis case above was fixed.
fn check_non_last_dim(dtype: DType) {
    let n = 4096usize;
    let t = Tensor::ones((n, 2), dtype, &Device::Cpu).unwrap();
    let s = t
        .sum(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(s, vec![n as f32, n as f32], "{dtype:?} column sum = {s:?}");
}

#[test]
fn f16_sum_non_last_dim_no_saturation() {
    check_non_last_dim(DType::F16);
}

#[test]
fn bf16_sum_non_last_dim_no_saturation() {
    check_non_last_dim(DType::BF16);
}

// A f16 sum can overflow to +inf well before the true mean would (f16's max finite
// value is 65504), so mean must scale down in f32 rather than dividing an
// already-narrowed, possibly-infinite f16 sum.
#[test]
fn f16_mean_all_overflowing_sum() {
    let n = 65520usize;
    let t = Tensor::ones((n,), DType::F16, &Device::Cpu).unwrap();
    let m = t
        .mean_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((m - 1.0).abs() < 1e-3, "f16 mean of {n} ones = {m}");
}
