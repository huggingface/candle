use candle_core::{DType, Device, Tensor};

// Global average pooling sums a large window; a narrow-dtype accumulator
// saturates, so the average of an all-ones map comes out far too small.
fn check(dtype: DType) {
    let dev = Device::Cpu;
    let (h, w) = (64usize, 64usize); // 4096-element window
    let t = Tensor::ones((1, 1, h, w), dtype, &dev).unwrap();
    let pooled = t.avg_pool2d((h, w)).unwrap();
    let v = pooled
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .get(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        (v - 1.0).abs() < 1e-2,
        "{dtype:?} global avg_pool of ones = {v}"
    );
}

#[test]
fn f16_global_avg_pool_no_saturation() {
    check(DType::F16);
}

#[test]
fn bf16_global_avg_pool_no_saturation() {
    check(DType::BF16);
}
