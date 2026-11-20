use candle::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;

// f16/bf16 softmax must accumulate the denominator in f32; otherwise the
// in-dtype sum saturates on a long axis and the row fails to normalize.
fn check_row_sums_to_one(dtype: DType) {
    let dev = Device::Cpu;
    let n = 4096usize;
    let t = Tensor::zeros((1, n), dtype, &dev).unwrap();
    let sm = softmax_last_dim(&t).unwrap();
    // Measure in f32 so the check itself does not saturate.
    let row_sum = sm
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        (row_sum - 1.0).abs() < 1e-2,
        "{dtype:?} softmax row sum {row_sum} != 1.0"
    );
}

#[test]
fn f16_softmax_row_sums_to_one() {
    check_row_sums_to_one(DType::F16);
}

#[test]
fn bf16_softmax_row_sums_to_one() {
    check_row_sums_to_one(DType::BF16);
}
