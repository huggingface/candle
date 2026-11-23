use candle_core::quantized::{GgmlDType, QStorage, QTensor};

#[test]
fn qtensor_new_rejects_storage_shape_mismatch() {
    // Storage holds one Q4_0 block (32 elements) but the shape claims 64
    // elements (2 blocks). check_shape only checks last-dim % block_size, so
    // this used to construct fine and then read out of bounds in dequantize.
    let storage = QStorage::Cpu(GgmlDType::Q4_0.cpu_zeros(32));
    let res = QTensor::new(storage, (64,));
    assert!(res.is_err(), "expected Err from storage/shape mismatch");
}
