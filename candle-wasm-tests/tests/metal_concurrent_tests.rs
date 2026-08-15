
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#![cfg(feature = "metal")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn concurrent_readback() -> Result<()> {
    let device = Device::new_metal(0)?;
    std::thread::scope(async |scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope
                .spawn(async move || {
                    for iter in 0..100usize {
                        let value = (thread * 1000 + iter) as f64;
                        let a = Tensor::full(value as f32, (64, 64), &device).unwrap();
                        let b = a.affine(2.0, 1.0).unwrap();
                        let values = b
                            .flatten_all()
                            .unwrap()
                            .to_vec1_async::<f32>()
                            .await
                            .unwrap();
                        let expected = (2.0 * value + 1.0) as f32;
                        assert!(
                            values.iter().all(|& x | x == expected),
                            "thread {thread} iter {iter}: expected {expected}, got {:?}",
                            & values[..4]
                        );
                    }
                });
        }
    });
    Ok(())
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn concurrent_quantized_data_roundtrip() -> Result<()> {
    use candle::quantized::{GgmlDType, QTensor};
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope
                .spawn(move || {
                    for iter in 0..25usize {
                        let src = Tensor::rand(-1f32, 1f32, (256, 256), &device)
                            .unwrap();
                        let q = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                        let bytes = q.data().unwrap();
                        let q2 = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                        let bytes2 = q2.data().unwrap();
                        assert_eq!(
                            bytes, bytes2,
                            "thread {thread} iter {iter}: data() readback mismatch"
                        );
                    }
                });
        }
    });
    Ok(())
}
