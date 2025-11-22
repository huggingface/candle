
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};
#[test]
async fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(
        Tensor::cat(&[&w, &w], 0)?,
        Tensor::cat(&[&b, &b], 0)?,
        1e-8,
    );
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1_async::< f32 > (). await ?, [0.5f32]);
    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3_async::< f32 > (). await ?, [[[3.5f32, - 2.5]]]);
    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        to_vec3_round_async(& res, 4). await ?, [[[- 3.1742, 0.5, 4.1742], [- 3.1742,
        0.5, 4.1742], [4.1742, 0.5, - 3.1742]]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    assert_eq!(to_vec3_round_async(& mean, 4). await ?, [[[0.5], [0.5], [0.5]]]);
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    assert_eq!(to_vec3_round_async(& std, 4). await ?, [[[1.7321], [1.7321], [1.7321]]]);
    Ok(())
}
