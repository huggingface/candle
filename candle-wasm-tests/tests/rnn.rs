
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
use candle::{test_utils::to_vec2_round, DType, Device, Result, Tensor};
use candle_nn::RNN;
#[test]
async fn lstm() -> Result<()> {
    let cpu = &Device::Cpu;
    let w_ih = Tensor::arange(0f32, 24f32, cpu)?.reshape((12, 2))?;
    let w_ih = w_ih.cos()?;
    let w_hh = Tensor::arange(0f32, 36f32, cpu)?.reshape((12, 3))?;
    let w_hh = w_hh.sin()?;
    let b_ih = Tensor::new(
        &[-1f32, 1., -0.5, 2., -1., 1., -0.5, 2., -1., 1., -0.5, 2.],
        cpu,
    )?;
    let b_hh = b_ih.cos()?;
    let tensors: std::collections::HashMap<_, _> = [
        ("weight_ih_l0".to_string(), w_ih),
        ("weight_hh_l0".to_string(), w_hh),
        ("bias_ih_l0".to_string(), b_ih),
        ("bias_hh_l0".to_string(), b_hh),
    ]
        .into_iter()
        .collect();
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, cpu);
    let lstm = candle_nn::lstm(2, 3, Default::default(), vb)?;
    let mut state = lstm.zero_state(1)?;
    for inp in [3f32, 1., 4., 1., 5., 9., 2.] {
        let inp = Tensor::new(&[[inp, inp * 0.5]], cpu)?;
        state = lstm.step(&inp, &state)?;
    }
    let h = state.h();
    let c = state.c();
    assert_eq!(to_vec2_round_async(h, 4). await ?, & [[0.9919, 0.1738, - 0.1451]]);
    assert_eq!(to_vec2_round_async(c, 4). await ?, & [[5.725, 0.4458, - 0.2908]]);
    Ok(())
}
#[test]
async fn gru() -> Result<()> {
    let cpu = &Device::Cpu;
    let w_ih = Tensor::arange(0f32, 18f32, cpu)?.reshape((9, 2))?;
    let w_ih = w_ih.cos()?;
    let w_hh = Tensor::arange(0f32, 27f32, cpu)?.reshape((9, 3))?;
    let w_hh = w_hh.sin()?;
    let b_ih = Tensor::new(&[-1f32, 1., -0.5, 2., -1., 1., -0.5, 2., -1.], cpu)?;
    let b_hh = b_ih.cos()?;
    let tensors: std::collections::HashMap<_, _> = [
        ("weight_ih_l0".to_string(), w_ih),
        ("weight_hh_l0".to_string(), w_hh),
        ("bias_ih_l0".to_string(), b_ih),
        ("bias_hh_l0".to_string(), b_hh),
    ]
        .into_iter()
        .collect();
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, cpu);
    let gru = candle_nn::gru(2, 3, Default::default(), vb)?;
    let mut state = gru.zero_state(1)?;
    for inp in [3f32, 1., 4., 1., 5., 9., 2.] {
        let inp = Tensor::new(&[[inp, inp * 0.5]], cpu)?;
        state = gru.step(&inp, &state)?;
    }
    let h = state.h();
    assert_eq!(to_vec2_round_async(h, 4). await ?, & [[0.0579, 0.8836, - 0.9991]]);
    Ok(())
}
