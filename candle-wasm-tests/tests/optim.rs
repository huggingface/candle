
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
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
use candle::test_utils::{to_vec0_round, to_vec2_round};
use anyhow::Result;
use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, SGD};
#[test]
async fn sgd_optim() -> Result<()> {
    let x = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![x.clone()], 0.1)?;
    let xt = x.as_tensor();
    for _step in 0..100 {
        let loss = ((xt - 4.2)? * (xt - 4.2)?)?;
        sgd.backward_step(&loss)?
    }
    assert_eq!(x.to_scalar_async::< f32 > (). await ?, 4.199999);
    Ok(())
}
#[test]
async fn sgd_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![w.clone(), b.clone()], 0.004)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        sgd.backward_step(&loss)?;
    }
    assert_eq!(w.to_vec2_async::< f32 > (). await ?, & [[2.9983196, 0.99790204]]);
    assert_eq!(b.to_scalar_async::< f32 > (). await ?, - 1.9796902);
    Ok(())
}
#[test]
async fn adamw_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round_async(w.as_tensor(), 4). await ?, & [[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round_async(b.as_tensor(), 4). await ?, 0.7873);
    Ok(())
}
#[test]
async fn adamw_linear_regression_varmap() -> Result<()> {
    use candle_nn::Init::Const;
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let mut var_map = candle_nn::VarMap::new();
    let w = var_map.get((1, 2), "w", Const(0.), DType::F32, &Device::Cpu)?;
    let b = var_map.get((), "b", Const(0.), DType::F32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(var_map.all_vars(), params)?;
    let lin = Linear::new(w, Some(b));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round_async(lin.weight(), 4). await ?, & [[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round_async(lin.bias().unwrap(), 4). await ?, 0.7873);
    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;
    assert_eq!(to_vec2_round_async(lin.weight(), 4). await ?, & [[0., 0.]]);
    assert_eq!(to_vec0_round_async(lin.bias().unwrap(), 4). await ?, 1.);
    Ok(())
}
