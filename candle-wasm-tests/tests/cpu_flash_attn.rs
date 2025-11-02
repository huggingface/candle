#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{DType, Device, Result, Tensor};
use candle_nn::cpu_flash_attention::run_flash_attn_cpu;
#[test]
async fn cpu_flash_attn() -> Result<()> {
    let b = 1;
    let s = 2;
    let h = 1;
    let d = 4;
    let softmax_scale = 1.0f32 / (d as f32).sqrt();
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let ground_truth = {
        let att = (q.clone() * softmax_scale as f64)?.matmul(&k.clone().t()?)?;
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        att.matmul(&v.clone())?
    };
    let out = run_flash_attn_cpu::<
        f32,
    >(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        None,
        softmax_scale,
        None,
        None,
    )?;
    let out_arr: Vec<f32> = out.flatten_all()?.to_vec1_async().await?;
    let ground_truth_arr: Vec<f32> = ground_truth.flatten_all()?.to_vec1_async().await?;
    for (a, b) in out_arr.iter().zip(ground_truth_arr.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} {b}");
    }
    Ok(())
}
