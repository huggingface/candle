
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::{Ok, Result};
use candle::{test_device, Device, Tensor};
async fn convert(device: &Device) -> Result<()> {
    let vf32 = Tensor::arange(0f32, 4f32, device)?;
    let vf32_u32: Vec<u32> = vf32.to_dtype(candle::DType::U32)?.to_vec1_async().await?;
    assert_eq!(vf32_u32, [0u32, 1u32, 2u32, 3u32]);
    let vu32 = Tensor::new(vf32_u32, device)?;
    let vu32_f32: Vec<f32> = vu32.to_dtype(candle::DType::F32)?.to_vec1_async().await?;
    assert_eq!(vu32_f32, [0f32, 1f32, 2f32, 3f32]);
    let vu32_u8: Vec<u8> = vu32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vu32_u8, [0, 1, 2, 3]);
    let vf32_u8: Vec<u8> = vf32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vf32_u8, [0, 1, 2, 3]);
    let vu8 = vu32.to_dtype(candle::DType::U8)?;
    let vu8_f32: Vec<f32> = vu8.to_dtype(candle::DType::F32)?.to_vec1_async().await?;
    assert_eq!(vu8_f32, [0f32, 1f32, 2f32, 3f32]);
    Ok(())
}
async fn alloc(device: &Device) -> Result<()> {
    let t = 5.0f64;
    let ratio = (Tensor::ones(1, candle::DType::F32, device)? * t)?;
    assert_eq!(ratio.to_vec1_async::< f32 > (). await ?, [5f32]);
    let ratio = (Tensor::ones(1, candle::DType::U32, device)? * t)?;
    assert_eq!(ratio.to_vec1_async::< u32 > (). await ?, [5u32]);
    Ok(())
}
candle_wasm_tests::test_device!(
    convert, convert_cpu, convert_gpu, convert_metal, convert_wgpu
);
candle_wasm_tests::test_device!(alloc, alloc_cpu, alloc_gpu, alloc_metal, alloc_wgpu);
