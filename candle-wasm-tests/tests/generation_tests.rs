#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
#[test]
async fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    Ok(())
}
#[test]
async fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 0);
    Ok(())
}
#[test]
async fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 2);
    Ok(())
}
#[test]
async fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 2);
    Ok(())
}
