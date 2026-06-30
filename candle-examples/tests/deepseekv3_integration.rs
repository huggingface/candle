//! End-to-end smoke test for the DeepSeek-V3 model against a tiny,
//! randomly-initialized checkpoint that uses the real DeepSeek-V3 architecture
//! (sigmoid noaux_tc MoE gate, MLA, YaRN RoPE) at a size small enough to
//! download and run anywhere (~9 MB), unlike the real 671B-parameter model.
//!
//! Network access required, so this is `#[ignore]`d by default. Run with:
//!   cargo test -p candle-examples --release --test deepseekv3_integration -- --ignored
#![cfg(not(target_arch = "wasm32"))]

use anyhow::Result;
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::deepseek3::{DeepSeekV3, DeepSeekV3Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "yujiepan/deepseek-v3-tiny-random";

#[test]
#[ignore]
fn deepseekv3_tiny_random_generates() -> Result<()> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        MODEL_ID.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let config: DeepSeekV3Config = {
        let config_file = repo.get("config.json")?;
        serde_json::from_slice(&std::fs::read(config_file)?)?
    };

    let filenames =
        match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
            Ok(filenames) => filenames,
            Err(_) => vec![repo.get("model.safetensors")?],
        };

    let device = candle_examples::device(false)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let mut model = DeepSeekV3::new(&config, vb)?;

    let prompt_ids = tokenizer
        .encode("The quick brown fox", true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    // Prefill: forward the whole prompt and check the logits are well-formed.
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    assert_eq!(logits.dims(), &[1, config.vocab_size]);
    let logits = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "prefill logits contain NaN/Inf"
    );
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
        .unwrap();
    assert!((next_token as usize) < config.vocab_size);

    // Decode step: forward a single new token using the KV cache populated above.
    let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
    let logits = model.forward(&next_input, prompt_ids.len())?;
    assert_eq!(logits.dims(), &[1, config.vocab_size]);
    let logits = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "decode-step logits contain NaN/Inf"
    );

    model.clear_kv_cache();
    Ok(())
}
