#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    SmolLM135M,
}

#[derive(Parser)]
struct Args {
    /// The prompt to be used.
    #[arg(long, default_value = "My favorite theorem is ")]
    prompt: String,

    /// The model to be used.
    #[arg(value_enum, long, default_value_t = Which::SmolLM135M)]
    which: Which,

    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// The number of tokens to generate.
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// The temperature used for sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let (model_id, tokenizer_id) = match args.which {
        Which::SmolLM135M => ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-135M"),
    };

    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let tokenizer_repo = api.model(tokenizer_id.to_string());

    let model_path = model_repo.get("onnx/model.onnx")?;
    let config_file = model_repo.get("config.json")?;
    let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;

    let tokenizer_path = tokenizer_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let tokens_u32 = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let tokens: Vec<i64> = tokens_u32.iter().map(|&t| t as i64).collect();

    println!("Loading ONNX model from {:?}", model_path);
    let model = candle_onnx::read_file(model_path)?;

    let mut generated_tokens = tokens.clone();
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let mut logits_processor = {
        let temperature = args.temperature as f64;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
    let num_layers = config.num_hidden_layers;

    for _ in 0..args.max_tokens {
        let mut inputs = std::collections::HashMap::new();

        if let Some(past_kv) = &past_key_values {
            let last_token = vec![generated_tokens[generated_tokens.len() - 1]];
            let input_tensor = Tensor::new(last_token, &device)?.unsqueeze(0)?;
            inputs.insert("input_ids".to_string(), input_tensor);

            let seq_len = generated_tokens.len();
            let attention_mask = vec![vec![1i64; seq_len]];
            let attention_mask_tensor = Tensor::new(attention_mask, &device)?;
            inputs.insert("attention_mask".to_string(), attention_mask_tensor);

            let position_ids = vec![vec![(seq_len - 1) as i64]];
            let position_ids_tensor = Tensor::new(position_ids, &device)?;
            inputs.insert("position_ids".to_string(), position_ids_tensor);

            for (i, (key, value)) in past_kv.iter().enumerate() {
                inputs.insert(format!("past_key_values.{}.key", i), key.clone());
                inputs.insert(format!("past_key_values.{}.value", i), value.clone());
            }
        } else {
            let input_tensor = Tensor::new(generated_tokens.clone(), &device)?.unsqueeze(0)?;
            inputs.insert("input_ids".to_string(), input_tensor);

            let seq_len = generated_tokens.len();
            let attention_mask = vec![vec![1i64; seq_len]];
            let attention_mask_tensor = Tensor::new(attention_mask, &device)?;
            inputs.insert("attention_mask".to_string(), attention_mask_tensor);

            let position_ids: Vec<i64> = (0..seq_len as i64).collect();
            let position_ids_tensor = Tensor::new(position_ids, &device)?.unsqueeze(0)?;
            inputs.insert("position_ids".to_string(), position_ids_tensor);

            // Create empty key and value tensors
            for i in 0..num_layers {
                let batch_size = 1;
                let num_heads = config.num_key_value_heads;
                let head_dim = config.hidden_size / config.num_attention_heads;
                let seq_len = 0;

                let empty_key = Tensor::zeros(
                    &[batch_size, num_heads, seq_len, head_dim],
                    DType::F32,
                    &device,
                )?;
                let empty_value = Tensor::zeros(
                    &[batch_size, num_heads, seq_len, head_dim],
                    DType::F32,
                    &device,
                )?;

                inputs.insert(format!("past_key_values.{}.key", i), empty_key);
                inputs.insert(format!("past_key_values.{}.value", i), empty_value);
            }
        }

        let outputs = candle_onnx::simple_eval(&model, inputs)?;

        let logits = outputs.get("logits").unwrap();

        let mut new_past_kv = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let key = outputs
                .get(&format!("present.{}.key", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present.{}.key", i))?;
            let value = outputs
                .get(&format!("present.{}.value", i))
                .ok_or_else(|| anyhow::anyhow!("Missing present.{}.value", i))?;
            new_past_kv.push((key.clone(), value.clone()));
        }
        past_key_values = Some(new_past_kv);

        let logits_dim = logits.dims();
        let seq_len = logits_dim[1];

        let next_token_id = logits_processor.sample(&logits.get(0)?.get(seq_len - 1)?)?;
        generated_tokens.push(next_token_id as i64);

        if let Some(token_str) = tokenizer.decode(&[next_token_id], true).ok() {
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }

        if let Some(eos_id) = tokenizer.token_to_id("<|endoftext|>") {
            if next_token_id == eos_id {
                break;
            }
        }
    }

    println!("\nGeneration complete!");
    Ok(())
}
