#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Result, bail};
use candle::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    SmolLM135M,
}

#[derive(Parser)]
struct Args {
    /// The prompt to be used.
    #[arg(long, default_value = "My favorite theorem is ")]
    prompt: String,

    /// Path to a local ONNX model file.
    #[arg(long)]
    model: Option<String>,

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
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    
    let (model_id, tokenizer_id) = match args.which {
        Which::SmolLM135M => ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-135M"),
    };
    
    let api = Api::new()?;
    let model_repo = api.model(model_id.to_string());
    let tokenizer_repo = api.model(tokenizer_id.to_string());
    
    let model_path = match &args.model {
        Some(path) => std::path::PathBuf::from(path),
        None => model_repo.get("onnx/model.onnx")?,
    };
    
    let tokenizer_path = tokenizer_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    
    // Get tokens and convert them to i64
    let tokens_u32 = tokenizer.encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    
    let tokens: Vec<i64> = tokens_u32.iter().map(|&t| t as i64).collect();
    
    println!("Loading ONNX model from {:?}", model_path);
    let model = candle_onnx::read_file(model_path)?;
    let graph = model.graph.as_ref().unwrap();
    
    let mut generated_tokens = tokens.clone();
    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    
    // State for maintaining past key values between generations
    let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
    // Number of layers in the model (based on the input document, there are 36 layers)
    let num_layers = 30;
    
    for _ in 0..args.max_tokens {
        let mut inputs = std::collections::HashMap::new();
        
        if let Some(past_kv) = &past_key_values {
            // If we have past_key_values, only process the last token
            let last_token = vec![generated_tokens[generated_tokens.len() - 1]];
            let input_tensor = Tensor::new(last_token, &device)?.unsqueeze(0)?;
            inputs.insert("input_ids".to_string(), input_tensor);
            
            // Add attention mask for all tokens seen so far
            let seq_len = generated_tokens.len();
            let attention_mask = vec![vec![1i64; seq_len]];
            let attention_mask_tensor = Tensor::new(attention_mask, &device)?;
            inputs.insert("attention_mask".to_string(), attention_mask_tensor);
            
            // Add position_ids for the current token
            // Position IDs are usually just the indices of the tokens in the sequence
            let position_ids = vec![vec![(seq_len - 1) as i64]];
            let position_ids_tensor = Tensor::new(position_ids, &device)?;
            inputs.insert("position_ids".to_string(), position_ids_tensor);
            
            // Add past_key_values
            for (i, (key, value)) in past_kv.iter().enumerate() {
                inputs.insert(format!("past_key_values.{}.key", i), key.clone());
                inputs.insert(format!("past_key_values.{}.value", i), value.clone());
            }
        } else {
            // First pass, process all tokens without past_key_values
            let input_tensor = Tensor::new(generated_tokens.clone(), &device)?.unsqueeze(0)?;
            inputs.insert("input_ids".to_string(), input_tensor);
            
            // Add attention mask (all 1s for the current sequence)
            let seq_len = generated_tokens.len();
            let attention_mask = vec![vec![1i64; seq_len]];
            let attention_mask_tensor = Tensor::new(attention_mask, &device)?;
            inputs.insert("attention_mask".to_string(), attention_mask_tensor);
            
            // Add position_ids (indices of the sequence)
            let position_ids: Vec<i64> = (0..seq_len as i64).collect();
            let position_ids_tensor = Tensor::new(position_ids, &device)?.unsqueeze(0)?;
            inputs.insert("position_ids".to_string(), position_ids_tensor);
            
            // Initialize empty past_key_values if needed by the model
            // This is only for the first pass; most models don't require this
            for i in 0..num_layers {
                // Create empty tensors with the right shape for keys and values
                // The shape depends on your model architecture
                let batch_size = 1;
                let num_heads = 3; // Adjust based on your model
                let head_dim = 64; // Adjust based on your model
                let seq_len = 0;    // Empty sequence for first run
                
                // Create empty key and value tensors
                let empty_key = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], DType::F32, &device)?;
                let empty_value = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], DType::F32, &device)?;
                
                inputs.insert(format!("past_key_values.{}.key", i), empty_key);
                inputs.insert(format!("past_key_values.{}.value", i), empty_value);
            }
        }
        
        // Perform inference
        let outputs = candle_onnx::simple_eval(&model, inputs)?;
        
        // Extract logits and present KV cache for next iteration
        if !outputs.contains_key("logits") {
            bail!("Model output doesn't contain 'logits'");
        }
        
        let logits = outputs.get("logits").unwrap();
        
        // Update past_key_values for next iteration if present in outputs
        if outputs.contains_key("present.0.key") {
            // Extract present key values for all layers
            let mut new_past_kv = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let key = outputs.get(&format!("present.{}.key", i))
                    .ok_or_else(|| anyhow::anyhow!("Missing present.{}.key", i))?;
                let value = outputs.get(&format!("present.{}.value", i))
                    .ok_or_else(|| anyhow::anyhow!("Missing present.{}.value", i))?;
                new_past_kv.push((key.clone(), value.clone()));
            }
            past_key_values = Some(new_past_kv);
        }
        
        // Get the last token's logits
        let logits_dim = logits.dims();
        let seq_len = logits_dim[1]; // Assuming shape is [batch, seq_len, vocab]
        let last_logits = logits.get(0)?.get(seq_len - 1)?;
        
        // Apply temperature
        let logits_with_temp = if args.temperature > 0.0 {
            (args.temperature.recip() as f64 * &last_logits)?
        } else {
            last_logits.clone()
        };
        
        // Apply softmax and sample
        let probs = candle_nn::ops::softmax(&logits_with_temp, 0)?;
        let next_token = probs.argmax(0)?;
        
        // Convert back to i64 for storing in our generated tokens vector
        let next_token_id = next_token.to_scalar::<u32>()?;
        generated_tokens.push(next_token_id as i64);
        
        // Convert to u32 for tokenizer (which expects u32)
        let next_token_u32 = next_token_id;
        
        if let Some(token_str) = tokenizer.decode(&[next_token_u32], false).ok() {
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }
        
        // Check for EOS token (need to use the u32 version for comparison with tokenizer output)
        if let Some(eos_id) = tokenizer.token_to_id("</s>") {
            if next_token_u32 == eos_id {
                break;
            }
        }
    }
    
    println!("\nGeneration complete!");
    Ok(())
}