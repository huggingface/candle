//! End-to-end example: download a GPTQ-quantized Qwen2 checkpoint from the Hugging Face Hub
//! (e.g. `Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4`) and run text generation through the fused/CPU
//! GPTQ kernels in `candle-gptq-kernels` / `candle_transformers::quantized_gptq`.
//!
//! The checkpoint's `config.json` carries a `quantization_config` block (`bits`, `group_size`,
//! ...) produced by AutoGPTQ/GPTQModel; this example reads it to build a
//! [`candle_transformers::quantized_gptq::GptqConfig`] and constructs the model with
//! [`candle_transformers::models::gptq_qwen2::ModelForCausalLM`], which mirrors the dense Qwen2
//! model but routes every attention/MLP projection through the GPTQ dequantized linear layer.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gptq_qwen2::{Config, ModelForCausalLM};
use candle_transformers::quantized_gptq::GptqConfig;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(serde::Deserialize)]
struct QuantizationConfig {
    bits: usize,
    group_size: usize,
}

#[derive(serde::Deserialize)]
struct ConfigFile {
    quantization_config: QuantizationConfig,
}

struct TextGeneration {
    model: ModelForCausalLM,
    device: candle::Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &candle::Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 1000)]
    sample_len: usize,

    /// The GPTQ-quantized model repo on the Hugging Face Hub.
    #[arg(long, default_value = "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Skip chat template formatting (use the raw prompt, like a base model).
    #[arg(long)]
    no_chat_template: bool,
}

fn format_prompt(prompt: &str, use_chat_template: bool) -> String {
    if !use_chat_template {
        return prompt.to_string();
    }
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    // GPTQ exports on the Hub are typically a single `model.safetensors` file (no shard index).
    let weights_filename = if repo.get("model.safetensors.index.json").is_ok() {
        candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
    } else {
        vec![repo.get("model.safetensors")?]
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config_bytes = std::fs::read(&config_filename)?;
    let config: Config = serde_json::from_slice(&config_bytes)?;
    let quant_config: ConfigFile = serde_json::from_slice(&config_bytes)?;
    let gptq_config = GptqConfig {
        bits: quant_config.quantization_config.bits,
        group_size: quant_config.quantization_config.group_size,
    };

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    // The packed `qweight`/`qzeros`/`g_idx` tensors are always read as i32 regardless of this
    // dtype (see `gptq_linear`); this only controls the dense embed/norm/lm_head tensors and the
    // dequantized GPTQ weights, which are converted to it.
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filename, dtype, &device)? };
    let model = ModelForCausalLM::new(&config, gptq_config, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    let prompt = format_prompt(&args.prompt, !args.no_chat_template);
    pipeline.run(&prompt, args.sample_len)?;
    Ok(())
}
