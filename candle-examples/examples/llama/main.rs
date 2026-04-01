// LLaMA text generation using AutoModelForCausalLM.
//
// All LLaMA variants (v1/v2/v3/v3.1/v3.2, SmolLM2, Solar, TinyLlama) load
// through the unified auto-model interface — no variant-specific code needed.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use candle::{DType, Tensor};
use candle_transformers::auto::{AutoModelForCausalLM, AutoModelOptions};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

const DEFAULT_PROMPT: &str = "My favourite theorem is ";

/// Convenience shortcuts for common LLaMA model IDs.
/// Pass `--model-id <HF_REPO>` to use any other repo directly.
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V3Instruct,
    V31,
    V31Instruct,
    #[value(name = "v3.2-1b")]
    V32_1b,
    #[value(name = "v3.2-1b-instruct")]
    V32_1bInstruct,
    #[value(name = "v3.2-3b")]
    V32_3b,
    #[value(name = "v3.2-3b-instruct")]
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "smollm2-1.7b")]
    SmolLM2_1B,
    #[value(name = "smollm2-1.7b-instruct")]
    SmolLM2_1BInstruct,
    #[value(name = "smollm2-360m")]
    SmolLM2_360M,
    #[value(name = "smollm2-360m-instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "smollm2-135m")]
    SmolLM2_135M,
    #[value(name = "smollm2-135m-instruct")]
    SmolLM2_135MInstruct,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::V1 => "Narsil/amall-7b",
            Self::V2 => "meta-llama/Llama-2-7b-hf",
            Self::V3 => "meta-llama/Meta-Llama-3-8B",
            Self::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
            Self::V31 => "meta-llama/Llama-3.1-8B",
            Self::V31Instruct => "meta-llama/Llama-3.1-8B-Instruct",
            Self::V32_1b => "meta-llama/Llama-3.2-1B",
            Self::V32_1bInstruct => "meta-llama/Llama-3.2-1B-Instruct",
            Self::V32_3b => "meta-llama/Llama-3.2-3B",
            Self::V32_3bInstruct => "meta-llama/Llama-3.2-3B-Instruct",
            Self::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
            Self::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Self::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B",
            Self::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            Self::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
            Self::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
            Self::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
            Self::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "v3")]
    which: Which,

    /// Override the HF repo ID (takes precedence over --which).
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    /// Run on CPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "bf16")]
    dtype: String,

    #[arg(long, default_value_t = 200)]
    sample_len: usize,

    #[arg(long)]
    temperature: Option<f64>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

    #[arg(long)]
    use_flash_attn: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        other => bail!("unsupported dtype {other}"),
    };

    let model_id = args
        .model_id
        .unwrap_or_else(|| args.which.model_id().to_string());
    let revision = args.revision.unwrap_or_else(|| "main".to_string());

    println!("loading model {model_id} (revision: {revision})");

    // Load tokenizer
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.clone(),
        RepoType::Model,
        revision.clone(),
    ));
    let tokenizer =
        Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(anyhow::Error::msg)?;

    // Load model — one call regardless of LLaMA version
    let options = AutoModelOptions {
        revision: Some(revision),
        dtype: Some(dtype),
        use_flash_attn: args.use_flash_attn,
        ..Default::default()
    };
    let mut model = AutoModelForCausalLM::from_pretrained(&model_id, &device, options)?;
    println!("model loaded (type: {})", model.model_type());

    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut token_output = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    let eos_token_id = token_output
        .tokenizer()
        .token_to_id("</s>")
        .or_else(|| token_output.tokenizer().token_to_id("<|end_of_text|>"));

    let mut logits_processor = {
        let temperature = args.temperature.unwrap_or(0.);
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

    print!("{prompt}");
    let start = std::time::Instant::now();
    let mut seqlen_offset = 0usize;
    let mut token_generated = 0usize;

    for _ in 0..args.sample_len {
        let context_size = if seqlen_offset > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, seqlen_offset)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        seqlen_offset += context_size;
        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);
        if Some(next_token) == eos_token_id {
            break;
        }
        if let Some(t) = token_output.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = token_output.decode_rest().map_err(anyhow::Error::msg)? {
        print!("{rest}");
    }
    println!();
    let dt = start.elapsed();
    println!(
        "{token_generated} tokens generated ({:.2} token/s)",
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
