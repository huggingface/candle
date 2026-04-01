// Starcoder2 text generation using AutoModelForCausalLM.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use candle::{DType, Tensor};
use candle_transformers::auto::{AutoModelForCausalLM, AutoModelOptions};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// HuggingFace model repo ID.
    #[arg(long, default_value = "bigcode/starcoder2-3b")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    prompt: Option<String>,

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

    println!("loading {}", args.model_id);
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let tokenizer =
        Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(anyhow::Error::msg)?;

    let options = AutoModelOptions {
        revision: Some(args.revision),
        dtype: Some(dtype),
        use_flash_attn: args.use_flash_attn,
        ..Default::default()
    };
    let mut model = AutoModelForCausalLM::from_pretrained(&args.model_id, &device, options)?;
    println!("loaded (type: {})", model.model_type());

    let prompt = args.prompt.as_deref().unwrap_or("def fibonacci(");
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
        "{token_generated} tokens ({:.2} token/s)",
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
