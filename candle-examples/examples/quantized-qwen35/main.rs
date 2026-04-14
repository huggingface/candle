#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use std::io::Write;
use tokenizers::Tokenizer;

use candle::quantized::gguf_file;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen35::ModelWeights as Qwen35;

const DEFAULT_PROMPT: &str = "What is 2+2?";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the GGUF model file (required — no HF download, local only)
    #[arg(long)]
    model: String,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// Prompt string
    #[arg(long)]
    prompt: Option<String>,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 200)]
    sample_len: usize,

    /// Temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Top-p nucleus sampling
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling
    #[arg(long)]
    top_k: Option<usize>,

    /// RNG seed
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Run on CPU
    #[arg(long)]
    cpu: bool,

    /// Repeat penalty (1.0 = none)
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Context window for repeat penalty
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let mut file = std::fs::File::open(&args.model)?;
    let start = std::time::Instant::now();

    let mut model = {
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(&args.model))?;
        println!(
            "loaded {} tensors in {:.2}s",
            ct.tensor_infos.len(),
            start.elapsed().as_secs_f32()
        );
        Qwen35::from_gguf(ct, &mut file, &device)?
    };
    println!("model built");

    let tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let prompt_str = args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let prompt_str =
        format!("<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n");
    print!("prompt: {}", &prompt_str);

    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();

    let mut logits_processor = {
        let temperature = args.temperature;
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

    let start_prefill = std::time::Instant::now();

    // Prefill — process the full prompt with the chunked scan
    let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;

    let prefill_dt = start_prefill.elapsed();
    println!(
        "\nprefill: {} tokens in {:.2}s ({:.1} t/s)",
        tokens.len(),
        prefill_dt.as_secs_f64(),
        tokens.len() as f64 / prefill_dt.as_secs_f64()
    );

    let mut all_tokens = vec![next_token];
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos_token = *tos
        .tokenizer()
        .get_vocab(true)
        .get("<|im_end|>")
        .unwrap_or(&0);

    let start_decode = std::time::Instant::now();
    let to_sample = args.sample_len.saturating_sub(1);

    for index in 0..to_sample {
        // Decode — single token at a time via sequential_step
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if next_token == eos_token {
            break;
        }
    }

    let decode_dt = start_decode.elapsed();
    let decode_tokens = all_tokens.len().saturating_sub(1);
    if let Some(rest) = tos.decode_rest()? {
        print!("{rest}");
    }
    println!(
        "\ndecode: {} tokens in {:.2}s ({:.1} t/s)",
        decode_tokens,
        decode_dt.as_secs_f64(),
        decode_tokens as f64 / decode_dt.as_secs_f64()
    );

    Ok(())
}
