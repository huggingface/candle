#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Write;

use candle::quantized::{gguf_file, tokenizer::TokenizerFromGguf};
use candle::Tensor;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma4::ModelWeights;
use clap::Parser;

const DEFAULT_PROMPT: &str = "Write a Rust function to calculate the factorial of a number.";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGUF file to load.
    #[arg(long)]
    model: std::path::PathBuf,

    /// The initial prompt.
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// The length of the sample to generate, in tokens.
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value_t = 0.)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Process prompt tokens separately.
    #[arg(long)]
    split_prompt: bool,

    /// Run on CPU rather than GPU even if a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Penalty to apply for repeating tokens, 1 means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let mut file = std::fs::File::open(&args.model)?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|error| error.with_path(&args.model))?;
    let tokenizer = tokenizers::Tokenizer::from_gguf(&content)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;
    let mut token_stream = TokenOutputStream::new(tokenizer);

    let prompt = format!("<|turn>user\n{}<turn|>\n<|turn>model\n", args.prompt.trim());
    let encoding = token_stream
        .tokenizer()
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?;
    let prompt_tokens = encoding.get_ids();
    let mut logits_processor = {
        let sampling = if args.temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All {
                    temperature: args.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: args.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: args.temperature,
                },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: args.temperature,
                },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let start_prompt_processing = std::time::Instant::now();
    let mut next_token = if args.split_prompt {
        let mut next_token = 0;
        for (position, token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
            next_token = logits_processor.sample(&model.forward(&input, position)?.squeeze(0)?)?;
        }
        next_token
    } else {
        let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
        logits_processor.sample(&model.forward(&input, 0)?.squeeze(0)?)?
    };
    let prompt_duration = start_prompt_processing.elapsed();
    let mut generated_tokens = vec![next_token];
    if let Some(text) = token_stream.next_token(next_token)? {
        print!("{text}");
        std::io::stdout().flush()?;
    }

    let end_token = token_stream
        .get_token("<turn|>")
        .ok_or_else(|| anyhow::anyhow!("cannot find <turn|> in the tokenizer"))?;
    let start_generation = std::time::Instant::now();
    for index in 1..args.sample_len {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model
            .forward(&input, prompt_tokens.len() + index - 1)?
            .squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = generated_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &generated_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        if next_token == end_token {
            break;
        }
        if let Some(text) = token_stream.next_token(next_token)? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = token_stream.decode_rest().map_err(candle::Error::msg)? {
        print!("{rest}");
    }
    let generation_duration = start_generation.elapsed();
    println!(
        "\n\n{} prompt tokens processed: {:.2} token/s",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prompt_duration.as_secs_f64()
    );
    println!(
        "{} tokens generated: {:.2} token/s",
        generated_tokens.len(),
        generated_tokens.len() as f64 / generation_duration.as_secs_f64()
    );
    Ok(())
}
