use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::{Config, Model};
use clap::Parser;
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 50)]
    sample_len: usize,

    /// The model repository to use on HuggingFace.
    #[arg(long, default_value = "microsoft/DialoGPT-medium")]
    model_id: String,

    /// The tokenizer config repository to use on HuggingFace.
    #[arg(long)]
    tokenizer_repo: Option<String>,

    /// Local tokenizer config file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args.model_id.clone();
    let repo = api.model(model_id.clone());

    let tokenizer_filename = match (args.tokenizer_repo, args.tokenizer_file) {
        (Some(repo), _) => {
            let tokenizer_repo = api.model(repo);
            tokenizer_repo.get("tokenizer.json")?
        }
        (None, Some(file)) => file.into(),
        (None, None) => repo.get("tokenizer.json")?,
    };

    let filenames = [
        "config.json",
        "model.safetensors",
    ];
    let filenames = filenames
        .iter()
        .map(|f| repo.get(f))
        .collect::<Result<Vec<_>, _>>()?;

    println!("retrieved the files in {:?}", start.elapsed());
    let config_filename = &filenames[0];
    let weights_filename = &filenames[1];

    let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
    let device = candle_examples::device(args.cpu)?;

    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? };
    let mut model = Model::new(&config, vb)?;
    model.clear_kv_cache();

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let prompt = args.prompt.as_ref().map_or("Hello", |p| p.as_str());
    let tokens = tos
        .tokenizer()
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    if tokens.is_empty() {
        anyhow::bail!("Empty prompts are not supported in the GPT-OSS model.")
    }

    println!("starting the inference loop");
    print!("{}", prompt);
    let mut tokens = tokens;
    let mut generated_tokens = 0usize;
    let eos_token = match tos.tokenizer().token_to_id("</s>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };

    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
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

        let next_token = candle_transformers::utils::sample_with_temperature_and_nucleus(
            &logits,
            args.temperature,
            args.top_p,
            args.seed + index as u64,
        )?;
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        if let Some(t) = tos.next_token(next_token)? {
            print!("{}", t);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    std::io::Write::flush(&mut std::io::stdout())?;

    let dt = start_gen.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );

    Ok(())
}