//! TranslateGemma CLI Example
//!
//! Translation using Google's TranslateGemma models (4B, 12B, 27B).
//!
//! # Usage
//! ```bash
//! # Basic translation
//! cargo run --example translate_gemma --release -- \
//!     --text "Hello, how are you today?" \
//!     --source en --target fr
//!
//! # Interactive mode
//! cargo run --example translate_gemma --release -- \
//!     --interactive --source en --target de
//!
//! # Batch translation from file
//! cargo run --example translate_gemma --release -- \
//!     --input-file texts.txt --output-file translations.txt \
//!     --source en --target es
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::{self, BufRead, Write};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma::gemma3::{Config, Model};
use candle_transformers::models::gemma::translate_gemma::format_translate_prompt;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// TranslateGemma model variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ModelVariant {
    /// 4B parameters - mobile/edge optimized
    #[value(name = "4b")]
    T4B,
    /// 12B parameters - laptop deployment
    #[value(name = "12b")]
    T12B,
    /// 27B parameters - maximum quality
    #[value(name = "27b")]
    T27B,
}

impl ModelVariant {
    fn model_id(&self) -> &str {
        match self {
            ModelVariant::T4B => "google/translategemma-4b-it",
            ModelVariant::T12B => "google/translategemma-12b-it",
            ModelVariant::T27B => "google/translategemma-27b-it",
        }
    }
}

/// ISO 639-1 language codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Lang {
    #[value(name = "ar")]
    Arabic,
    #[value(name = "bg")]
    Bulgarian,
    #[value(name = "bn")]
    Bengali,
    #[value(name = "cs")]
    Czech,
    #[value(name = "da")]
    Danish,
    #[value(name = "de")]
    German,
    #[value(name = "el")]
    Greek,
    #[value(name = "en")]
    English,
    #[value(name = "es")]
    Spanish,
    #[value(name = "et")]
    Estonian,
    #[value(name = "fa")]
    Persian,
    #[value(name = "fi")]
    Finnish,
    #[value(name = "fr")]
    French,
    #[value(name = "he")]
    Hebrew,
    #[value(name = "hi")]
    Hindi,
    #[value(name = "hr")]
    Croatian,
    #[value(name = "hu")]
    Hungarian,
    #[value(name = "id")]
    Indonesian,
    #[value(name = "it")]
    Italian,
    #[value(name = "ja")]
    Japanese,
    #[value(name = "ko")]
    Korean,
    #[value(name = "lt")]
    Lithuanian,
    #[value(name = "lv")]
    Latvian,
    #[value(name = "nl")]
    Dutch,
    #[value(name = "no")]
    Norwegian,
    #[value(name = "pl")]
    Polish,
    #[value(name = "pt")]
    Portuguese,
    #[value(name = "ro")]
    Romanian,
    #[value(name = "ru")]
    Russian,
    #[value(name = "sk")]
    Slovak,
    #[value(name = "sl")]
    Slovenian,
    #[value(name = "sv")]
    Swedish,
    #[value(name = "th")]
    Thai,
    #[value(name = "tr")]
    Turkish,
    #[value(name = "uk")]
    Ukrainian,
    #[value(name = "vi")]
    Vietnamese,
    #[value(name = "zh")]
    Chinese,
}

impl Lang {
    fn code(&self) -> &str {
        match self {
            Lang::Arabic => "ar",
            Lang::Bulgarian => "bg",
            Lang::Bengali => "bn",
            Lang::Czech => "cs",
            Lang::Danish => "da",
            Lang::German => "de",
            Lang::Greek => "el",
            Lang::English => "en",
            Lang::Spanish => "es",
            Lang::Estonian => "et",
            Lang::Persian => "fa",
            Lang::Finnish => "fi",
            Lang::French => "fr",
            Lang::Hebrew => "he",
            Lang::Hindi => "hi",
            Lang::Croatian => "hr",
            Lang::Hungarian => "hu",
            Lang::Indonesian => "id",
            Lang::Italian => "it",
            Lang::Japanese => "ja",
            Lang::Korean => "ko",
            Lang::Lithuanian => "lt",
            Lang::Latvian => "lv",
            Lang::Dutch => "nl",
            Lang::Norwegian => "no",
            Lang::Polish => "pl",
            Lang::Portuguese => "pt",
            Lang::Romanian => "ro",
            Lang::Russian => "ru",
            Lang::Slovak => "sk",
            Lang::Slovenian => "sl",
            Lang::Swedish => "sv",
            Lang::Thai => "th",
            Lang::Turkish => "tr",
            Lang::Ukrainian => "uk",
            Lang::Vietnamese => "vi",
            Lang::Chinese => "zh",
        }
    }

    fn name(&self) -> &str {
        match self {
            Lang::Arabic => "Arabic",
            Lang::Bulgarian => "Bulgarian",
            Lang::Bengali => "Bengali",
            Lang::Czech => "Czech",
            Lang::Danish => "Danish",
            Lang::German => "German",
            Lang::Greek => "Greek",
            Lang::English => "English",
            Lang::Spanish => "Spanish",
            Lang::Estonian => "Estonian",
            Lang::Persian => "Persian",
            Lang::Finnish => "Finnish",
            Lang::French => "French",
            Lang::Hebrew => "Hebrew",
            Lang::Hindi => "Hindi",
            Lang::Croatian => "Croatian",
            Lang::Hungarian => "Hungarian",
            Lang::Indonesian => "Indonesian",
            Lang::Italian => "Italian",
            Lang::Japanese => "Japanese",
            Lang::Korean => "Korean",
            Lang::Lithuanian => "Lithuanian",
            Lang::Latvian => "Latvian",
            Lang::Dutch => "Dutch",
            Lang::Norwegian => "Norwegian",
            Lang::Polish => "Polish",
            Lang::Portuguese => "Portuguese",
            Lang::Romanian => "Romanian",
            Lang::Russian => "Russian",
            Lang::Slovak => "Slovak",
            Lang::Slovenian => "Slovenian",
            Lang::Swedish => "Swedish",
            Lang::Thai => "Thai",
            Lang::Turkish => "Turkish",
            Lang::Ukrainian => "Ukrainian",
            Lang::Vietnamese => "Vietnamese",
            Lang::Chinese => "Chinese",
        }
    }
}

/// Translator wrapping the model with inference logic.
struct Translator {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    max_tokens: usize,
    eos_token: u32,
    eot_token: u32,
}

impl Translator {
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        max_tokens: usize,
        device: Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        let eos_token = tokenizer.token_to_id("<eos>").unwrap_or(1);
        let eot_token = tokenizer.token_to_id("<end_of_turn>").unwrap_or(eos_token);

        Self {
            model,
            tokenizer,
            device,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            eos_token,
            eot_token,
        }
    }

    fn translate(&mut self, text: &str, source: Lang, target: Lang) -> Result<String> {
        self.model.clear_kv_cache();

        let prompt = format_translate_prompt(text, source.code(), target.code());

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(E::msg)?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mut output_tokens = Vec::new();

        let start = std::time::Instant::now();

        for index in 0..self.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let logits = if self.repeat_penalty == 1.0 {
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

            if next_token == self.eos_token || next_token == self.eot_token {
                break;
            }

            output_tokens.push(next_token);
        }

        let elapsed = start.elapsed();
        let tokens_generated = output_tokens.len();

        eprintln!(
            "[{} tokens in {:.2}s ({:.2} tok/s)]",
            tokens_generated,
            elapsed.as_secs_f64(),
            tokens_generated as f64 / elapsed.as_secs_f64()
        );

        let result = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(E::msg)?;

        // Clean up output
        let result = result
            .trim()
            .trim_end_matches("<end_of_turn>")
            .trim_end_matches("<eos>")
            .trim()
            .to_string();

        Ok(result)
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Translate text using Google's TranslateGemma models"
)]
struct Args {
    /// Run on CPU instead of GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing.
    #[arg(long)]
    tracing: bool,

    /// Text to translate.
    #[arg(long)]
    text: Option<String>,

    /// Source language.
    #[arg(long, short = 's', default_value = "en")]
    source: Lang,

    /// Target language.
    #[arg(long, short = 't', default_value = "fr")]
    target: Lang,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Top-p nucleus sampling.
    #[arg(long)]
    top_p: Option<f64>,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// Model variant (4b, 12b, 27b).
    #[arg(long, default_value = "4b")]
    model: ModelVariant,

    /// Custom model ID (overrides --model).
    #[arg(long)]
    model_id: Option<String>,

    /// Model revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Repetition penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Repetition context window.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use flash attention.
    #[arg(long)]
    use_flash_attn: bool,

    /// Interactive mode.
    #[arg(long, short = 'i')]
    interactive: bool,

    /// Input file (one text per line).
    #[arg(long)]
    input_file: Option<String>,

    /// Output file for translations.
    #[arg(long)]
    output_file: Option<String>,
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

    let model_id = args
        .model_id
        .unwrap_or_else(|| args.model.model_id().to_string());

    println!("Model: {}", model_id);
    println!(
        "Translation: {} ({}) → {} ({})",
        args.source.name(),
        args.source.code(),
        args.target.name(),
        args.target.code()
    );

    // Load model
    println!("\nLoading model...");
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.clone(),
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;

    // TranslateGemma uses sharded weights
    let weight_files =
        candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;

    println!("Retrieved files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    // Load config - TranslateGemma uses Gemma 3 architecture
    // The config has nested text_config, we need to extract it
    let config_content = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;

    // Extract text_config if present (for multimodal config)
    let text_config = if let Some(tc) = config_json.get("text_config") {
        serde_json::from_value(tc.clone())?
    } else {
        serde_json::from_str(&config_content)?
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
    let model = Model::new(args.use_flash_attn, &text_config, vb)?;

    println!("Loaded model in {:?}", start.elapsed());

    let temp = if args.temperature <= 0.0 {
        None // Greedy
    } else {
        Some(args.temperature)
    };

    let mut translator = Translator::new(
        model,
        tokenizer,
        args.seed,
        temp,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.max_tokens,
        device,
    );

    // Handle different input modes
    if let Some(input_file) = args.input_file {
        // Batch mode
        let file = std::fs::File::open(&input_file)?;
        let reader = io::BufReader::new(file);
        let mut output: Box<dyn Write> = match args.output_file {
            Some(ref path) => Box::new(std::fs::File::create(path)?),
            None => Box::new(io::stdout()),
        };

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                writeln!(output)?;
                continue;
            }

            eprint!("Translating line {}... ", i + 1);
            let translation = translator.translate(&line, args.source, args.target)?;
            writeln!(output, "{}", translation)?;
        }
    } else if args.interactive {
        // Interactive mode
        println!("\nInteractive mode. Type 'quit' to exit.");
        println!(
            "Translating {} → {}\n",
            args.source.name(),
            args.target.name()
        );

        let stdin = io::stdin();
        loop {
            print!("> ");
            io::stdout().flush()?;

            let mut input = String::new();
            stdin.lock().read_line(&mut input)?;
            let input = input.trim();

            if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
                break;
            }

            if input.is_empty() {
                continue;
            }

            match translator.translate(input, args.source, args.target) {
                Ok(translation) => println!("\n{}\n", translation),
                Err(e) => eprintln!("Error: {}\n", e),
            }
        }
    } else if let Some(text) = args.text {
        // Single text
        let translation = translator.translate(&text, args.source, args.target)?;
        println!("\n{}", translation);
    } else {
        // Read from stdin
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if !input.is_empty() {
            let translation = translator.translate(input, args.source, args.target)?;
            println!("{}", translation);
        }
    }

    Ok(())
}

