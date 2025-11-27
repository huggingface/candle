#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Import both model implementations
use candle_transformers::models::smol::quantized_smollm3::QuantizedModelForCausalLM;
use candle_transformers::models::smol::smollm3::{Config, ModelForCausalLM};

const DEFAULT_PROMPT: &str = "Write a Rust function to calculate the factorial of a given number.";

// ==================== Model Type Enum ====================

enum SmolLM3Model {
    Quantized(QuantizedModelForCausalLM),
    Full(ModelForCausalLM, Config), // Store config alongside model
}

impl SmolLM3Model {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        match self {
            Self::Quantized(model) => Ok(model.forward(input, pos)?),
            Self::Full(model, _) => Ok(model.forward(input, pos)?),
        }
    }

    fn config(&self) -> ModelConfig {
        match self {
            Self::Quantized(model) => {
                let cfg = model.config();
                ModelConfig {
                    vocab_size: cfg.vocab_size,
                    hidden_size: cfg.hidden_size,
                    num_hidden_layers: cfg.num_hidden_layers,
                    num_attention_heads: cfg.num_attention_heads,
                    num_key_value_heads: cfg.num_key_value_heads,
                    rope_theta: cfg.rope_theta as f32, // Convert f64 to f32
                    eos_token_id: Some(128012),        // Default SmolLM3 EOS
                    no_rope_layers: None,
                    no_rope_layer_interval: None,
                }
            }
            Self::Full(_, cfg) => {
                ModelConfig {
                    vocab_size: cfg.vocab_size,
                    hidden_size: cfg.hidden_size,
                    num_hidden_layers: cfg.num_hidden_layers,
                    num_attention_heads: cfg.num_attention_heads,
                    num_key_value_heads: cfg.num_key_value_heads,
                    rope_theta: cfg.rope_theta as f32, // Convert f64 to f32
                    eos_token_id: cfg.eos_token_id,
                    no_rope_layers: cfg
                        .no_rope_layers
                        .as_ref()
                        .map(|v| v.iter().map(|&x| x as u32).collect()), // Convert Vec<usize> to Vec<u32>
                    no_rope_layer_interval: cfg.no_rope_layer_interval,
                }
            }
        }
    }
}

// Unified config representation
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
    eos_token_id: Option<u32>,
    no_rope_layers: Option<Vec<u32>>,
    no_rope_layer_interval: Option<usize>,
}

impl ModelConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ==================== CLI Arguments ====================

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum ModelType {
    /// Use quantized GGUF model (smaller, faster)
    #[value(name = "quantized")]
    Quantized,
    /// Use full precision safetensors model (larger, more accurate)
    #[value(name = "full")]
    Full,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Quantization {
    #[value(name = "q4_k_m")]
    Q4KM,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "f16")]
    F16,
}

impl Quantization {
    fn filename_unsloth(&self) -> &'static str {
        match self {
            Self::Q4KM => "SmolLM3-3B-Q4_K_M.gguf",
            Self::Q8_0 => "SmolLM3-3B-Q8_0.gguf",
            Self::F16 => "SmolLM3-3B-F16.gguf",
        }
    }

    fn size_gb(&self) -> f32 {
        match self {
            Self::Q4KM => 1.92,
            Self::Q8_0 => 3.28,
            Self::F16 => 6.16,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    #[value(name = "3b")]
    W3b,
    #[value(name = "3b-base")]
    W3bBase,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model type: 'quantized' for GGUF or 'full' for safetensors
    #[arg(long, default_value = "quantized")]
    model_type: ModelType,

    /// Which model variant to use
    #[arg(long, default_value = "3b")]
    model: WhichModel,

    /// Quantization level (only for quantized models)
    /// Q8_0: 3.3GB, best quality | Q4_K_M: 1.9GB, good balance | F16: 6.2GB, full precision
    #[arg(long, default_value = "q8_0")]
    quantization: Quantization,

    /// Data type (only for full models: f32, f16, bf16, or auto)
    #[arg(long, default_value = "auto")]
    dtype: String,

    /// Path to model file (optional, will auto-download if not provided)
    #[arg(long)]
    model_path: Option<String>,

    /// Path to tokenizer file (optional, will auto-download if not provided)
    #[arg(long)]
    tokenizer: Option<String>,

    /// The initial prompt
    #[arg(long)]
    prompt: Option<String>,

    /// The length of the sample to generate (in tokens)
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Skip chat template formatting (use raw prompt, like base model)
    #[arg(long)]
    no_chat_template: bool,

    /// Enable thinking/reasoning mode (allows model to show its reasoning process)
    #[arg(long)]
    thinking: bool,

    /// Process prompt elements separately (slower, for debugging)
    #[arg(long)]
    split_prompt: bool,

    /// Enable tracing (generates a trace-timestamp.json file)
    #[arg(long)]
    tracing: bool,
}

impl Args {
    fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer {
            Some(path) => std::path::PathBuf::from(path),
            None => {
                let api = Api::new()?;
                let api = api.model("HuggingFaceTB/SmolLM3-3B".to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(E::msg)
    }

    fn should_use_chat_template(&self) -> bool {
        matches!(self.model, WhichModel::W3b) && !self.no_chat_template
    }
}

// ==================== Model Loading ====================

fn load_quantized_model(args: &Args, device: &Device) -> Result<SmolLM3Model> {
    let model_path = match &args.model_path {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let filename = args.quantization.filename_unsloth();
            let repo_id = "unsloth/SmolLM3-3B-GGUF";
            let api = Api::new()?;
            println!(
                "Downloading {} from {} (~{:.2}GB)...",
                filename,
                repo_id,
                args.quantization.size_gb()
            );
            api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                "main".to_string(),
            ))
            .get(filename)?
        }
    };

    println!("Loading quantized model from {:?}...", model_path);
    let model = QuantizedModelForCausalLM::from_gguf(&model_path, device)?;
    Ok(SmolLM3Model::Quantized(model))
}

fn load_full_model(args: &Args, device: &Device) -> Result<SmolLM3Model> {
    let api = Api::new()?;
    let model_id = match args.model {
        WhichModel::W3b => "HuggingFaceTB/SmolLM3-3B",
        WhichModel::W3bBase => "HuggingFaceTB/SmolLM3-3B-Base",
    };

    println!("Loading full model from: {}", model_id);
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let filenames = match &args.model_path {
        Some(path) => vec![std::path::PathBuf::from(path)],
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    let config_file = repo.get("config.json")?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;

    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "auto" => {
            if device.is_cuda() || device.is_metal() {
                DType::BF16
            } else {
                DType::F32
            }
        }
        other => anyhow::bail!("Unsupported dtype: {}, use f16, bf16, f32, or auto", other),
    };

    println!("Using dtype: {:?}", dtype);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
    let model = ModelForCausalLM::new(&config, vb)?;

    Ok(SmolLM3Model::Full(model, config))
}

// ==================== Text Generation ====================

fn format_prompt(prompt: &str, use_chat_template: bool, enable_thinking: bool) -> String {
    if use_chat_template {
        // Generate current date dynamically
        let now = chrono::Local::now();
        let today_date = now.format("%d %B %Y").to_string();

        // Set reasoning mode based on thinking flag
        let reasoning_mode = if enable_thinking {
            "/think"
        } else {
            "/no_think"
        };

        // Build the assistant start with or without thinking tags
        let assistant_start = if enable_thinking {
            "<|im_start|>assistant\n<think>\n" // Open for reasoning
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n" // Empty = skip reasoning
        };

        format!(
            "<|im_start|>system\n\
## Metadata\n\
\n\
Knowledge Cutoff Date: June 2025\n\
Today Date: {}\n\
Reasoning Mode: {}\n\
\n\
## Custom Instructions\n\
\n\
You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\
\n\
<|im_start|>user\n\
{}<|im_end|>\n\
{}",
            today_date, reasoning_mode, prompt, assistant_start
        )
    } else {
        prompt.to_string()
    }
}

fn get_eos_token(tokenizer: &Tokenizer, config: &ModelConfig) -> u32 {
    if let Some(eos_id) = config.eos_token_id {
        return eos_id;
    }

    let vocab = tokenizer.get_vocab(true);
    if let Some(&eos_id) = vocab.get("<|im_end|>") {
        return eos_id;
    }
    if let Some(&eos_id) = vocab.get("<|endoftext|>") {
        return eos_id;
    }

    128012 // Default SmolLM3 EOS token
}

fn run_generation(
    model: &mut SmolLM3Model,
    tokenizer: Tokenizer,
    args: &Args,
    device: &Device,
) -> Result<()> {
    let mut tos = TokenOutputStream::new(tokenizer);

    // Prepare prompt
    let prompt_str = args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let use_chat_template = args.should_use_chat_template();
    let formatted_prompt = format_prompt(&prompt_str, use_chat_template, args.thinking);

    println!("\n=== Generation Settings ===");
    println!("Model type: {:?}", args.model_type);
    println!(
        "Chat template: {}",
        if use_chat_template {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "Thinking mode: {}",
        if args.thinking {
            "enabled (/think)"
        } else {
            "disabled (/no_think)"
        }
    );
    println!("Raw prompt: {}", prompt_str);

    // Encode prompt
    let tokens = tos
        .tokenizer()
        .encode(formatted_prompt.as_str(), false)
        .map_err(E::msg)?;
    let tokens = tokens.get_ids();
    println!("Encoded {} tokens", tokens.len());

    // Setup logits processor
    let sampling = if args.temperature <= 0.0 {
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
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    // Process prompt
    let start_prompt = std::time::Instant::now();
    let mut next_token = if !args.split_prompt {
        let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, &token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[token], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            next_token = logits_processor.sample(&logits)?;
        }
        next_token
    };
    let prompt_dt = start_prompt.elapsed();

    // Get EOS token
    let config = model.config();
    let eos_token = get_eos_token(tos.tokenizer(), &config);

    // Generate tokens
    let mut all_tokens = vec![next_token];
    print!("\n=== Output ===\n");
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let start_generation = std::time::Instant::now();
    let to_sample = args.sample_len.saturating_sub(1);
    let mut sampled = 0;

    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

        let logits = if args.repeat_penalty == 1.0 {
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

        sampled += 1;
        if next_token == eos_token {
            break;
        }
    }

    if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }

    let generation_dt = start_generation.elapsed();

    // Print statistics
    println!(
        "\n\n=== Statistics ===\n\
         {:4} prompt tokens processed: {:.2} token/s\n\
         {:4} tokens generated: {:.2} token/s",
        tokens.len(),
        tokens.len() as f64 / prompt_dt.as_secs_f64(),
        sampled,
        sampled as f64 / generation_dt.as_secs_f64(),
    );

    Ok(())
}

// ==================== Main ====================

fn print_model_info(config: &ModelConfig) {
    println!("\n=== Model Configuration ===");
    println!("Vocab size: {}", config.vocab_size);
    println!("Hidden size: {}", config.hidden_size);
    println!("Num layers: {}", config.num_hidden_layers);
    println!("Num attention heads: {}", config.num_attention_heads);
    println!("Num KV heads: {}", config.num_key_value_heads);
    println!("Head dim: {}", config.head_dim());
    println!("RoPE theta: {:.0}", config.rope_theta);

    // Print RoPE/NoPE layer info for full models
    if let Some(ref no_rope_layers) = config.no_rope_layers {
        let num_rope_layers = no_rope_layers.iter().filter(|&&x| x == 1).count();
        let num_nope_layers = no_rope_layers.iter().filter(|&&x| x == 0).count();
        println!("\nLayer Configuration:");
        println!(
            "  RoPE layers: {} ({}%)",
            num_rope_layers,
            num_rope_layers * 100 / config.num_hidden_layers
        );
        println!(
            "  NoPE layers: {} ({}%)",
            num_nope_layers,
            num_nope_layers * 100 / config.num_hidden_layers
        );
    } else if let Some(interval) = config.no_rope_layer_interval {
        let num_nope_layers = config.num_hidden_layers / interval;
        let num_rope_layers = config.num_hidden_layers - num_nope_layers;
        println!("\nLayer Configuration:");
        println!(
            "  RoPE layers: {} ({}%)",
            num_rope_layers,
            num_rope_layers * 100 / config.num_hidden_layers
        );
        println!(
            "  NoPE layers: {} ({}%) - every {}th layer",
            num_nope_layers,
            num_nope_layers * 100 / config.num_hidden_layers,
            interval
        );
    }
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

    println!("=== SmolLM3 Unified Inference ===");
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2}, repeat-penalty: {:.2}, repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let device = candle_examples::device(false)?;

    // Load model
    let mut model = match args.model_type {
        ModelType::Quantized => load_quantized_model(&args, &device)?,
        ModelType::Full => load_full_model(&args, &device)?,
    };

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Print model info
    let config = model.config();
    print_model_info(&config);

    // Load tokenizer
    let tokenizer = args.get_tokenizer()?;

    // Run generation
    run_generation(&mut model, tokenizer, &args, &device)?;

    Ok(())
}
