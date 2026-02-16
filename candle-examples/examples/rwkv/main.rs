#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use candle_transformers::models::quantized_rwkv_v5::Model as Q5;
use candle_transformers::models::quantized_rwkv_v6::Model as Q6;
use candle_transformers::models::rwkv_v5::{Config, Model as M5, State, Tokenizer};
use candle_transformers::models::rwkv_v6::Model as M6;
use candle_transformers::models::rwkv_v7::{
    Config as ConfigV7, Model as M7, ModelVersion, State as StateV7,
};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};

const EOS_TOKEN_ID: u32 = 261;

enum Model {
    M5(M5),
    Q5(Q5),
    M6(M6),
    Q6(Q6),
}

impl Model {
    fn forward(&self, xs: &Tensor, state: &mut State) -> candle::Result<Tensor> {
        match self {
            Self::M5(m) => m.forward(xs, state),
            Self::Q5(m) => m.forward(xs, state),
            Self::M6(m) => m.forward(xs, state),
            Self::Q6(m) => m.forward(xs, state),
        }
    }
}

struct TextGeneration {
    model: Model,
    config: Config,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        config: Config,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            config,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = 0usize;
        let mut state = State::new(1, &self.config, &self.device)?;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[[t]], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
            print!("{}", self.tokenizer.decode(&[t])?)
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
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
            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }
            print!("{}", self.tokenizer.decode(&[next_token])?);
            std::io::stdout().flush()?;

            let input = Tensor::new(&[[next_token]], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

/// Text generation pipeline for RWKV v7 models.
/// Separate from v5/v6 because v7 has different Config, State, and forward signature.
struct TextGenerationV7 {
    model: M7,
    config: ConfigV7,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    alpha_presence: f32,
    alpha_frequency: f32,
    alpha_decay: f32,
    stop: Option<String>,
}

impl TextGenerationV7 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: M7,
        config: ConfigV7,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        alpha_presence: f32,
        alpha_frequency: f32,
        alpha_decay: f32,
        device: &Device,
        dtype: DType,
        stop: Option<String>,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            config,
            tokenizer,
            logits_processor,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            device: device.clone(),
            dtype,
            stop,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        // Strip trailing whitespace — RWKV tokenizer produces non-English output otherwise
        let prompt = prompt.trim_end();
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = 0usize;
        let mut state = StateV7::new_with_dtype(&self.config, &self.device, self.dtype)?;

        // RWKV penalty state: per-token occurrence counts with exponential decay
        let vocab_size = self.config.vocab_size;
        let penalties_enabled = self.alpha_presence != 0.0 || self.alpha_frequency != 0.0;
        let mut occurrence: Vec<f32> = vec![0.0; vocab_size];

        // Process prompt using batched forward_seq for efficiency
        let start_prompt = std::time::Instant::now();
        let next_logits = self.model.forward_seq(&tokens, &mut state)?;
        let prompt_time = start_prompt.elapsed();

        // Update penalty counts for prompt tokens
        if penalties_enabled {
            for &t in tokens.iter() {
                for count in occurrence.iter_mut() {
                    *count *= self.alpha_decay;
                }
                if (t as usize) < vocab_size {
                    occurrence[t as usize] += 1.0;
                }
            }
        }

        // Print the prompt
        print!("{}", self.tokenizer.decode(&tokens)?);
        std::io::stdout().flush()?;

        let mut next_logits = Some(next_logits);
        println!(
            "\n[prompt: {} tokens in {:.2}s, {:.1} tok/s]",
            tokens.len(),
            prompt_time.as_secs_f64(),
            tokens.len() as f64 / prompt_time.as_secs_f64()
        );

        // Track generated text for stop sequence detection
        let mut generated_text = String::new();
        let mut printed_len = 0; // How many chars we've already printed

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.to_dtype(DType::F32)?;

            // Apply RWKV presence + frequency penalty
            let logits = if penalties_enabled {
                let mut logits_vec = logits.to_vec1::<f32>()?;
                for (i, logit) in logits_vec.iter_mut().enumerate() {
                    if occurrence[i] > 0.0 {
                        *logit -= self.alpha_presence + self.alpha_frequency * occurrence[i];
                    }
                }
                Tensor::from_vec(logits_vec, vocab_size, logits.device())?
            } else {
                logits
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if penalties_enabled {
                for count in occurrence.iter_mut() {
                    *count *= self.alpha_decay;
                }
                if (next_token as usize) < vocab_size {
                    occurrence[next_token as usize] += 1.0;
                }
            }

            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }

            let token_text = self.tokenizer.decode(&[next_token])?;
            generated_text.push_str(&token_text);

            // Check for stop sequence
            if let Some(stop) = &self.stop {
                if let Some(pos) = generated_text.find(stop.as_str()) {
                    // Print only up to the stop sequence
                    if pos > printed_len {
                        print!("{}", &generated_text[printed_len..pos]);
                        std::io::stdout().flush()?;
                    }
                    break;
                }
                // Only print text that can't be the start of stop sequence
                // Keep the last (stop.chars().count() - 1) chars buffered
                // Use char boundaries to avoid splitting multi-byte UTF-8 characters
                let stop_char_count = stop.chars().count();
                let total_chars = generated_text.chars().count();
                let safe_char_count = total_chars.saturating_sub(stop_char_count - 1);
                // Convert char count back to byte offset at a valid boundary
                let safe_len = generated_text
                    .char_indices()
                    .nth(safe_char_count)
                    .map(|(i, _)| i)
                    .unwrap_or(generated_text.len());
                if safe_len > printed_len {
                    print!("{}", &generated_text[printed_len..safe_len]);
                    std::io::stdout().flush()?;
                    printed_len = safe_len;
                }
            } else {
                print!("{}", token_text);
                std::io::stdout().flush()?;
            }

            let input = Tensor::new(&[[next_token]], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state, &[next_token])?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    // RWKV v5 models
    Eagle7b,
    World1b5,
    World3b,
    // RWKV v6 models
    World6_1b6,
    // RWKV v7 models: rwkv7-g1d (original v7 architecture, generation 1 dataset d)
    #[value(name = "rwkv7-g1d-0.1b")]
    Rwkv7G1d0_1b,
    #[value(name = "rwkv7-g1d-0.4b")]
    Rwkv7G1d0_4b,
    #[value(name = "rwkv7-g1d-1.5b")]
    Rwkv7G1d1_5b,
    #[value(name = "rwkv7-g1d-2.9b")]
    Rwkv7G1d2_9b,
    #[value(name = "rwkv7-g1d-7.2b")]
    Rwkv7G1d7_2b,
    #[value(name = "rwkv7-g1d-13.3b")]
    Rwkv7G1d13_3b,
    // RWKV v7a models: rwkv7a-g1d (v7a variant, generation 1 dataset d)
    #[value(name = "rwkv7a-g1d-0.1b")]
    Rwkv7aG1d0_1b,
    // RWKV v7b models: rwkv7b-g1b (v7b variant, generation 1 dataset b)
    #[value(name = "rwkv7b-g1b-0.1b")]
    Rwkv7bG1b0_1b,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Which {
    fn is_v7(&self) -> bool {
        matches!(
            self,
            Self::Rwkv7G1d0_1b
                | Self::Rwkv7G1d0_4b
                | Self::Rwkv7G1d1_5b
                | Self::Rwkv7G1d2_9b
                | Self::Rwkv7G1d7_2b
                | Self::Rwkv7G1d13_3b
                | Self::Rwkv7aG1d0_1b
                | Self::Rwkv7bG1b0_1b
        )
    }

    fn model_id(&self) -> &'static str {
        match self {
            Self::Eagle7b => "RWKV/v5-Eagle-7B-HF",
            Self::World1b5 => "RWKV/rwkv-5-world-1b5",
            Self::World3b => "RWKV/rwkv-5-world-3b",
            Self::World6_1b6 => "paperfun/rwkv",
            Self::Rwkv7G1d0_1b
            | Self::Rwkv7G1d0_4b
            | Self::Rwkv7G1d1_5b
            | Self::Rwkv7G1d2_9b
            | Self::Rwkv7G1d7_2b
            | Self::Rwkv7G1d13_3b
            | Self::Rwkv7aG1d0_1b
            | Self::Rwkv7bG1b0_1b => "DanielClough/rwkv7-g1-safetensors",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Eagle7b => "refs/pr/1",
            Self::World1b5 | Self::World3b => "refs/pr/2",
            Self::World6_1b6 => "main",
            Self::Rwkv7G1d0_1b
            | Self::Rwkv7G1d0_4b
            | Self::Rwkv7G1d1_5b
            | Self::Rwkv7G1d2_9b
            | Self::Rwkv7G1d7_2b
            | Self::Rwkv7G1d13_3b
            | Self::Rwkv7aG1d0_1b
            | Self::Rwkv7bG1b0_1b => "main",
        }
    }

    fn v7_version(&self) -> Option<ModelVersion> {
        match self {
            Self::Rwkv7G1d0_1b
            | Self::Rwkv7G1d0_4b
            | Self::Rwkv7G1d1_5b
            | Self::Rwkv7G1d2_9b
            | Self::Rwkv7G1d7_2b
            | Self::Rwkv7G1d13_3b => Some(ModelVersion::V7),
            Self::Rwkv7aG1d0_1b => Some(ModelVersion::V7a),
            Self::Rwkv7bG1b0_1b => Some(ModelVersion::V7b),
            _ => None,
        }
    }

    fn v7_config(&self) -> Option<ConfigV7> {
        let version = self.v7_version()?;
        let (hidden_size, num_hidden_layers) = match self {
            Self::Rwkv7G1d0_1b | Self::Rwkv7aG1d0_1b | Self::Rwkv7bG1b0_1b => (768, 12),
            Self::Rwkv7G1d0_4b => (1024, 24),
            Self::Rwkv7G1d1_5b => (2048, 24),
            Self::Rwkv7G1d2_9b => (2560, 32),
            Self::Rwkv7G1d7_2b => (4096, 32),
            Self::Rwkv7G1d13_3b => (4096, 61),
            _ => return None,
        };
        Some(ConfigV7 {
            version,
            vocab_size: 65536,
            hidden_size,
            num_hidden_layers,
            head_size: 64,
            intermediate_size: None, // defaults to hidden_size * 4
            rescale_every: 0,
        })
    }
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Preset {
    /// Chat: temp 1.0, top_p 0.5, presence 2.0, frequency 0.1, decay 0.99
    Chat,
    /// Creative (fiction etc.): temp 0.6, top_p 0.7, presence 2.0, frequency 0.2, decay 0.99
    Creative,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum PromptTemplate {
    /// Pass prompt as-is with no formatting.
    Raw,
    /// Chat format: User: {prompt}\n\nA:
    Chat,
    /// Think format: User: {prompt}\n\nA: <think>
    Think,
    /// Fake think (recommended): User: {prompt}\n\nA: <think></think
    FakeThink,
    /// Fill-in-middle for G1c+ models (text, code, everything): ✿prefix✿✿suffix✿{suffix}✿middle✿{prompt}
    Fim,
}

/// Format the user prompt according to the selected template.
fn apply_template(
    template: PromptTemplate,
    prompt: &str,
    system: Option<&str>,
    suffix: Option<&str>,
) -> String {
    match template {
        PromptTemplate::Raw => prompt.to_string(),
        PromptTemplate::Chat => {
            // Replace \n\n in user prompt with \n (double newline is chat round separator)
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA:"));
            out
        }
        PromptTemplate::Think => {
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA: <think>"));
            out
        }
        PromptTemplate::FakeThink => {
            let prompt = prompt.replace("\n\n", "\n");
            let mut out = String::new();
            if let Some(sys) = system {
                out.push_str(&format!("System: {sys}\n\n"));
            }
            out.push_str(&format!("User: {prompt}\n\nA: <think></think"));
            out
        }
        PromptTemplate::Fim => {
            let suffix = suffix.unwrap_or("");
            // FIM prompt for G1c and newer models (works for text, code, and everything)
            // Recommended format: ✿prefix✿✿suffix✿<suffix>✿middle✿<prompt>
            // The model continues from <prompt> and generates until it reaches <suffix>
            format!("✿prefix✿✿suffix✿{suffix}✿middle✿{prompt}")
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// Prompt template to apply (v7 only).
    #[arg(long, default_value = "raw")]
    template: PromptTemplate,

    /// System prompt for chat/think templates.
    #[arg(long)]
    system: Option<String>,

    /// Suffix text for FIM (fill-in-middle) template.
    #[arg(long)]
    suffix: Option<String>,

    /// Sampling preset (v7 only). Overrides temperature, top_p, and penalty defaults.
    #[arg(long)]
    preset: Option<Preset>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.5")]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    /// Stop generation when this text is produced (e.g., --stop "User:").
    #[arg(long)]
    stop: Option<String>,

    #[arg(long, default_value = "world1b5")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Data type for inference: f32, f16, or bf16. Half precision (f16/bf16) is faster.
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty (v5/v6 only).
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty (v5/v6 only).
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// RWKV presence penalty (v7 only). Flat additive penalty for any token that has appeared.
    #[arg(long, default_value_t = 2.0)]
    alpha_presence: f32,

    /// RWKV frequency penalty (v7 only). Additive penalty proportional to token count.
    #[arg(long, default_value_t = 0.1)]
    alpha_frequency: f32,

    /// RWKV penalty count decay (v7 only). Exponential decay applied to token counts each step.
    #[arg(long, default_value_t = 0.99)]
    alpha_decay: f32,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let mut args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Apply preset overrides (v7 only)
    if let Some(preset) = args.preset {
        match preset {
            Preset::Chat => {
                args.temperature = 1.0;
                args.top_p = Some(0.5);
                args.alpha_presence = 2.0;
                args.alpha_frequency = 0.1;
                args.alpha_decay = 0.99;
            }
            Preset::Creative => {
                args.temperature = 0.6;
                args.top_p = Some(0.7);
                args.alpha_presence = 2.0;
                args.alpha_frequency = 0.2;
                args.alpha_decay = 0.99;
            }
        }
    }

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id
            .unwrap_or_else(|| args.which.model_id().to_string()),
        RepoType::Model,
        args.revision
            .unwrap_or_else(|| args.which.revision().to_string()),
    ));
    let tokenizer = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("lmz/candle-rwkv".to_string())
            .get("rwkv_vocab_v20230424.json")?,
    };
    let config_filename = match (&args.config_file, args.which.is_v7()) {
        (Some(file), _) => Some(std::path::PathBuf::from(file)),
        (None, true) => None, // v7 models use built-in config, no config.json needed
        (None, false) => Some(repo.get("config.json")?),
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                if args.which.is_v7() {
                    anyhow::bail!("quantized RWKV v7 models are not yet supported");
                }
                vec![match args.which {
                    Which::World1b5 => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("world1b5-q4k.gguf")?,
                    Which::World3b => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("world3b-q4k.gguf")?,
                    Which::Eagle7b => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("eagle7b-q4k.gguf")?,
                    Which::World6_1b6 => repo.get("rwkv-6-world-1b6-q4k.gguf")?,
                    _ => unreachable!(),
                }]
            } else {
                vec![match args.which {
                    Which::World1b5 | Which::World3b | Which::Eagle7b => {
                        repo.get("model.safetensors")?
                    }
                    Which::World6_1b6 => repo.get("rwkv-6-world-1b6.safetensors")?,
                    Which::Rwkv7G1d0_1b => {
                        repo.get("rwkv7-g1d-0.1b-20260129-ctx8192.safetensors")?
                    }
                    Which::Rwkv7G1d0_4b => {
                        repo.get("rwkv7-g1d-0.4b-20260210-ctx8192.safetensors")?
                    }
                    Which::Rwkv7G1d1_5b => {
                        repo.get("rwkv7-g1d-1.5b-20260212-ctx8192.safetensors")?
                    }
                    Which::Rwkv7G1d2_9b => {
                        repo.get("rwkv7-g1d-2.9b-20260131-ctx8192.safetensors")?
                    }
                    Which::Rwkv7G1d7_2b => {
                        repo.get("rwkv7-g1d-7.2b-20260131-ctx8192.safetensors")?
                    }
                    Which::Rwkv7G1d13_3b => {
                        repo.get("rwkv7-g1d-13.3b-20260131-ctx8192.safetensors")?
                    }
                    Which::Rwkv7aG1d0_1b => {
                        repo.get("rwkv7a-g1d-0.1b-20260212-ctx8192.safetensors")?
                    }
                    Which::Rwkv7bG1b0_1b => {
                        repo.get("rwkv7b-g1b-0.1b-20250822-ctx4096.safetensors")?
                    }
                }]
            }
        }
    };
    let tokenizer = Tokenizer::new(tokenizer)?;
    let device = candle_examples::device(args.cpu)?;

    if args.which.is_v7() {
        // RWKV v7 path — different Config, State, and forward signature
        let config: ConfigV7 = if let Some(config_file) = &config_filename {
            serde_json::from_slice(&std::fs::read(config_file)?)?
        } else {
            args.which
                .v7_config()
                .expect("v7 variant must have built-in config")
        };

        // Parse dtype from string
        let dtype = match args.dtype.to_lowercase().as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "f32" => DType::F32,
            other => anyhow::bail!("Unknown dtype '{}'. Use f32, f16, or bf16.", other),
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = M7::new(&config, vb)?;

        // For FIM template, auto-set stop sequence to ✿ (delimiter signals end of middle)
        let stop = match (&args.stop, args.template) {
            (Some(s), _) => Some(s.clone()), // User-specified stop takes precedence
            (None, PromptTemplate::Fim) => Some("✿".to_string()), // FIM auto-stops on delimiter
            (None, _) => None,
        };

        let mut pipeline = TextGenerationV7::new(
            model,
            config,
            tokenizer,
            args.seed,
            Some(args.temperature),
            args.top_p,
            args.alpha_presence,
            args.alpha_frequency,
            args.alpha_decay,
            &device,
            dtype,
            stop,
        );
        let prompt = apply_template(
            args.template,
            &args.prompt,
            args.system.as_deref(),
            args.suffix.as_deref(),
        );
        pipeline.run(&prompt, args.sample_len)?;
    } else {
        // v5/v6 path (existing behavior)
        let config: Config = serde_json::from_slice(&std::fs::read(config_filename.unwrap())?)?;
        let model = if args.quantized {
            let filename = &filenames[0];
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                filename, &device,
            )?;
            match args.which {
                Which::World1b5 | Which::World3b | Which::Eagle7b => {
                    Model::Q5(Q5::new(&config, vb)?)
                }
                Which::World6_1b6 => Model::Q6(Q6::new(&config, vb)?),
                _ => unreachable!(),
            }
        } else {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
            match args.which {
                Which::World1b5 | Which::World3b | Which::Eagle7b => {
                    Model::M5(M5::new(&config, vb)?)
                }
                Which::World6_1b6 => Model::M6(M6::new(&config, vb)?),
                _ => unreachable!(),
            }
        };

        let mut pipeline = TextGeneration::new(
            model,
            config,
            tokenizer,
            args.seed,
            Some(args.temperature),
            args.top_p,
            args.repeat_penalty,
            args.repeat_last_n,
            &device,
        );
        pipeline.run(&args.prompt, args.sample_len)?;
    }

    Ok(())
}
