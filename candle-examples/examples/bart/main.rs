//! BART example for text-to-text and vision-encoder-decoder models.
//!
//! This example demonstrates loading BART models from HuggingFace:
//! - Full encoder-decoder BART/mBART for text generation (summarization, translation)
//! - VisionEncoderDecoder models (Donut, TrOCR) for document understanding
//! - Beam search decoding with length penalty and n-gram blocking
//!
//! ```bash
//! # Text-to-text summarization with BART (sampling)
//! cargo run --example bart --release -- \
//!     --model-id facebook/bart-large-cnn \
//!     --prompt "The tower is 324 metres tall..." \
//!     --sample-len 50
//!
//! # Beam search decoding (better quality for summarization)
//! cargo run --example bart --release -- \
//!     --model-id facebook/bart-large-cnn \
//!     --prompt "The Eiffel Tower is a wrought-iron lattice tower..." \
//!     --beam-size 4 \
//!     --length-penalty 2.0 \
//!     --sample-len 50
//!
//! # Multilingual translation with mBART (requires tokenizer conversion first)
//! # Step 1: Convert SentencePiece tokenizer to tokenizer.json
//! python convert_mbart_tokenizer.py --model-id facebook/mbart-large-50-many-to-many-mmt
//!
//! # Step 2: Run translation (English to French)
//! cargo run --example bart --release -- \
//!     --model-id facebook/mbart-large-50-many-to-many-mmt \
//!     --prompt "Hello, how are you?" \
//!     --source-lang en_XX \
//!     --target-lang fr_XX
//!
//! # VisionEncoderDecoder with dummy encoder (default)
//! cargo run --example bart --release -- \
//!     --model-id naver-clova-ix/donut-base \
//!     --use-dummy-encoder \
//!     --sample-len 50
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::bart::{self, beam_search, BeamSearchConfig};

use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "naver-clova-ix/donut-base";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model to use.
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Revision (branch) of the model.
    #[arg(long, default_value = "main")]
    revision: String,

    /// The number of tokens to generate.
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// The seed for random sampling.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Use dummy encoder output for VisionEncoderDecoder models.
    #[arg(long)]
    use_dummy_encoder: bool,

    /// Input text for text-to-text generation (triggers text mode).
    #[arg(long)]
    prompt: Option<String>,

    /// Target language token for generation (e.g., "fr_XX" for mBART).
    /// For mBART models, run convert_mbart_tokenizer.py first.
    #[arg(long)]
    target_lang: Option<String>,

    /// Source language token for mBART (e.g., "en_XX").
    /// Prepended to input for multilingual models.
    #[arg(long)]
    source_lang: Option<String>,

    /// Encoder sequence length for vision models (default: 577 for Donut).
    #[arg(long, default_value_t = 577)]
    encoder_seq_len: usize,

    /// Temperature for sampling.
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-p (nucleus) sampling.
    #[arg(long)]
    top_p: Option<f64>,

    /// Beam size for beam search decoding. When > 1, uses beam search instead of sampling.
    #[arg(long, default_value_t = 1)]
    beam_size: usize,

    /// Length penalty α for beam search. Higher values (1.5-2.0) favor longer outputs.
    /// Wu et al. 2016 formula: score = log_prob / (length^α)
    #[arg(long, default_value_t = 2.0)]
    length_penalty: f64,

    /// Minimum generation length before EOS is allowed (beam search only).
    #[arg(long, default_value_t = 10)]
    min_length: usize,

    /// Block n-gram repetition (0 = disabled, beam search only).
    #[arg(long, default_value_t = 3)]
    no_repeat_ngram_size: usize,
}

impl Args {
    fn validate(&self) -> Result<()> {
        // Conflicting options
        if self.prompt.is_some() && self.use_dummy_encoder {
            anyhow::bail!("Cannot use --use-dummy-encoder with --prompt (text mode)");
        }

        // Language options require text mode
        if self.target_lang.is_some() && self.prompt.is_none() {
            anyhow::bail!("--target-lang requires --prompt (text generation mode)");
        }
        if self.source_lang.is_some() && self.prompt.is_none() {
            anyhow::bail!("--source-lang requires --prompt (text generation mode)");
        }

        Ok(())
    }
}

/// Config structure that handles both VisionEncoderDecoder and direct BART configs.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
enum ModelConfig {
    /// VisionEncoderDecoder format (Donut, TrOCR)
    VisionEncoderDecoder {
        decoder: bart::BartConfig,
        /// Encoder config is parsed but not used (decoder-only mode)
        #[serde(default)]
        #[allow(dead_code)]
        encoder: Option<serde_json::Value>,
    },
    /// Direct BART/mBART config
    Direct(bart::BartConfig),
}

impl ModelConfig {
    fn bart_config(&self) -> &bart::BartConfig {
        match self {
            Self::VisionEncoderDecoder { decoder, .. } => decoder,
            Self::Direct(config) => config,
        }
    }
}

fn run_text_generation(args: &Args, device: &candle::Device) -> Result<()> {
    use hf_hub::api::sync::Api;

    let prompt = args
        .prompt
        .as_ref()
        .expect("--prompt is required for text generation");

    println!("Loading model from: {}", args.model_id);

    let api = Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    let config_file = repo.get("config.json")?;
    let config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    let mut bart_config = config.bart_config().clone();
    println!(
        "Config: d_model={}, layers={}",
        bart_config.d_model, bart_config.decoder_layers
    );

    // Load tokenizer - handle mBART models that need conversion
    let tokenizer_file = match repo.get("tokenizer.json") {
        Ok(f) => f,
        Err(_) => {
            // Check if this is an mBART model with SentencePiece
            if repo.get("sentencepiece.bpe.model").is_ok() {
                anyhow::bail!(
                    "Model uses SentencePiece tokenizer (no tokenizer.json).\n\n\
                     For mBART models, convert the tokenizer first:\n  \
                     cd candle-examples/examples/bart\n  \
                     python convert_mbart_tokenizer.py --model-id {}\n\n\
                     Then run the example again.",
                    args.model_id
                );
            }
            anyhow::bail!("tokenizer.json not found for model: {}", args.model_id);
        }
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
    let mut tokenizer_stream = TokenOutputStream::new(tokenizer.clone());

    // Handle target language for mBART - set in config BEFORE initial_decoder_tokens()
    if let Some(target_lang) = &args.target_lang {
        if let Some(lang_token_id) = tokenizer.token_to_id(target_lang) {
            println!(
                "Setting target language: {} (token_id={})",
                target_lang, lang_token_id
            );
            bart_config.forced_bos_token_id = Some(lang_token_id);
        } else {
            anyhow::bail!(
                "Target language '{}' not found in tokenizer vocabulary.\n\
                 Available language codes include: en_XX, fr_XX, de_DE, es_XX, etc.",
                target_lang
            );
        }
    }

    // Handle source language for mBART - prepend to input
    let input_text = if let Some(source_lang) = &args.source_lang {
        if tokenizer.token_to_id(source_lang).is_none() {
            anyhow::bail!(
                "Source language '{}' not found in tokenizer vocabulary.\n\
                 Available language codes include: en_XX, fr_XX, de_DE, es_XX, etc.",
                source_lang
            );
        }
        // mBART expects source language token prepended to input
        println!("Using source language: {}", source_lang);
        format!("{} {}", source_lang, prompt)
    } else {
        prompt.to_string()
    };

    println!("Loading model weights...");
    let model_file = match repo.get("model.safetensors") {
        Ok(f) => f,
        Err(_) => repo.get("pytorch_model.bin")?,
    };
    let vb = if model_file.extension().is_some_and(|ext| ext == "bin") {
        VarBuilder::from_pth(&model_file, DType::F32, device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, device)? }
    };

    let mut model = bart::BartForConditionalGeneration::new(&bart_config, vb)?;
    println!("Model loaded successfully!");

    // Tokenize input (with source language prepended if specified)
    let encoding = tokenizer
        .encode(input_text.as_str(), true)
        .map_err(E::msg)?;
    let input_ids = encoding.get_ids();
    println!("Input tokens: {:?}", input_ids);

    let input_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;

    // Encode input
    println!("Encoding input...");
    let encoder_output = model.encode(&input_tensor)?;
    println!("Encoder output shape: {:?}", encoder_output.dims());

    // Initialize decoder tokens
    let initial_tokens = bart_config.initial_decoder_tokens();
    let forced_bos = bart_config.get_forced_bos_token_id();
    println!("Initial decoder tokens: {:?}", initial_tokens);
    if let Some(lang_id) = forced_bos {
        println!("Forced first token: {} (target language)", lang_id);
    }

    // Dispatch between beam search and sampling
    let token_ids = if args.beam_size > 1 {
        println!(
            "\nRunning beam search (beam_size={}, length_penalty={})...\n",
            args.beam_size, args.length_penalty
        );
        let start = std::time::Instant::now();
        let result_tokens = beam_search(
            &mut model,
            &encoder_output,
            &bart_config,
            &BeamSearchConfig::new(
                args.beam_size,
                args.length_penalty,
                args.min_length,
                args.no_repeat_ngram_size,
                args.sample_len,
            ),
            device,
        )?;
        let elapsed = start.elapsed();

        // Decode and print output
        let num_start_tokens = initial_tokens.len();
        let output_tokens = &result_tokens[num_start_tokens..];
        let output_text = tokenizer.decode(output_tokens, true).map_err(E::msg)?;

        println!("---");
        println!("{}", output_text);
        println!("---");
        println!(
            "\nGenerated {} tokens in {:.2}s ({:.1} tokens/sec)",
            output_tokens.len(),
            elapsed.as_secs_f32(),
            output_tokens.len() as f32 / elapsed.as_secs_f32()
        );

        result_tokens
    } else {
        // Sampling-based generation
        let mut token_ids = initial_tokens;

        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
            args.seed,
            args.temperature,
            args.top_p,
        );

        println!("\nGenerating {} tokens...\n", args.sample_len);
        println!("---");

        for index in 0..args.sample_len {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let past_kv_len = if index >= 1 { start_pos } else { 0 };

            let decoder_input: Vec<u32> = token_ids[start_pos..].to_vec();
            let decoder_tensor = Tensor::new(decoder_input.as_slice(), device)?.unsqueeze(0)?;

            let logits = model.decode(&decoder_tensor, &encoder_output, past_kv_len)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;

            // For mBART: force first generated token to be target language
            let token = if index == 0 {
                if let Some(forced) = forced_bos {
                    forced
                } else {
                    logits_processor.sample(&logits)?
                }
            } else {
                logits_processor.sample(&logits)?
            };
            token_ids.push(token);

            if let Some(t) = tokenizer_stream.next_token(token)? {
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
            }

            if token == bart_config.eos_token_id {
                break;
            }
            if let Some(forced_eos) = bart_config.forced_eos_token_id {
                if token == forced_eos {
                    break;
                }
            }
        }

        if let Some(rest) = tokenizer_stream.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        println!("\n---");

        println!(
            "\nGenerated {} tokens (including start tokens)",
            token_ids.len()
        );

        token_ids
    };

    let _ = token_ids; // Suppress unused warning

    Ok(())
}

fn run_vision_generation(args: &Args, device: &candle::Device) -> Result<()> {
    use hf_hub::api::sync::Api;

    println!("Loading VisionEncoderDecoder model from: {}", args.model_id);

    let api = Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    let config_file = repo.get("config.json")?;
    let config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    let bart_config = config.bart_config();
    println!(
        "Decoder config: d_model={}, layers={}",
        bart_config.d_model, bart_config.decoder_layers
    );

    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    println!("Loading model weights...");
    let model_file = match repo.get("model.safetensors") {
        Ok(f) => f,
        Err(_) => repo.get("pytorch_model.bin")?,
    };
    let vb = if model_file.extension().is_some_and(|ext| ext == "bin") {
        VarBuilder::from_pth(&model_file, DType::F32, device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, device)? }
    };

    let mut model = bart::BartForCausalLM::new(bart_config, vb)?;
    println!("Model loaded successfully!");

    let encoder_hidden_size = bart_config
        .cross_attention_hidden_size
        .unwrap_or(bart_config.d_model);
    let encoder_seq_len = args.encoder_seq_len;

    let encoder_xs = if args.use_dummy_encoder {
        println!(
            "Using dummy encoder output: (1, {}, {})",
            encoder_seq_len, encoder_hidden_size
        );
        Tensor::randn(
            0f32,
            1f32,
            (1, encoder_seq_len, encoder_hidden_size),
            device,
        )?
    } else {
        anyhow::bail!(
            "Real vision encoder not implemented.\n\n\
             To test decoder functionality, run:\n  \
             cargo run --example bart --release -- \\\n    \
             --model-id {} \\\n    \
             --use-dummy-encoder \\\n    \
             --sample-len 50",
            args.model_id
        );
    };

    let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
        args.seed,
        args.temperature,
        args.top_p,
    );

    let start_token = bart_config.decoder_start_token_id.unwrap_or(2);
    let mut token_ids = vec![start_token];

    println!("\nGenerating {} tokens...\n", args.sample_len);
    println!("---");

    for index in 0..args.sample_len {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let past_kv_len = if index >= 1 { start_pos } else { 0 };

        let input_ids = Tensor::new(&token_ids[start_pos..], device)?.unsqueeze(0)?;

        let logits = model.decode(&input_ids, &encoder_xs, past_kv_len)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;

        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);

        if let Some(t) = tokenizer.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }

        if token == bart_config.eos_token_id {
            break;
        }
        if let Some(forced_eos) = bart_config.forced_eos_token_id {
            if token == forced_eos {
                break;
            }
        }
    }

    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!("\n---");

    println!(
        "\nGenerated {} tokens (including start token)",
        token_ids.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    args.validate()?;
    let device = candle_examples::device(args.cpu)?;

    if args.prompt.is_some() {
        // Text-to-text mode: use BartForConditionalGeneration
        run_text_generation(&args, &device)
    } else {
        // Vision mode: use BartForCausalLM with external encoder
        run_vision_generation(&args, &device)
    }
}
