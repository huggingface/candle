mod audio;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::voxtral::{
    VoxtralForConditionalGeneration, VoxtralCache, VoxtralConfig, 
    VoxtralEncoderConfig
};
use candle_transformers::models::llama::{Config as LlamaConfig, LlamaEosToks};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json;
use tokenizers::Tokenizer;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the audio file to process
    #[arg(long, default_value = "hello.mp4")]
    audio_file: String,

    /// The prompt to use for generation
    #[arg(long, default_value = "Transcribe the following audio:")]
    prompt: String,

    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Temperature for sampling (0 for greedy decoding)
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// Top-p sampling parameter
    #[arg(long)]
    top_p: Option<f64>,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "512")]
    max_new_tokens: usize,

    /// Audio token ID for the model
    #[arg(long, default_value = "128256")]
    audio_token_id: usize,

    /// Model weights directory path or Hugging Face model ID
    #[arg(long)]
    model_dir: Option<String>,

    /// Hugging Face model ID to download (alternative to model-dir)
    #[arg(long, default_value = "fixie-ai/ultravox_v0_3")]
    model_id: String,

    /// Download model from Hugging Face if not found locally
    #[arg(long)]
    download: bool,

    /// Use demonstration mode (no model weights required)
    #[arg(long)]
    demo_mode: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Set up device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    
    println!("Using device: {:?}", device);
    println!("Audio file: {}", args.audio_file);
    
    // Check if audio file exists
    if !std::path::Path::new(&args.audio_file).exists() {
        anyhow::bail!("Audio file not found: {}. Try using the default 'hello.mp4'", args.audio_file);
    }
    
    // Load and process audio
    println!("Loading audio features...");
    let audio_features = audio::load_audio_features(
        &args.audio_file,
        128, // n_mels
        &device,
    )?;
    
    println!("Successfully loaded audio features with shape: {:?}", audio_features.shape());
    
    // Run either demonstration mode or full model inference
    if args.demo_mode || (!args.download && args.model_dir.is_none()) {
        run_demo_mode(&args, &audio_features)?;
    } else {
        run_full_model(&args, &audio_features, &device)?;
    }
    
    Ok(())
}

fn run_demo_mode(args: &Args, audio_features: &Tensor) -> Result<()> {
    println!("\n=== Voxtral Demo Mode ===");
    println!("Prompt: {}", args.prompt);
    println!("Audio processed: {} frames", audio_features.dim(2)?);
    println!("Temperature: {}", args.temperature);
    if let Some(top_p) = args.top_p {
        println!("Top-p: {}", top_p);
    }
    println!("Max new tokens: {}", args.max_new_tokens);
    
    // Simulate processing
    println!("\n[Simulated] Processing audio through Voxtral encoder...");
    println!("[Simulated] Projecting audio features to text space...");
    println!("[Simulated] Generating response with LLaMA...");
    
    // Mock output based on the audio file
    let mock_output = if args.audio_file.contains("hello") {
        "Hello! How are you doing today? This audio contains a greeting message."
    } else {
        "I can hear audio content that would be processed by the Voxtral model for transcription and understanding."
    };
    
    println!("\n--- Generated Output ---");
    println!("{}", mock_output);
    println!("--- End Output ---\n");
    
    println!("✓ Audio processing demonstration complete!");
    println!("\nTo use with a real model:");
    println!("1. Download Voxtral model weights");
    println!("2. Use --model-dir /path/to/weights");
    println!("3. Ensure proper tokenizer configuration");
    
    Ok(())
}

fn run_full_model(args: &Args, audio_features: &Tensor, device: &Device) -> Result<()> {
    println!("\n=== Voxtral Full Model Inference ===");
    
    // Determine model source
    let (model_files, tokenizer_file) = if args.download || args.model_dir.is_none() {
        println!("Downloading model from Hugging Face: {}", args.model_id);
        download_model(&args.model_id)?
    } else {
        let model_dir = args.model_dir.as_ref().unwrap();
        println!("Loading model from: {}", model_dir);
        load_local_model(model_dir)?
    };
    
    // Load model configuration
    println!("Loading model configuration...");
    let config = load_model_config(&model_files.0)?;
    
    // Load safetensors files
    println!("Loading model weights from safetensors...");
    let vb = load_model_weights(&model_files.1, device)?;
    
    // Create model
    println!("Creating Voxtral model...");
    let model = VoxtralForConditionalGeneration::new(&config, vb)?;
    
    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    
    // Create cache
    let mut _cache = VoxtralCache::new(true, DType::F32, &config.text_config, device)?;
    
    // Process audio through the model
    println!("Processing audio through Voxtral encoder...");
    let audio_embeds = model.get_audio_embeds(audio_features)?;
    println!("Audio embeddings shape: {:?}", audio_embeds.shape());
    
    // Tokenize input prompt
    println!("Tokenizing input prompt...");
    let prompt_tokens = tokenize_prompt(&tokenizer, &args.prompt, args.audio_token_id, device)?;
    
    // Generate response
    println!("Generating response...");
    let generated_tokens = model.generate(
        &prompt_tokens,
        Some(audio_features),
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )?;
    
    // Decode tokens with proper tokenizer
    let output_text = tokenizer.decode(&generated_tokens, true).map_err(E::msg)?;
    
    println!("\n--- Generated Output ---");
    println!("{}", output_text);
    println!("--- End Output ---\n");
    
    println!("✓ Full model inference complete!");
    
    Ok(())
}

// Model loading helper functions

/// Download model from Hugging Face
fn download_model(model_id: &str) -> Result<((PathBuf, Vec<PathBuf>), PathBuf)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    
    // Download configuration file
    let config_file = repo.get("config.json")?;
    
    // Download model files - look for safetensors
    let mut model_files = Vec::new();
    
    // Common Voxtral/Ultravox safetensors file patterns
    let safetensors_files = [
        "model.safetensors",
        "pytorch_model.safetensors", 
        "model-00001-of-00001.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    
    for filename in &safetensors_files {
        if let Ok(file) = repo.get(filename) {
            model_files.push(file);
        }
    }
    
    if model_files.is_empty() {
        anyhow::bail!("No safetensors files found in model repository {}", model_id);
    }
    
    // Download tokenizer
    let tokenizer_file = repo.get("tokenizer.json")
        .or_else(|_| repo.get("tokenizer/tokenizer.json"))?;
    
    println!("Downloaded {} safetensors files and tokenizer", model_files.len());
    
    Ok(((config_file, model_files), tokenizer_file))
}

/// Load model from local directory
fn load_local_model(model_dir: &str) -> Result<((PathBuf, Vec<PathBuf>), PathBuf)> {
    let model_path = PathBuf::from(model_dir);
    
    // Find config file
    let config_file = model_path.join("config.json");
    if !config_file.exists() {
        anyhow::bail!("config.json not found in {}", model_dir);
    }
    
    // Find safetensors files
    let mut model_files = Vec::new();
    let safetensors_patterns = [
        "model.safetensors",
        "pytorch_model.safetensors",
    ];
    
    for pattern in &safetensors_patterns {
        let file_path = model_path.join(pattern);
        if file_path.exists() {
            model_files.push(file_path);
        }
    }
    
    // Also check for sharded files
    let model_dir_read = std::fs::read_dir(&model_path)?;
    for entry in model_dir_read {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        if file_name_str.ends_with(".safetensors") && file_name_str.contains("model") {
            model_files.push(entry.path());
        }
    }
    
    if model_files.is_empty() {
        anyhow::bail!("No safetensors files found in {}", model_dir);
    }
    
    // Find tokenizer
    let tokenizer_file = model_path.join("tokenizer.json")
        .canonicalize()
        .or_else(|_| model_path.join("tokenizer/tokenizer.json").canonicalize())?;
    
    println!("Found {} safetensors files and tokenizer in local directory", model_files.len());
    
    Ok(((config_file, model_files), tokenizer_file))
}

/// Load model configuration from JSON file
fn load_model_config(config_file: &PathBuf) -> Result<VoxtralConfig> {
    let config_str = std::fs::read_to_string(config_file)?;
    
    // Try to parse as Voxtral config first, then fallback to creating default
    match serde_json::from_str::<serde_json::Value>(&config_str) {
        Ok(json) => {
            // Extract relevant config values or use defaults
            let audio_token_id = json.get("audio_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(128256) as usize;
                
            // Create config with defaults (in production, parse all fields)
            Ok(create_voxtral_config(audio_token_id))
        }
        Err(_) => {
            println!("Warning: Could not parse config.json, using defaults");
            Ok(create_voxtral_config(128256))
        }
    }
}

/// Load model weights from safetensors files
fn load_model_weights(model_files: &[PathBuf], device: &Device) -> Result<VarBuilder> {
    let dtype = DType::F32; // or F16 for memory efficiency
    
    println!("Loading {} safetensors files...", model_files.len());
    for file in model_files {
        println!("  - {}", file.display());
    }
    
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(model_files, dtype, device)? };
    Ok(vb)
}

/// Tokenize prompt with proper audio token handling
fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str, audio_token_id: usize, device: &Device) -> Result<Tensor> {
    // Add special audio token to prompt
    let prompt_with_audio = format!("{} <|audio|>", prompt);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt_with_audio, true).map_err(E::msg)?;
    let mut tokens = encoding.get_ids().to_vec();
    
    // Replace the <|audio|> token with the proper audio token ID
    // This is a simplified approach - in practice you'd need to handle this more carefully
    if let Some(last_token) = tokens.last_mut() {
        // Replace last token with audio token (simplified logic)
        *last_token = audio_token_id as u32;
    }
    
    // Convert to tensor
    let input_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
    
    Ok(input_ids)
}

fn create_voxtral_config(audio_token_id: usize) -> VoxtralConfig {
    // Create default audio encoder config
    let audio_config = VoxtralEncoderConfig::default();
    
    // Create LLaMA config for text model
    let text_config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 3584,
        intermediate_size: 9216,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: Some(4),
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        max_position_embeddings: 32768,
        use_flash_attn: false,
    };
    
    VoxtralConfig {
        audio_config,
        text_config,
        audio_token_id,
        projector_hidden_act: "gelu".to_string(),
    }
}

fn encode_prompt(prompt: &str, audio_token_id: usize, device: &Device) -> Result<Tensor> {
    // Simple tokenization (in real usage, use proper tokenizer)
    let mut tokens = vec![1]; // BOS token
    
    // Add some dummy tokens for the prompt
    for _ in prompt.chars().take(10) {
        tokens.push(2000 + (tokens.len() % 1000) as u32);
    }
    
    // Add audio token
    tokens.push(audio_token_id as u32);
    
    Ok(Tensor::new(tokens, device)?.unsqueeze(0)?)
}

fn decode_simple_tokens(tokens: &[u32]) -> String {
    // Simple decoding (in real usage, use proper tokenizer)
    format!("Generated {} tokens: [Audio transcription would appear here with proper tokenizer]", tokens.len())
}

/// Example function to demonstrate processing long audio files  
#[allow(dead_code)]
fn process_long_audio(
    model: &VoxtralForConditionalGeneration,
    audio_features: &Tensor,
    chunk_frames: usize,
    overlap_frames: usize,
    prompt: &str,
    args: &Args,
    device: &Device,
) -> Result<String> {
    let (_batch, _n_mels, total_frames) = audio_features.dims3()?;
    
    if total_frames <= chunk_frames {
        // Process as single chunk
        let input_ids = encode_prompt(prompt, args.audio_token_id, device)?;
        let tokens = model.generate(
            &input_ids,
            Some(audio_features),
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )?;
        return Ok(decode_simple_tokens(&tokens));
    }
    
    // Process in chunks using the model's chunked processing
    let audio_embeds = model.get_audio_embeds_chunked(
        audio_features,
        chunk_frames,
        overlap_frames,
    )?;
    
    // Generate using the full model pipeline
    let input_ids = encode_prompt(prompt, args.audio_token_id, device)?;
    let tokens = model.generate(
        &input_ids,
        Some(audio_features),
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )?;
    
    Ok(decode_simple_tokens(&tokens))
}