//! CosyVoice3 Text-to-Speech Example
//!
//! This example demonstrates text-to-speech synthesis using CosyVoice3.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cosyvoice3 --features="metal" -- \
//!     --text "Hello, this is a test." \
//!     --model-dir /path/to/CosyVoice3-0.5B-2512-Candle \
//!     --output output.wav
//! ```
//!
//! # Text Normalization
//!
//! For better TTS quality, text normalization is recommended. The official Python
//! implementation uses WeText for this purpose. You can integrate wetext-rs:
//!
//! 1. Add to Cargo.toml: `wetext-rs = "0.1.2"`
//! 2. Use `wetext_rs::Normalizer` to normalize text before tokenization
//!
//! Example:
//! ```rust,ignore
//! use wetext_rs::Normalizer;
//! let normalizer = Normalizer::new(false)?; // remove_erhua = false
//! let normalized_text = normalizer.normalize(text)?;
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Error as E, Result};
use clap::Parser;
use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::cosyvoice::{
    CausalHiFTGenerator, CausalMaskedDiffWithDiT, CosyVoice3LM, DiT, FlowConfig, HiFTConfig,
    MelSpectrogram, SamplingConfig,
};
use tokenizers::Tokenizer;

/// Special token IDs
#[allow(dead_code)]
const LLM_EOS_TOKEN: u32 = 151643;
const LLM_TASK_TOKEN: u32 = 151665;
#[allow(dead_code)]
const LLM_PAD_TOKEN: u32 = 151643;
#[allow(dead_code)]
const SPEECH_SOS_TOKEN: u32 = 6561;
#[allow(dead_code)]
const SPEECH_EOS_TOKEN: u32 = 6562;

/// Runtime configuration parsed from model's config.json
#[derive(Debug, Clone, serde::Deserialize)]
struct RuntimeConfig {
    pub sample_rate: usize,
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,
    pub spk_embed_dim: usize,
    #[allow(dead_code)]
    pub token_frame_rate: usize,
    pub token_mel_ratio: usize,
    pub chunk_size: usize,
    pub pre_lookahead_len: usize,
    pub dit: DiTRuntimeConfig,
    pub hift: HiFTRuntimeConfig,
    pub qwen2: Qwen2RuntimeConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct DiTRuntimeConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub ff_mult: usize,
    pub mel_dim: usize,
    pub spk_dim: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct HiFTRuntimeConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop_len: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub source_resblock_kernel_sizes: Vec<usize>,
    pub source_resblock_dilation_sizes: Vec<Vec<usize>>,
    pub conv_pre_look_right: usize,
    pub nsf_alpha: f64,
    pub nsf_sigma: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Qwen2RuntimeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

impl RuntimeConfig {
    fn to_llm_config(&self) -> candle_transformers::models::cosyvoice::CosyVoice3LMConfig {
        candle_transformers::models::cosyvoice::CosyVoice3LMConfig {
            llm_input_size: self.llm_input_size,
            llm_output_size: self.llm_output_size,
            speech_token_size: self.speech_token_size,
            mix_ratio: (5, 15),
            qwen2: candle_transformers::models::cosyvoice::Qwen2Config {
                hidden_size: self.qwen2.hidden_size,
                num_hidden_layers: self.qwen2.num_hidden_layers,
                num_attention_heads: self.qwen2.num_attention_heads,
                num_key_value_heads: self.qwen2.num_key_value_heads,
                intermediate_size: self.qwen2.intermediate_size,
                max_position_embeddings: 32768,
                rope_theta: self.qwen2.rope_theta,
                rms_norm_eps: self.qwen2.rms_norm_eps,
                vocab_size: self.qwen2.vocab_size,
                tie_word_embeddings: true,
            },
        }
    }

    fn to_hift_config(&self) -> HiFTConfig {
        HiFTConfig {
            in_channels: self.hift.in_channels,
            base_channels: self.hift.base_channels,
            nb_harmonics: self.hift.nb_harmonics,
            sampling_rate: self.sample_rate,
            nsf_alpha: self.hift.nsf_alpha,
            nsf_sigma: self.hift.nsf_sigma,
            upsample_rates: self.hift.upsample_rates.clone(),
            upsample_kernel_sizes: self.hift.upsample_kernel_sizes.clone(),
            istft_n_fft: self.hift.istft_n_fft,
            istft_hop_len: self.hift.istft_hop_len,
            resblock_kernel_sizes: self.hift.resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: self.hift.resblock_dilation_sizes.clone(),
            source_resblock_kernel_sizes: self.hift.source_resblock_kernel_sizes.clone(),
            source_resblock_dilation_sizes: self.hift.source_resblock_dilation_sizes.clone(),
            conv_pre_look_right: self.hift.conv_pre_look_right,
        }
    }

    fn to_flow_config(&self) -> FlowConfig {
        FlowConfig {
            input_size: self.dit.mel_dim,
            output_size: self.dit.mel_dim,
            vocab_size: self.speech_token_size,
            token_mel_ratio: self.token_mel_ratio,
            pre_lookahead_len: self.pre_lookahead_len,
            dit: candle_transformers::models::cosyvoice::DiTConfig {
                dim: self.dit.dim,
                depth: self.dit.depth,
                heads: self.dit.heads,
                dim_head: self.dit.dim_head,
                ff_mult: self.dit.ff_mult,
                mel_dim: self.dit.mel_dim,
                spk_dim: self.dit.spk_dim,
                static_chunk_size: self.chunk_size * self.token_mel_ratio,
            },
            cfm: candle_transformers::models::cosyvoice::CFMConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    /// Zero-shot voice cloning
    ZeroShot,
    /// Cross-lingual voice cloning
    CrossLingual,
    /// Instruction-based synthesis
    Instruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "CosyVoice3 Text-to-Speech CLI", long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Prompt text for zero-shot mode
    #[arg(long, default_value = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")]
    prompt_text: String,

    /// Prompt audio file path (WAV format)
    #[arg(long)]
    prompt_wav: Option<PathBuf>,

    /// Pre-extracted prompt features file (safetensors format)
    /// This can be used instead of prompt_wav when ONNX models are not available
    #[arg(long)]
    prompt_features: Option<PathBuf>,

    /// Model directory path
    #[arg(long)]
    model_dir: PathBuf,

    /// Output audio file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Inference mode
    #[arg(long, default_value = "zero-shot")]
    mode: Mode,

    /// Instruction text (only for instruct mode)
    #[arg(long)]
    instruct: Option<String>,

    /// Speech synthesis speed (0.5-2.0)
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Number of CFM sampling steps
    #[arg(long, default_value_t = 10)]
    n_timesteps: usize,

    /// Run on CPU rather than on GPU
    #[arg(long)]
    cpu: bool,

    /// Use f16 precision (Metal/CUDA only)
    #[arg(long)]
    f16: bool,

    /// Sampling temperature
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Top-k sampling
    #[arg(long, default_value_t = 25)]
    top_k: usize,

    /// Top-p sampling
    #[arg(long, default_value_t = 0.8)]
    top_p: f32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,

    /// Enable tracing
    #[arg(long)]
    tracing: bool,
}

/// CosyVoice3 model wrapper
struct CosyVoice3Model {
    llm: CosyVoice3LM,
    flow_decoder: CausalMaskedDiffWithDiT,
    vocoder: CausalHiFTGenerator,
    mel_extractor: MelSpectrogram,
    tokenizer: Tokenizer,
    config: RuntimeConfig,
    device: Device,
}

impl CosyVoice3Model {
    fn new(
        config: RuntimeConfig,
        llm_vb: VarBuilder,
        flow_vb: VarBuilder,
        hift_vb: VarBuilder,
        tokenizer: Tokenizer,
        device: Device,
    ) -> Result<Self> {
        println!("Loading LLM...");
        let llm_config = config.to_llm_config();
        let llm = CosyVoice3LM::new(&llm_config, llm_vb)?;

        println!("Loading Flow Decoder (DiT)...");
        let flow_config = config.to_flow_config();

        // Create DiT estimator
        let dit = DiT::new(flow_config.dit.clone(), flow_vb.pp("dit"))?;

        // Create Flow Decoder
        // input_embedding maps vocab -> output_size (mel_dim), not dit.dim
        let flow_decoder = CausalMaskedDiffWithDiT::new(
            flow_config.vocab_size,
            flow_config.output_size,  // input_size = mel_dim (80), not dit.dim
            flow_config.output_size,
            config.spk_embed_dim,
            flow_config.token_mel_ratio,
            flow_config.pre_lookahead_len,
            dit,
            flow_config.cfm.clone(),
            flow_vb.clone(),
        )?;

        println!("Loading HiFT Vocoder...");
        let hift_config = config.to_hift_config();
        let vocoder = CausalHiFTGenerator::new(hift_config, hift_vb)?;

        println!("Initializing Mel extractor...");
        let mel_extractor = MelSpectrogram::new_cosyvoice_speech_feat(&device)?;

        Ok(Self {
            llm,
            flow_decoder,
            vocoder,
            mel_extractor,
            tokenizer,
            config,
            device,
        })
    }

    /// Tokenize text using the Qwen2 tokenizer
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(E::msg)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Run zero-shot inference
    fn inference_zero_shot(
        &mut self,
        text: &str,
        prompt_text: &str,
        prompt_speech_tokens: &[u32],
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
        sampling_config: &SamplingConfig,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        // 1. Tokenize text
        // NOTE: In Python, text and prompt_text are passed separately to LLM.inference()
        // which concatenates them internally. We need to match this behavior.
        let text_tokens = self.tokenize(text)?;
        let prompt_text_tokens = self.tokenize(prompt_text)?;

        println!(
            "Text tokens: {} (prompt: {})",
            text_tokens.len(),
            prompt_text_tokens.len()
        );

        // 2. Convert to tensors - pass them separately, LLM will concatenate
        let text_tokens_tensor =
            Tensor::from_slice(&text_tokens, (1, text_tokens.len()), &self.device)?
                .to_dtype(DType::U32)?;
        let prompt_text_tensor =
            Tensor::from_slice(&prompt_text_tokens, (1, prompt_text_tokens.len()), &self.device)?
                .to_dtype(DType::U32)?;
        let prompt_speech_tensor = Tensor::from_slice(
            prompt_speech_tokens,
            (1, prompt_speech_tokens.len()),
            &self.device,
        )?
        .to_dtype(DType::U32)?;

        // 3. LLM inference - generate speech tokens
        println!("Running LLM inference...");
        let speech_tokens = self.llm.inference(
            &text_tokens_tensor,
            &prompt_text_tensor,
            &prompt_speech_tensor,
            sampling_config,
        )?;

        println!("Generated {} speech tokens", speech_tokens.len());

        if speech_tokens.is_empty() {
            anyhow::bail!("LLM generated no speech tokens");
        }

        // 4. Flow decoder - convert speech tokens to mel
        println!("Running Flow decoder...");
        let speech_tokens_tensor =
            Tensor::from_slice(&speech_tokens, (1, speech_tokens.len()), &self.device)?
                .to_dtype(DType::U32)?;

        let mel = self.flow_decoder.inference(
            &speech_tokens_tensor,
            &prompt_speech_tensor,
            prompt_mel,
            speaker_embedding,
            n_timesteps,
            false,
        )?;

        println!("Generated mel shape: {:?}", mel.shape());

        // Ensure mel is on the correct device
        let mel = mel.to_device(&self.device)?.to_dtype(DType::F32)?;

        // 5. Vocoder - convert mel to waveform
        println!("Running HiFT vocoder...");
        let waveform = self.vocoder.inference(&mel, true)?;

        println!("Generated waveform shape: {:?}", waveform.shape());

        Ok(waveform)
    }

    /// Sample rate
    fn sample_rate(&self) -> usize {
        self.config.sample_rate
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

    println!("CosyVoice3 Text-to-Speech");
    println!("========================");
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    // Device selection
    let device = candle_examples::device(args.cpu)?;
    println!("Device: {:?}", device);

    let dtype = if args.f16 {
        println!("Using f16 precision");
        DType::F16
    } else {
        DType::F32
    };

    // Load configuration
    let config_path = args.model_dir.join("config.json");
    println!("Loading config from {:?}", config_path);
    let config: RuntimeConfig =
        serde_json::from_reader(std::fs::File::open(&config_path)?).context("Failed to load config.json")?;

    if args.verbose {
        println!("Config: {:#?}", config);
    }

    // Load tokenizer
    let tokenizer_path = args.model_dir.join("tokenizer");
    println!("Loading tokenizer from {:?}", tokenizer_path);
    
    // Try to load tokenizer.json first, then fall back to building from vocab.json
    let tokenizer = if tokenizer_path.join("tokenizer.json").exists() {
        Tokenizer::from_file(tokenizer_path.join("tokenizer.json")).map_err(E::msg)?
    } else {
        // Build tokenizer from vocab.json + merges.txt
        println!("tokenizer.json not found, building from vocab.json + merges.txt...");
        let vocab_path = tokenizer_path.join("vocab.json");
        let merges_path = tokenizer_path.join("merges.txt");
        
        // Read vocab and parse merges
        let vocab_str = std::fs::read_to_string(&vocab_path)
            .context("Failed to read vocab.json")?;
        let _vocab: std::collections::HashMap<String, u32> = serde_json::from_str(&vocab_str)?;
        
        let _merges_str = std::fs::read_to_string(&merges_path)
            .context("Failed to read merges.txt")?;
        
        // Use tokenizers' BPE from_file
        use tokenizers::models::bpe::BPE;
        let bpe = BPE::from_file(&vocab_path.to_string_lossy(), &merges_path.to_string_lossy())
            .build()
            .map_err(E::msg)?;
        
        Tokenizer::new(bpe)
    };

    // Load model weights
    println!("Loading model weights...");
    let start = std::time::Instant::now();

    let llm_path = args.model_dir.join("llm.safetensors");
    let flow_path = args.model_dir.join("flow.safetensors");
    let hift_path = args.model_dir.join("hift.safetensors");

    let llm_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&llm_path], dtype, &device)? };
    let flow_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&flow_path], dtype, &device)? };
    let hift_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&hift_path], dtype, &device)? };

    println!("Weights loaded in {:?}", start.elapsed());

    // Create model
    let mut model = CosyVoice3Model::new(
        config,
        llm_vb,
        flow_vb,
        hift_vb,
        tokenizer,
        device.clone(),
    )?;

    println!("Model initialized");

    // Prepare sampling config
    let sampling_config = SamplingConfig {
        top_k: args.top_k,
        top_p: args.top_p,
        temperature: args.temperature,
        repetition_penalty: 1.0,
    };

    // Prepare prompt data
    // For now, we'll use placeholder data since ONNX models have compatibility issues
    // In production, these would come from:
    // - campplus.onnx: speaker embedding from prompt_wav
    // - speech_tokenizer_v3.onnx: speech tokens from prompt_wav
    // - mel extractor: prompt mel from prompt_wav
    
    println!("\nNote: Using placeholder prompt data.");
    println!("For full functionality, ONNX model support is needed for:");
    println!("  - campplus.onnx (speaker embedding)");
    println!("  - speech_tokenizer_v3.onnx (speech tokenization)");
    println!("");
    
    // Load prompt features if provided
    let (prompt_speech_tokens, prompt_mel, speaker_embedding) = if let Some(features_path) = &args.prompt_features {
        println!("Loading prompt features from {:?}", features_path);
        let features = candle::safetensors::load(features_path, &device)?;
        
        // Load speech tokens - may be saved as I32, convert to U32
        let prompt_tokens_tensor = features
            .get("prompt_speech_tokens")
            .context("Missing prompt_speech_tokens in features file")?;
        let prompt_tokens: Vec<u32> = if prompt_tokens_tensor.dtype() == DType::I64 {
            prompt_tokens_tensor.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|x| x as u32).collect()
        } else {
            // Assume I32
            prompt_tokens_tensor.flatten_all()?.to_vec1::<i32>()?.into_iter().map(|x| x as u32).collect()
        };
        let prompt_mel = features
            .get("prompt_mel")
            .context("Missing prompt_mel in features file")?
            .clone();
        let spk_embed = features
            .get("speaker_embedding")
            .context("Missing speaker_embedding in features file")?
            .clone();
        
        (prompt_tokens, prompt_mel, spk_embed)
    } else if let Some(wav_path) = &args.prompt_wav {
        // Try to extract features from audio
        println!("Loading prompt audio from {:?}", wav_path);
        let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(wav_path)
            .context("Failed to decode prompt audio")?;
        
        // Resample to 24kHz if needed
        let pcm_data = if sample_rate != model.sample_rate() as u32 {
            println!("Resampling from {} to {} Hz", sample_rate, model.sample_rate());
            candle_transformers::models::cosyvoice::resample(
                &Tensor::from_vec(pcm_data.clone(), pcm_data.len(), &device)?,
                sample_rate as usize,
                model.sample_rate(),
            )?
            .to_vec1()?
        } else {
            pcm_data
        };
        
        // Extract mel spectrogram
        let audio_tensor = Tensor::from_vec(pcm_data.clone(), pcm_data.len(), &device)?;
        let prompt_mel = model.mel_extractor.forward(&audio_tensor)?;
        println!("Extracted prompt mel shape: {:?}", prompt_mel.shape());
        
        // Use placeholder for speech tokens and speaker embedding
        // These would normally come from ONNX models
        println!("Warning: Using placeholder speech tokens and speaker embedding");
        println!("         Full ONNX support needed for proper voice cloning");
        
        // Calculate number of tokens based on mel length
        // mel_len = tokens * token_mel_ratio, so tokens = mel_len / token_mel_ratio
        let mel_t = prompt_mel.dim(2)?;
        let num_tokens = mel_t / 2; // token_mel_ratio = 2
        let prompt_speech_tokens: Vec<u32> = (0..num_tokens).map(|i| (i * 100) as u32 % 6561).collect();
        println!("Generated {} placeholder speech tokens for {} mel frames", num_tokens, mel_t);
        
        let speaker_embedding = Tensor::randn(0f32, 1.0, (1, 192), &device)?.to_dtype(dtype)?;
        
        // Reshape mel for flow decoder: [1, 80, T] -> [1, T*2, 80]
        // Note: We need to match the expected prompt_mel shape which is [1, T*token_mel_ratio, 80]
        // where T is the number of tokens
        let prompt_mel = prompt_mel.transpose(1, 2)?; // [1, T, 80]
        // Upsample by token_mel_ratio (2x) to match expected shape
        let prompt_mel = prompt_mel.unsqueeze(2)?; // [1, T, 1, 80]
        let prompt_mel = prompt_mel.broadcast_as((1, mel_t, 2, 80))?;
        let prompt_mel = prompt_mel.reshape((1, mel_t * 2, 80))?.to_dtype(dtype)?;
        
        (prompt_speech_tokens, prompt_mel, speaker_embedding)
    } else {
        // Use placeholder data
        println!("No prompt audio provided, using placeholder data");
        
        let prompt_speech_tokens: Vec<u32> = (0..50).map(|i| (i * 100) as u32 % 6561).collect();
        let prompt_mel = Tensor::randn(0f32, 1.0, (1, 100, 80), &device)?.to_dtype(dtype)?;
        let speaker_embedding = Tensor::randn(0f32, 1.0, (1, 192), &device)?.to_dtype(dtype)?;
        
        (prompt_speech_tokens, prompt_mel, speaker_embedding)
    };

    // Run inference
    println!("\nSynthesizing: \"{}\"", args.text);
    println!("Mode: {:?}", args.mode);

    let start = std::time::Instant::now();

    let waveform = match args.mode {
        Mode::ZeroShot => model.inference_zero_shot(
            &args.text,
            &args.prompt_text,
            &prompt_speech_tokens,
            &prompt_mel,
            &speaker_embedding,
            &sampling_config,
            args.n_timesteps,
        )?,
        Mode::CrossLingual => {
            // Cross-lingual uses the same flow but without prompt text tokens
            model.inference_zero_shot(
                &args.text,
                "",  // Empty prompt text for cross-lingual
                &prompt_speech_tokens,
                &prompt_mel,
                &speaker_embedding,
                &sampling_config,
                args.n_timesteps,
            )?
        }
        Mode::Instruct => {
            let instruct_text = args.instruct.as_deref().unwrap_or("You are a helpful assistant.<|endofprompt|>");
            model.inference_zero_shot(
                &args.text,
                instruct_text,
                &prompt_speech_tokens,
                &prompt_mel,
                &speaker_embedding,
                &sampling_config,
                args.n_timesteps,
            )?
        }
    };

    let inference_time = start.elapsed();

    // Extract PCM data
    let pcm = if waveform.dims().len() == 3 {
        waveform.squeeze(0)?.squeeze(0)?
    } else if waveform.dims().len() == 2 {
        waveform.squeeze(0)?
    } else {
        waveform
    };
    
    let pcm_data: Vec<f32> = pcm.to_dtype(DType::F32)?.to_vec1()?;

    // Calculate audio duration
    let audio_duration = pcm_data.len() as f32 / model.sample_rate() as f32;
    let rtf = inference_time.as_secs_f32() / audio_duration;

    println!("\nInference completed!");
    println!("  Inference time: {:.2}s", inference_time.as_secs_f32());
    println!("  Audio duration: {:.2}s", audio_duration);
    println!("  Real-time factor (RTF): {:.2}", rtf);

    // Normalize audio
    let pcm_normalized = candle_examples::audio::normalize_loudness(
        &Tensor::from_vec(pcm_data.clone(), pcm_data.len(), &device)?,
        model.sample_rate() as u32,
        true,
    )?;
    let pcm_final: Vec<f32> = pcm_normalized.to_vec1()?;

    // Save audio
    println!("Saving to {:?}", args.output);
    let mut output_file = std::fs::File::create(&args.output)?;
    candle_examples::wav::write_pcm_as_wav(
        &mut output_file,
        &pcm_final,
        model.sample_rate() as u32,
    )?;

    println!("Done!");

    Ok(())
}
