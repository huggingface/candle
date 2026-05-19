//! Wav2Vec2ForCTC — CTC speech recognition example
//!
//! This example runs Wav2Vec2ForCTC inference on a 16 kHz mono WAV file.
//!
//! ## Usage
//!
//! ```bash
//! # Download from HuggingFace Hub
//! cargo run --example wav2vec2 --release -- \
//!     --model facebook/wav2vec2-base-960h \
//!     --audio  path/to/audio.wav
//!
//! # Use a local model directory
//! cargo run --example wav2vec2 --release -- \
//!     --model-dir /path/to/wav2vec2-model \
//!     --audio      path/to/audio.wav
//! ```
//!
//! The audio must be 16 kHz, mono, 16-bit PCM WAV.
//! Multi-channel files are mixed down to mono automatically.
//!
//! ## Notes on weight_norm
//!
//! HuggingFace wav2vec2 models store the positional conv embedding weights
//! using `nn.utils.weight_norm` parametrization.  Before loading into Candle
//! you must export a clean safetensors file with weight_norm removed:
//!
//! ```python
//! from transformers import Wav2Vec2ForCTC
//! import torch
//! from safetensors.torch import save_file
//!
//! model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
//! model.eval()
//! torch.nn.utils.parametrize.remove_parametrizations(
//!     model.wav2vec2.encoder.pos_conv_embed.conv, "weight")
//! save_file({k: v.float() for k, v in model.state_dict().items()
//!            if "parametrizations" not in k},
//!           "wav2vec2_clean.safetensors")
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::wav2vec2::{
    normalise_audio, Wav2Vec2Config, Wav2Vec2ForCTC,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(author, version, about = "Wav2Vec2ForCTC speech recognition")]
struct Args {
    /// HuggingFace model ID (e.g. "facebook/wav2vec2-base-960h").
    #[arg(long, default_value = "facebook/wav2vec2-base-960h")]
    model: String,

    /// Path to a local model directory with safetensors + config.json.
    /// Takes precedence over --model.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Path to a clean safetensors file (weight_norm already removed).
    /// Takes precedence over --model-dir/--model for weight loading.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Path to a 16 kHz mono WAV file to transcribe.
    #[arg(long)]
    audio: PathBuf,

    /// Run on CPU even if a GPU backend is available.
    #[arg(long)]
    cpu: bool,
}

// ---------------------------------------------------------------------------
// WAV reader (16-bit PCM, mono mix-down)
// ---------------------------------------------------------------------------

fn read_wav_f32(path: &PathBuf) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot open WAV: {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(spec.bits_per_sample == 16, "Only 16-bit PCM WAV is supported");
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .enumerate()
            .filter_map(|(i, s): (usize, _)| if i % channels == 0 { Some(s.unwrap() as f32 / 32768.0) } else { None })
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .enumerate()
            .filter_map(|(i, s): (usize, _)| if i % channels == 0 { Some(s.unwrap()) } else { None })
            .collect(),
    };
    Ok((samples, spec.sample_rate))
}

// ---------------------------------------------------------------------------
// Vocab loader
// ---------------------------------------------------------------------------

fn load_vocab(dir: &PathBuf) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(dir.join("vocab.json"))?;
    let map: std::collections::HashMap<String, usize> = serde_json::from_str(&text)?;
    let mut vocab = vec![String::new(); map.len()];
    for (token, id) in &map {
        if *id < vocab.len() {
            vocab[*id] = token.clone();
        }
    }
    Ok(vocab)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    println!("Device: {device:?}");

    // --- Resolve model directory -------------------------------------------
    let model_dir: PathBuf = match args.model_dir {
        Some(d) => d,
        None => {
            println!("Fetching '{}' from HuggingFace Hub...", args.model);
            let api = Api::new()?;
            let repo = api.repo(Repo::new(args.model.clone(), RepoType::Model));
            let config = repo.get("config.json")?;
            repo.get("vocab.json")?;
            // NOTE: caller must provide --weights pointing to a clean safetensors;
            // or pre-export with weight_norm removed (see module docs).
            config.parent().unwrap().to_path_buf()
        }
    };

    // --- Load config -------------------------------------------------------
    let cfg: Wav2Vec2Config = {
        let text = std::fs::read_to_string(model_dir.join("config.json"))?;
        serde_json::from_str(&text)?
    };
    println!(
        "Config: hidden={} layers={} heads={} ffn={} cnn_norm={} do_stable_ln={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
        cfg.intermediate_size, cfg.feat_extract_norm, cfg.do_stable_layer_norm
    );

    // --- Load weights -------------------------------------------------------
    let weights_path = match &args.weights {
        Some(p) => p.clone(),
        None => {
            let p = model_dir.join("model.safetensors");
            anyhow::ensure!(
                p.exists(),
                "model.safetensors not found at {}. \
                 See module docs for weight_norm removal steps.",
                p.display()
            );
            p
        }
    };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)?
    };
    let model = Wav2Vec2ForCTC::load(&cfg, vb)?;
    println!("Model loaded.");

    // --- Vocab --------------------------------------------------------------
    let vocab = load_vocab(&model_dir)?;
    println!("Vocab size: {}", vocab.len());

    // --- Audio --------------------------------------------------------------
    let (samples, sample_rate) = read_wav_f32(&args.audio)?;
    anyhow::ensure!(
        sample_rate == 16000,
        "Expected 16 kHz audio, got {sample_rate} Hz. Resample first."
    );
    println!(
        "Audio: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );

    let n = samples.len();
    let audio = Tensor::from_vec(samples, (n,), &device)?.unsqueeze(0)?; // [1, N]
    let audio = normalise_audio(&audio)?;

    // --- Inference ----------------------------------------------------------
    let t0 = std::time::Instant::now();
    let logits = model.forward(&audio, None)?;
    println!(
        "Inference: {}ms  (logits {:?})",
        t0.elapsed().as_millis(),
        logits.shape()
    );

    // --- Decode -------------------------------------------------------------
    let texts = model.decode_ctc(&logits, &vocab)?;
    println!("\nTranscription:\n{}", texts[0]);

    Ok(())
}
