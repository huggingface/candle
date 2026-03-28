//! Kokoro-82M TTS inference example.
//!
//! Downloads the model from `hexgrad/Kokoro-82M` and synthesizes speech.
//!
//! ```bash
//! cargo run --example kokoro -- --text "Hello, world!" --voice jf_alpha --output out.wav
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::kokoro::{Config, KokoroModel, VoiceEmbeddings};
use clap::Parser;
use hf_hub::api::sync::Api;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Kokoro-82M TTS inference")]
struct Args {
    /// Text to synthesize.
    #[arg(long, default_value = "Hello, this is Kokoro speaking.")]
    text: String,

    /// Voice name (e.g. jf_alpha, af_heart, am_echo). Defaults to jf_alpha.
    #[arg(long, default_value = "jf_alpha")]
    voice: String,

    /// Language code for G2P (ja, en, fr, es, zh, ...).
    #[arg(long, default_value = "en")]
    language: String,

    /// Speaking speed multiplier (1.0 = normal).
    #[arg(long, default_value_t = 1.0)]
    speed: f64,

    /// Output WAV file path.
    #[arg(long, default_value = "kokoro_output.wav")]
    output: String,

    /// HuggingFace model repo (default: hexgrad/Kokoro-82M).
    #[arg(long, default_value = "hexgrad/Kokoro-82M")]
    model_repo: String,

    /// Run on CPU even if Metal/CUDA is available.
    #[arg(long)]
    cpu: bool,

    /// Use f16 weights instead of f32.
    #[arg(long)]
    f16: bool,
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            println!("Using Metal device.");
            return Ok(dev);
        }
    }
    Ok(Device::Cpu)
}

/// Minimal G2P: for demo purposes, map characters to Kokoro phoneme indices using
/// the vocab from config.json. Production usage should use a proper G2P library.
fn text_to_phoneme_ids(text: &str, vocab: &std::collections::HashMap<String, u32>) -> Vec<u32> {
    // Prepend BOS (0) and append EOS (0); map each character to its vocab entry.
    // Characters not in vocab are skipped.
    let mut ids = vec![0u32]; // BOS
    for ch in text.chars() {
        let s = ch.to_string();
        if let Some(&id) = vocab.get(&s) {
            ids.push(id);
        }
    }
    ids.push(0u32); // EOS
    ids
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = device(args.cpu)?;
    let dtype = if args.f16 { DType::F16 } else { DType::F32 };

    println!("Downloading model from {} …", args.model_repo);
    let api = Api::new()?;
    let repo = api.model(args.model_repo.clone());

    let config_path = repo.get("config.json")?;
    let model_path = repo.get("kokoro-v1_0.pth")?;

    // Load config
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    let vocab = extract_vocab(&config_str)?;

    println!("Loading model weights …");
    // Try safetensors first, fall back to PyTorch pickle
    let vb = if let Ok(st_path) = repo.get("kokoro-v1_0.safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[st_path], dtype, &dev)? }
    } else {
        // Load from PyTorch .pth via candle's pickle reader
        let tensors = candle::pickle::read_all(model_path)?;
        let tensor_map: std::collections::HashMap<String, Tensor> = tensors.into_iter().collect();
        VarBuilder::from_tensors(tensor_map, dtype, &dev)
    };

    println!("Building model …");
    let model = KokoroModel::new(&config, vb)?;

    // Load voice embedding
    let voice_emb = load_voice_embedding(&args.voice, &args.model_repo, &dev, dtype)?;

    // Convert text to phoneme IDs
    let ids = text_to_phoneme_ids(&args.text, &vocab);
    if ids.len() <= 2 {
        bail!("No phonemes found for the given text. Use a proper G2P library for production.");
    }
    println!("Phoneme IDs ({}): {:?}", ids.len(), &ids[..ids.len().min(20)]);

    let phoneme_tensor = Tensor::from_vec(ids, (1, 0), &dev)?; // shape filled by from_vec
    // Re-shape: phoneme_tensor is [seq_len], need [1, seq_len]
    let seq_len = phoneme_tensor.elem_count();
    let phoneme_tensor = phoneme_tensor.reshape((1, seq_len))?;

    println!("Synthesizing speech …");
    let audio = model.forward(&phoneme_tensor, &voice_emb, args.speed)?;

    // Convert to f32 for WAV writing
    let pcm = audio.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("Generated {} samples ({:.2}s at 24 kHz)", pcm.len(), pcm.len() as f64 / 24_000.0);

    // Write WAV
    let mut output = std::fs::File::create(&args.output)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
    println!("Saved to {}", args.output);

    Ok(())
}

/// Extract the `vocab` dict from config.json (maps phoneme strings to integer IDs).
fn extract_vocab(config_str: &str) -> Result<std::collections::HashMap<String, u32>> {
    #[derive(serde::Deserialize)]
    struct WithVocab {
        vocab: std::collections::HashMap<String, u32>,
    }
    let parsed: WithVocab = serde_json::from_str(config_str)?;
    Ok(parsed.vocab)
}

/// Download and load a Kokoro voice embedding from HuggingFace.
/// Voice files are stored as `voices/{name}.pt` in the model repo.
fn load_voice_embedding(
    voice: &str,
    model_repo: &str,
    dev: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let api = Api::new()?;
    let repo = api.model(model_repo.to_string());
    let voice_path = repo.get(&format!("voices/{voice}.pt"))?;

    // Load PyTorch tensor via candle's pickle reader.
    // Voice files contain a single tensor of shape [511, 1, 256].
    // We use the first entry (index 0) as the reference style embedding.
    let tensors = candle::pickle::read_all(&voice_path)?;
    if tensors.is_empty() {
        bail!("No tensors found in voice file for '{voice}'");
    }
    let emb = tensors[0].1.to_dtype(dtype)?.to_device(dev)?;

    // Voice tensor shape: [511, 1, 256] → take first row → [1, 256]
    let emb = if emb.dims().len() == 3 {
        emb.i(0)?.reshape((1, 256))?
    } else if emb.dims().len() == 2 {
        emb
    } else {
        bail!("Unexpected voice embedding shape: {:?}", emb.dims());
    };

    println!("Loaded voice '{}' embedding shape: {:?}", voice, emb.dims());
    Ok(emb)
}
