//! Example: Qwen + SNAC TTS Integration
//!
//! This example demonstrates how to create a Text-to-Speech system using:
//! - Qwen 0.5B language model for text-to-audio token generation
//! - SNAC codec for audio token decoding to waveform
//!
//! Usage:
//! ```bash
//! cargo run --example qwen_snac_tts_example -- \
//!   --qwen-model-path ./models/qwen0.5b \
//!   --snac-model-path ./models/snac_24khz \
//!   --text "Hello, this is a test of SNAC-based text to speech synthesis."
//! ```

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{qwen2, snac, snac_tts_integration};
use clap::{Arg, Command};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

struct QwenSnacTts {
    qwen_model: qwen2::Model,
    tokenizer: Tokenizer,
    snac_codec: snac_tts_integration::SnacTtsCodec,
    device: Device,
    max_seq_len: usize,
}

impl QwenSnacTts {
    fn new(
        qwen_model: qwen2::Model,
        tokenizer: Tokenizer,
        snac_codec: snac_tts_integration::SnacTtsCodec,
        device: Device,
    ) -> Self {
        Self {
            qwen_model,
            tokenizer,
            snac_codec,
            device,
            max_seq_len: 2048,
        }
    }

    /// Generate speech from text input
    fn synthesize(&mut self, text: &str, temperature: f64, top_p: f64) -> Result<Tensor> {
        println!("Tokenizing input text...");
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let input_tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        
        println!("Generating audio tokens with Qwen...");
        
        // Generate audio tokens using the language model
        // In a real implementation, this would be trained to output SNAC token sequences
        let audio_tokens = self.generate_audio_tokens(&input_tokens, temperature, top_p)?;
        
        println!("Decoding tokens to audio with SNAC...");
        
        // Convert the generated tokens to audio using SNAC
        let audio_waveform = self.snac_codec.tokens_to_audio(&audio_tokens)?;
        
        Ok(audio_waveform)
    }

    /// Generate audio tokens from text tokens
    /// Note: In a real implementation, this would use a trained model that maps text to audio tokens
    fn generate_audio_tokens(&mut self, text_tokens: &Tensor, _temperature: f64, _top_p: f64) -> Result<Tensor> {
        // This is a placeholder implementation
        // In practice, you would:
        // 1. Use a trained Qwen model that has learned to generate SNAC tokens
        // 2. Implement proper sampling with temperature and top-p
        // 3. Handle start/end tokens and padding appropriately
        
        let batch_size = text_tokens.dim(0)?;
        let num_codebooks = self.snac_codec.num_codebooks();
        
        // For demonstration, create dummy audio tokens
        // In reality, these would come from the trained model
        let seq_length = 100; // ~4 seconds of audio at 24kHz
        let shape = (batch_size, num_codebooks, seq_length);
        
        // Generate random tokens as a placeholder
        // Real implementation would use: self.qwen_model.forward(&text_tokens)?
        let dummy_tokens = Tensor::rand(0f32, 4096f32, shape, &self.device)?.to_dtype(candle::DType::U32)?;
        
        println!("Generated {} audio token sequences of length {}", num_codebooks, seq_length);
        
        Ok(dummy_tokens)
    }
}

/// Load Qwen model for TTS
fn load_qwen_model(model_path: &str, device: &Device) -> Result<(qwen2::Model, Tokenizer)> {
    println!("Loading Qwen model from: {}", model_path);
    
    // Load tokenizer
    let api = Api::new()?;
    let repo = api.model(model_path.to_string());
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    
    // Load model config
    let config_filename = repo.get("config.json")?;
    let config: qwen2::Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    
    // Load model weights  
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F16, device)? };
    
    // Create model
    let model = qwen2::Model::load(&vb, &config)?;
    
    Ok((model, tokenizer))
}

/// Load SNAC codec model
fn load_snac_codec(model_path: &str, device: &Device) -> Result<snac_tts_integration::SnacTtsCodec> {
    println!("Loading SNAC codec from: {}", model_path);
    
    // Load SNAC model
    let api = Api::new()?;
    let repo = api.model(model_path.to_string());
    
    // Load config
    let config_filename = repo.get("config.json")?;
    let config: snac::Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    
    // Load model weights
    let weights_filename = repo.get("pytorch_model.bin")?;
    let vb = VarBuilder::from_pickle(&weights_filename, DType::F32, device)?;
    
    // Create codec
    let codec = snac_tts_integration::SnacTtsCodec::new(&config, vb)?;
    
    Ok(codec)
}

/// Save audio tensor to WAV file
fn save_audio_to_wav(audio: &Tensor, sample_rate: usize, filename: &str) -> Result<()> {
    println!("Saving audio to: {}", filename);
    
    // Convert tensor to Vec<f32>
    let audio_data = audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    
    // Create WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(filename, spec)?;
    
    for sample in audio_data {
        let sample_i16 = (sample * i16::MAX as f32) as i16;
        writer.write_sample(sample_i16)?;
    }
    
    writer.finalize()?;
    println!("Audio saved successfully!");
    
    Ok(())
}

fn main() -> Result<()> {
    let matches = Command::new("qwen-snac-tts")
        .about("Generate speech using Qwen + SNAC TTS")
        .arg(Arg::new("qwen-model-path")
            .long("qwen-model-path")
            .value_name("PATH")
            .help("Path to Qwen model directory")
            .required(true))
        .arg(Arg::new("snac-model-path")
            .long("snac-model-path")
            .value_name("PATH")
            .help("Path to SNAC model directory")
            .required(true))
        .arg(Arg::new("text")
            .long("text")
            .value_name("TEXT")
            .help("Text to synthesize")
            .required(true))
        .arg(Arg::new("output")
            .long("output")
            .short('o')
            .value_name("FILE")
            .help("Output WAV file")
            .default_value("output.wav"))
        .arg(Arg::new("temperature")
            .long("temperature")
            .value_name("FLOAT")
            .help("Generation temperature")
            .default_value("0.7"))
        .arg(Arg::new("top-p")
            .long("top-p")
            .value_name("FLOAT")
            .help("Top-p sampling parameter")
            .default_value("0.9"))
        .arg(Arg::new("cpu")
            .long("cpu")
            .help("Use CPU instead of GPU")
            .action(clap::ArgAction::SetTrue))
        .get_matches();

    let qwen_model_path = matches.get_one::<String>("qwen-model-path").unwrap();
    let snac_model_path = matches.get_one::<String>("snac-model-path").unwrap();
    let text = matches.get_one::<String>("text").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let temperature: f64 = matches.get_one::<String>("temperature").unwrap().parse()?;
    let top_p: f64 = matches.get_one::<String>("top-p").unwrap().parse()?;
    let use_cpu = matches.get_flag("cpu");

    // Setup device
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    
    println!("Using device: {:?}", device);

    // Load models
    let (qwen_model, tokenizer) = load_qwen_model(qwen_model_path, &device)?;
    let snac_codec = load_snac_codec(snac_model_path, &device)?;

    // Create TTS system
    let mut tts_system = QwenSnacTts::new(qwen_model, tokenizer, snac_codec, device);

    // Display codec information
    let codec_info = tts_system.snac_codec.codec_info();
    println!("SNAC Codec Info:");
    println!("  Sample Rate: {} Hz", codec_info.sample_rate);
    println!("  Codebooks: {}", codec_info.num_codebooks);
    println!("  Compression Ratio: {}:1", codec_info.compression_ratio);

    // Synthesize speech
    println!("\nSynthesizing: \"{}\"", text);
    let audio = tts_system.synthesize(text, temperature, top_p)?;
    
    println!("Generated audio shape: {:?}", audio.shape());

    // Save to file
    save_audio_to_wav(&audio, codec_info.sample_rate, output_file)?;

    // Display generation info
    let duration = tts_system.snac_codec.tokens_to_duration(100); // Assuming 100 tokens generated
    println!("Generated {:.2} seconds of audio", duration);
    println!("Synthesis complete!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loading() {
        // Test SNAC config creation
        let config = snac::Config::default_tts();
        assert_eq!(config.sampling_rate, 24000);
        assert!(!config.encoder_rates.is_empty());
    }

    #[test]
    fn test_memory_estimation() {
        use snac_tts_integration::utils::estimate_memory_usage;
        
        let estimate = estimate_memory_usage(10.0, 24000, 4, 1);
        assert!(estimate.audio_samples > 0);
        assert!(estimate.token_count > 0);
        assert!(estimate.estimated_bytes > 0);
    }
}