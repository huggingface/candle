#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use hf_hub::api::sync::Api;
use qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, StreamOptions, TranscribeOptions};

pub mod audio;
pub mod config;
pub mod inference;
pub mod model;
pub mod processor;
pub mod qwen3_asr;

const SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Input audio file in wav format.
    #[arg(long)]
    input: Option<String>,

    /// Model id to load from Hugging Face.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B")]
    model_id: String,

    /// Optional forced language (e.g. English, Chinese).
    #[arg(long)]
    language: Option<String>,

    /// Run in streaming mode.
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Streaming chunk size in seconds.
    #[arg(long, default_value_t = 1.0)]
    chunk_size_sec: f32,

    /// Max new tokens per decode.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Optional rolling audio window in seconds for streaming mode.
    #[arg(long)]
    audio_window_sec: Option<f32>,

    /// Optional rolling text window in tokens for streaming mode.
    #[arg(long)]
    text_window_tokens: Option<usize>,

    /// Print intermediate streaming outputs.
    #[arg(long, default_value_t = false)]
    print_intermediate: bool,
}

fn resolve_audio_path(input: Option<String>) -> Result<std::path::PathBuf> {
    if let Some(input) = input {
        if let Some(sample) = input.strip_prefix("sample:") {
            let api = Api::new()?;
            let dataset = api.dataset("Narsil/candle-examples".to_string());
            return Ok(dataset.get(&format!("samples_{sample}.wav"))?);
        }
        return Ok(std::path::PathBuf::from(input));
    }

    println!("No audio file submitted: downloading sample:jfk from Narsil/candle-examples");
    let api = Api::new()?;
    let dataset = api.dataset("Narsil/candle-examples".to_string());
    Ok(dataset.get("samples_jfk.wav")?)
}

fn load_audio(input: Option<String>) -> Result<Vec<f32>> {
    let path = resolve_audio_path(input)?;
    let (audio, sample_rate) = candle_examples::audio::pcm_decode(&path)?;
    if sample_rate == SAMPLE_RATE_HZ {
        return Ok(audio);
    }

    println!(
        "resampling input from {sample_rate}Hz to {SAMPLE_RATE_HZ}Hz: {}",
        path.display()
    );
    Ok(linear_resample(&audio, sample_rate, SAMPLE_RATE_HZ))
}

fn linear_resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if samples.is_empty() || from_hz == 0 || to_hz == 0 || from_hz == to_hz {
        return samples.to_vec();
    }

    let ratio = to_hz as f64 / from_hz as f64;
    let out_len = (samples.len() as f64 * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx0 = src.floor() as usize;
        let idx1 = idx0.saturating_add(1).min(samples.len().saturating_sub(1));
        let t = (src - idx0 as f64) as f32;
        let y0 = samples[idx0];
        let y1 = samples[idx1];
        out.push(y0 + (y1 - y0) * t);
    }

    out
}

fn run_offline(model: &Qwen3Asr, wav: &[f32], args: &Args) -> Result<()> {
    let opts = TranscribeOptions {
        context: Batch::one(String::new()),
        language: Batch::one(args.language.clone()),
        return_timestamps: false,
        max_new_tokens: args.max_new_tokens,
        max_batch_size: 32,
        chunk_max_sec: None,
        bucket_by_length: false,
    };
    let out = model.transcribe(
        vec![AudioInput::Waveform {
            samples: wav,
            sample_rate: SAMPLE_RATE_HZ,
        }],
        opts,
    )?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn run_stream(model: &Qwen3Asr, wav: &[f32], args: &Args) -> Result<()> {
    let opts = StreamOptions {
        context: String::new(),
        language: args.language.clone(),
        chunk_size_sec: args.chunk_size_sec,
        max_new_tokens: args.max_new_tokens,
        audio_window_sec: args.audio_window_sec,
        text_window_tokens: args.text_window_tokens,
        ..Default::default()
    };
    let mut stream = model.start_stream(opts)?;
    let chunk_len = (args.chunk_size_sec * SAMPLE_RATE_HZ as f32).round() as usize;
    if chunk_len == 0 {
        anyhow::bail!("chunk_size_sec too small for 16kHz audio");
    }

    let mut start = 0usize;
    while start < wav.len() {
        let end = (start + chunk_len).min(wav.len());
        if let Some(step) = stream.push_audio_chunk(&AudioInput::Waveform {
            samples: &wav[start..end],
            sample_rate: SAMPLE_RATE_HZ,
        })? {
            if args.print_intermediate {
                println!("{}", serde_json::to_string(&step)?);
            }
        }
        start = end;
    }

    let final_tx = stream.finish()?;
    println!("{}", serde_json::to_string_pretty(&final_tx)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let wav = load_audio(args.input.clone())?;
    let asr = Qwen3Asr::from_pretrained(args.model_id.as_str(), &device, &LoadOptions::default())?;
    if args.stream {
        run_stream(&asr, &wav, &args)
    } else {
        run_offline(&asr, &wav, &args)
    }
}
