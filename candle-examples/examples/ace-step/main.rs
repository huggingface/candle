//! ACE-Step 1.5 music generation example.
//!
//! ```bash
//! # DiT-only (base/sft)
//! cargo run --release --example ace-step --features metal -- \
//!     --prompt "A gentle acoustic guitar melody" --duration 10
//!
//! # LM+DiT (turbo)
//! cargo run --release --example ace-step --features metal -- \
//!     --infer-type lm-dit --lm-model ACE-Step/acestep-5Hz-lm-0.6B \
//!     --model ACE-Step/acestep-v15-turbo-shift3 \
//!     --prompt "A gentle guitar" --duration 30
//!
//! # Cover mode
//! cargo run --release --example ace-step --features metal -- \
//!     --reference-audio original.wav --prompt "electronic remix" --duration 30
//! ```

mod pipeline;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ace_step::{lm::LmConfig, sampling, AceStepConfig, VaeConfig};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo};
use pipeline::{AceStepPipeline, GenerationParams, LmMetadata};

#[derive(Parser, Debug)]
#[command(author, version, about = "ACE-Step 1.5 Text2Music generation")]
struct Args {
    #[arg(
        long,
        default_value = "A gentle acoustic guitar melody with soft piano accompaniment"
    )]
    prompt: String,
    #[arg(long, default_value = "")]
    lyrics: String,
    #[arg(long, default_value_t = 10.0)]
    duration: f64,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    num_steps: Option<usize>,
    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    shift: f64,
    #[arg(long, default_value = "ace_step_output.wav")]
    out_file: String,
    #[arg(long, default_value = "ACE-Step/acestep-v15-sft")]
    model: String,
    #[arg(long)]
    tracing: bool,
    /// "dit" (default) or "lm-dit"
    #[arg(long, default_value = "dit")]
    infer_type: String,
    #[arg(long)]
    lm_model: Option<String>,
    #[arg(long, default_value_t = 0.85)]
    temperature: f64,
    #[arg(long)]
    top_p: Option<f64>,
    /// Reference audio WAV for cover mode (structural source)
    #[arg(long)]
    reference_audio: Option<String>,
    /// Separate timbre audio WAV for cover mode; if omitted, uses reference-audio
    #[arg(long)]
    timbre_audio: Option<String>,
    /// Source audio WAV for repaint mode (selectively regenerate a region)
    #[arg(long)]
    repaint_audio: Option<String>,
    /// Repaint region start time in seconds
    #[arg(long, default_value_t = 0.0)]
    repaint_start: f64,
    /// Repaint region end time in seconds (required with --repaint-audio)
    #[arg(long)]
    repaint_end: Option<f64>,
    /// Cover condition strength (0.0–1.0); at this fraction of steps, switches to non-cover condition
    #[arg(long, default_value_t = 1.0)]
    audio_cover_strength: f64,
    /// CFG interval start (guidance applied when timestep >= this)
    #[arg(long, default_value_t = 0.0)]
    cfg_interval_start: f64,
    /// CFG interval end (guidance applied when timestep <= this)
    #[arg(long, default_value_t = 1.0)]
    cfg_interval_end: f64,
    /// Sampling method: "ode" (deterministic) or "sde" (stochastic)
    #[arg(long, default_value = "ode")]
    infer_method: String,
    /// Latent frames per VAE-decoder chunk (tiled decode). Lower = less peak
    /// memory at the cost of compute overhead. Set to 0 to let the decoder
    /// pick its default (128 frames ≈ 5.1 s of audio).
    #[arg(long, default_value_t = 0)]
    vae_chunk_frames: usize,
    /// Overlap in latent frames on each side of a VAE-decoder chunk. Must
    /// satisfy `chunk > 2 * overlap`. Set to 0 to use the default (16).
    #[arg(long, default_value_t = 0)]
    vae_chunk_overlap: usize,
}

/// Write stereo `(2, T)` tensor as 16-bit WAV with loudness normalization,
/// peak clamp, and fade-out.
fn write_stereo_wav(path: &str, audio: &Tensor, sample_rate: u32) -> Result<()> {
    let audio = audio.to_dtype(DType::F32)?;

    // BS1770 loudness normalization to -14 LUFS per channel.
    // VAE output can be very quiet or uneven — this brings it to a
    // consistent broadcast-standard level.
    let left = candle_examples::audio::normalize_loudness(&audio.get(0)?, sample_rate, true)?;
    let right = candle_examples::audio::normalize_loudness(&audio.get(1)?, sample_rate, true)?;

    let mut left = left.to_vec1::<f32>()?;
    let mut right = right.to_vec1::<f32>()?;
    for s in left.iter_mut().chain(right.iter_mut()) {
        *s = s.clamp(-1.0, 1.0);
    }
    // 50ms fade-out at end to avoid click
    let fade = (sample_rate as usize / 20).min(left.len());
    for i in 0..fade {
        let g = 1.0 - (i as f32 / fade as f32); // 1→0
        let idx = left.len() - fade + i;
        left[idx] *= g;
        right[idx] *= g;
    }
    let mut f = std::fs::File::create(path)?;
    candle_examples::wav::write_pcm_as_wav_stereo(&mut f, &[&left, &right], sample_rate)?;
    Ok(())
}

/// Load audio file as `(1, 2, T)` tensor, resampling if needed.
fn load_reference_audio(
    path: &str,
    target_sr: u32,
    dtype: DType,
    device: &candle::Device,
) -> Result<Tensor> {
    let (left, right, sr) = {
        #[cfg(feature = "symphonia")]
        {
            let (ch, sr) = candle_examples::audio::pcm_decode_all_channels(path)?;
            if ch.len() == 1 {
                (ch[0].clone(), ch[0].clone(), sr)
            } else {
                (ch[0].clone(), ch[1].clone(), sr)
            }
        }
        #[cfg(not(feature = "symphonia"))]
        {
            let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
            let (h, ch) = candle_examples::wav::read_pcm_from_wav::<_, f32>(&mut f)?;
            if ch.len() == 1 {
                (ch[0].clone(), ch[0].clone(), h.sample_rate)
            } else {
                (ch[0].clone(), ch[1].clone(), h.sample_rate)
            }
        }
    };
    let (left, right) = if sr != target_sr && sr > 0 {
        #[cfg(feature = "rubato")]
        {
            println!("Resampling {sr}Hz → {target_sr}Hz...");
            let l: Vec<f32> = candle_examples::audio::resample(&left, sr, target_sr)?
                .into_iter()
                .map(|v| v as f32)
                .collect();
            let r: Vec<f32> = candle_examples::audio::resample(&right, sr, target_sr)?
                .into_iter()
                .map(|v| v as f32)
                .collect();
            (l, r)
        }
        #[cfg(not(feature = "rubato"))]
        {
            println!(
                "Warning: {sr}Hz audio, model expects {target_sr}Hz. Enable 'rubato' feature."
            );
            (left, right)
        }
    } else {
        (left, right)
    };

    let n = left.len();
    let d = &candle::Device::Cpu;
    let audio = Tensor::stack(
        &[
            &Tensor::from_vec(left, n, d)?,
            &Tensor::from_vec(right, n, d)?,
        ],
        0,
    )?
    .unsqueeze(0)?;
    Ok(audio.to_dtype(dtype)?.to_device(device)?)
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let lyrics = if !args.lyrics.is_empty() && std::path::Path::new(&args.lyrics).is_file() {
        std::fs::read_to_string(&args.lyrics)?
    } else {
        args.lyrics.clone()
    };
    let _guard = if args.tracing {
        let (cl, g) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(cl).init();
        Some(g)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    // BF16 matches the original model weights dtype, saving ~50% memory.
    // F16 is insufficient (DiT produces silence, LM overflows).
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let is_turbo = args.model.contains("turbo");
    let needs_encoder = args.reference_audio.is_some()
        || args.repaint_audio.is_some()
        || args.timbre_audio.is_some();

    // Auto-detect shift from model name if not explicitly set
    let shift = if (args.shift - 1.0).abs() < 1e-6 && is_turbo {
        if args.model.contains("shift3") {
            3.0
        } else if args.model.contains("shift2") {
            2.0
        } else if args.model.contains("shift1") {
            1.0
        } else {
            3.0
        }
    } else {
        args.shift
    };

    println!(
        "Model: {}, device: {:?}, dtype: {:?}, duration: {}s{}",
        args.model,
        device,
        dtype,
        args.duration,
        if is_turbo {
            format!(", shift: {shift}")
        } else {
            String::new()
        },
    );

    // ---- Download files ----
    let api = Api::new()?;
    let main_repo = api.repo(Repo::model(args.model.clone()));
    let shared_repo = api.repo(Repo::model("ACE-Step/Ace-Step1.5".to_string()));

    // DiT weights: single file for 2B models, sharded for XL (4B)
    let dit_weight_files: Vec<std::path::PathBuf> = match main_repo.get("model.safetensors") {
        Ok(p) => vec![p],
        Err(_) => {
            candle_examples::hub_load_safetensors(&main_repo, "model.safetensors.index.json")?
        }
    };
    let dit_config_file = main_repo.get("config.json")?;
    let silence_latent_pt = main_repo.get("silence_latent.pt")?;
    let vae_weights = shared_repo.get("vae/diffusion_pytorch_model.safetensors")?;
    let vae_config_file = shared_repo.get("vae/config.json")?;
    let text_encoder_weights = shared_repo.get("Qwen3-Embedding-0.6B/model.safetensors")?;
    let text_encoder_config_file = shared_repo.get("Qwen3-Embedding-0.6B/config.json")?;
    let tokenizer_file = shared_repo.get("Qwen3-Embedding-0.6B/tokenizer.json")?;

    // ---- Load pipeline ----
    println!("Loading models...");
    let dit_config: AceStepConfig =
        serde_json::from_reader(std::fs::File::open(&dit_config_file)?)?;
    let vae_config: VaeConfig = serde_json::from_reader(std::fs::File::open(&vae_config_file)?)?;
    let text_config: candle_transformers::models::qwen3::Config =
        serde_json::from_reader(std::fs::File::open(&text_encoder_config_file)?)?;

    let dit_refs: Vec<&std::path::Path> = dit_weight_files.iter().map(|p| p.as_path()).collect();
    let dit_vb = unsafe { VarBuilder::from_mmaped_safetensors(&dit_refs, dtype, &device)? };
    let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_weights], dtype, &device)? };
    let text_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&text_encoder_weights], dtype, &device)? };
    let text_vb = text_vb.rename_f(|n| n.strip_prefix("model.").unwrap_or(n).to_string());

    let silence_latent = pipeline::load_silence_latent(
        &silence_latent_pt,
        dit_config.audio_acoustic_hidden_dim,
        dtype,
        &device,
    )?;

    let mut pipeline = AceStepPipeline::new(
        &dit_config,
        &vae_config,
        &text_config,
        dit_vb,
        vae_vb,
        text_vb,
        &tokenizer_file,
        silence_latent,
        needs_encoder,
    )?;

    // ---- Optional LM pipeline ----
    let (lm_hints, lm_metadata) = if args.infer_type == "lm-dit" {
        let lm_repo_name = args
            .lm_model
            .as_deref()
            .unwrap_or("ACE-Step/acestep-5Hz-lm-0.6B");
        println!("Loading LM: {lm_repo_name}...");
        let lm_repo = api.repo(Repo::model(lm_repo_name.to_string()));
        let lm_config: candle_transformers::models::qwen3::Config =
            serde_json::from_reader(std::fs::File::open(lm_repo.get("config.json")?)?)?;
        let lm_weight_files: Vec<std::path::PathBuf> = match lm_repo.get("model.safetensors") {
            Ok(p) => vec![p],
            Err(_) => {
                candle_examples::hub_load_safetensors(&lm_repo, "model.safetensors.index.json")?
            }
        };
        let lm_refs: Vec<&std::path::Path> = lm_weight_files.iter().map(|p| p.as_path()).collect();
        let lm_vb = unsafe { VarBuilder::from_mmaped_safetensors(&lm_refs, dtype, &device)? };
        pipeline.load_lm(&lm_config, lm_vb, &lm_repo.get("tokenizer.json")?)?;

        let lm_cfg = LmConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed.unwrap_or(42),
            ..Default::default()
        };

        println!("Generating audio codes...");
        let t = std::time::Instant::now();
        let (hints, metadata) =
            pipeline.generate_lm_hints(&args.prompt, &lyrics, args.duration, &lm_cfg)?;
        println!(
            "LM: {:.2}s, metadata: {:?}",
            t.elapsed().as_secs_f32(),
            metadata
        );
        println!("LM hints: {:?}", hints.shape());

        // Feed LM metadata back into DiT text prompt (matching Python's
        // _update_metadata_from_lm → _extract_caption_and_language flow).
        let lm_meta = LmMetadata {
            bpm: metadata.get("bpm").cloned(),
            timesignature: metadata.get("timesignature").cloned(),
            keyscale: metadata.get("keyscale").cloned(),
            caption: metadata.get("caption").cloned(),
            language: metadata
                .get("language")
                .or_else(|| metadata.get("vocal_language"))
                .cloned(),
        };

        // Free LM weights before DiT denoising to reclaim GPU memory.
        pipeline.unload_lm();

        // LM audio code hints are only effective with turbo models (which were
        // distilled to use them). Base/SFT models benefit from LM metadata
        // (caption, BPM, keyscale) but produce noise when LM hints replace
        // the silence context — matching the Python pipeline behaviour where
        // non-turbo models always run with infer_type="dit".
        let lm_hints = if is_turbo { Some(hints) } else {
            println!("Note: non-turbo model — using LM metadata only (hints skipped)");
            None
        };

        (lm_hints, Some(lm_meta))
    } else {
        if is_turbo {
            println!("Hint: turbo models work best with --infer-type lm-dit --lm-model ACE-Step/acestep-5Hz-lm-0.6B");
        }
        (None, None)
    };

    // ---- Generate ----
    let infer_method = match args.infer_method.as_str() {
        "sde" => sampling::InferMethod::Sde,
        _ => sampling::InferMethod::Ode,
    };

    let params = GenerationParams {
        duration_secs: args.duration,
        seed: args.seed,
        num_steps: args.num_steps,
        guidance_scale: args.guidance_scale,
        shift,
        is_turbo,
        lm_hints_25hz: lm_hints,
        lm_metadata,
        audio_cover_strength: args.audio_cover_strength,
        cfg_interval: (args.cfg_interval_start, args.cfg_interval_end),
        infer_method,
        vae_chunk_frames: (args.vae_chunk_frames > 0).then_some(args.vae_chunk_frames),
        vae_chunk_overlap: (args.vae_chunk_overlap > 0).then_some(args.vae_chunk_overlap),
    };

    let t = std::time::Instant::now();
    let output = if let Some(ref path) = args.repaint_audio {
        let end = args.repaint_end.unwrap_or(args.duration);
        println!(
            "Repaint mode: {path} [{:.1}s..{:.1}s]...",
            args.repaint_start, end
        );
        let src_audio =
            load_reference_audio(path, vae_config.sampling_rate as u32, dtype, &device)?;
        pipeline.repaint(
            &args.prompt,
            &lyrics,
            &src_audio,
            args.repaint_start,
            end,
            &params,
        )?
    } else if let Some(ref path) = args.reference_audio {
        if params.lm_hints_25hz.is_none() {
            println!(
                "Note: cover mode works best with LM. Add --infer-type lm-dit \
                 --lm-model ACE-Step/acestep-5Hz-lm-0.6B for proper remixing."
            );
        }
        println!("Encoding reference audio: {path}...");
        let ref_audio =
            load_reference_audio(path, vae_config.sampling_rate as u32, dtype, &device)?;
        let timbre = args
            .timbre_audio
            .as_ref()
            .map(|p| {
                println!("Encoding timbre audio: {p}...");
                load_reference_audio(p, vae_config.sampling_rate as u32, dtype, &device)
            })
            .transpose()?;
        println!("Cover mode: denoising...");
        pipeline.cover(&args.prompt, &lyrics, &ref_audio, timbre.as_ref(), &params)?
    } else {
        println!("Denoising...");
        pipeline.text2music(&args.prompt, &lyrics, &params)?
    };
    println!("Generation: {:.2}s", t.elapsed().as_secs_f32());

    write_stereo_wav(&args.out_file, &output.audio, output.sample_rate)?;
    println!("Saved {} ({:.1}s audio)", args.out_file, args.duration);
    Ok(())
}
