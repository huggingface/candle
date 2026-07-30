//! Qwen3-TTS — text-to-speech example for candle.
//!
//! Supports three synthesis modes:
//!
//! * **CustomVoice** – one of 9 preset speakers.
//! * **VoiceDesign** – voice described in natural language.
//! * **VoiceClone**  – voice cloned from a short reference WAV file.
//!
//! Run with `--help` for full usage.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::collections::HashMap;
use std::path::PathBuf;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_tts::{
    self,
    codec::{Decoder12Hz, Decoder12HzConfig, Decoder12HzState},
    compute_dtype_for_device, codes_to_tensor,
    config::ParsedModelConfig,
    talker::{
        codec_tokens, Language, Speaker, TalkerConfig, TalkerModel,
    },
    CodePredictor, CodePredictorConfig, AnyKVCache,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Qwen3-TTS: text-to-speech with candle", long_about = None)]
struct Args {
    /// HuggingFace model ID or local directory
    #[arg(long, default_value = "Qwen/Qwen3-TTS-0.6B-CustomVoice")]
    model_id: String,

    /// Text to synthesize
    #[arg(long, default_value = "Hello, this is Qwen three T T S.")]
    text: String,

    /// Preset speaker (CustomVoice only): ryan, serena, vivian, aiden, uncle_fu,
    /// ono_anna, sohee, eric, dylan
    #[arg(long, default_value = "ryan")]
    speaker: String,

    /// Target language: english, chinese, japanese, korean, german, french,
    /// russian, portuguese, spanish, italian
    #[arg(long, default_value = "english")]
    language: String,

    /// Voice description for VoiceDesign models
    #[arg(long)]
    instruct: Option<String>,

    /// Reference WAV file for voice cloning (Base models)
    #[arg(long)]
    ref_audio: Option<PathBuf>,

    /// Reference transcript for ICL voice cloning (requires --ref-audio)
    #[arg(long)]
    ref_text: Option<String>,

    /// Pre-encoded reference codec codes for ICL voice cloning.
    /// Shape must be [T_frames, 16] u32, stored as a safetensors file
    /// (key "codes").  Encode with the Mimi encoder or use a pre-built file.
    #[arg(long)]
    ref_codes: Option<PathBuf>,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-k
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p (nucleus sampling)
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f64,

    /// Maximum codec frames to generate (~12.5 frames/sec)
    #[arg(long, default_value_t = 2048)]
    max_frames: usize,

    /// Random seed (deterministic when set)
    #[arg(long)]
    seed: Option<u64>,

    /// Force CPU even if GPU is available
    #[arg(long)]
    cpu: bool,

    /// Use streaming (incremental) decoder — emits audio one frame at a time.
    /// Reduces latency; batch mode (default) has marginally better quality.
    #[arg(long)]
    streaming: bool,
}

// ── Device selection ──────────────────────────────────────────────────────────

fn select_device(cpu: bool) -> candle::Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::cuda_if_available(0) {
            if d.is_cuda() {
                eprintln!("Using CUDA");
                return Ok(d);
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            eprintln!("Using Metal");
            return Ok(d);
        }
    }
    eprintln!("Using CPU");
    Ok(Device::Cpu)
}

// ── Model loading helpers ─────────────────────────────────────────────────────

struct ModelFiles {
    model_weights: PathBuf,
    model_dir: PathBuf,
    config: Option<PathBuf>,
}

fn resolve_files(model_id: &str) -> anyhow::Result<ModelFiles> {
    let local = std::path::Path::new(model_id);
    if local.is_dir() {
        Ok(ModelFiles {
            model_weights: local.join("model.safetensors"),
            model_dir: local.to_path_buf(),
            config: {
                let p = local.join("config.json");
                if p.exists() { Some(p) } else { None }
            },
        })
    } else {
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
        let model_weights = repo.get("model.safetensors")?;
        let model_dir = model_weights.parent().unwrap().to_path_buf();
        Ok(ModelFiles {
            model_weights,
            model_dir,
            config: repo.get("config.json").ok(),
        })
    }
}

/// Load a Tokenizer from a model directory.
/// Tries `tokenizer.json` first, then falls back to building a Qwen2 BPE
/// tokenizer from `vocab.json` + `merges.txt`.
fn load_tokenizer(model_dir: &std::path::Path) -> anyhow::Result<Tokenizer> {
    // Fast path: pre-built tokenizer.json
    let tj = model_dir.join("tokenizer.json");
    if tj.exists() {
        return Tokenizer::from_file(&tj)
            .map_err(|e| anyhow::anyhow!("tokenizer.json load error: {e}"));
    }

    // Qwen2-style: build BPE from vocab.json + merges.txt
    let vocab_path  = model_dir.join("vocab.json");
    let merges_path = model_dir.join("merges.txt");
    if vocab_path.exists() && merges_path.exists() {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;

        let bpe = BPE::from_file(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
        )
        .build()
        .map_err(|e| anyhow::anyhow!("BPE build error: {e}"))?;

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_decoder(Some(ByteLevelDecoder::default()))
            .with_post_processor(Some(ByteLevelProcessor::default()));
        return Ok(tokenizer);
    }

    anyhow::bail!(
        "No tokenizer found in {}. Expected tokenizer.json, or vocab.json + merges.txt.",
        model_dir.display()
    );
}

fn load_flat_weights(
    path: &std::path::Path,
    device: &Device,
    dtype: DType,
) -> candle::Result<HashMap<String, Tensor>> {
    let raw = candle::safetensors::load(path, device)?;
    // Cast BF16 → dtype (F32 on CPU; keep BF16 on GPU)
    raw.into_iter()
        .map(|(k, t)| {
            let t = if t.dtype() == DType::BF16 && dtype == DType::F32 {
                t.to_dtype(DType::F32)?
            } else if t.dtype() != dtype {
                t.to_dtype(dtype)?
            } else {
                t
            };
            Ok((k, t))
        })
        .collect()
}

fn filter_prefix(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    weights
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(prefix)
                .map(|s| (s.to_string(), v.clone()))
        })
        .collect()
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

fn tokenize(tokenizer: &Tokenizer, text: &str) -> candle::Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| candle::Error::Msg(format!("tokenizer error: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

// ── Generation helpers ────────────────────────────────────────────────────────

fn apply_temperature(logits: &Tensor, temp: f64) -> candle::Result<Tensor> {
    if (temp - 1.0).abs() < 1e-6 || temp <= 0.0 {
        Ok(logits.clone())
    } else {
        logits / temp
    }
}

fn top_k_filter(logits: &Tensor, k: usize) -> candle::Result<Tensor> {
    let (_, vocab) = logits.dims2()?;
    let k = k.min(vocab);
    let (sorted, _) = logits.sort_last_dim(false)?;
    let threshold = sorted.narrow(1, k - 1, 1)?;
    let mask = logits.ge(&threshold.broadcast_as(logits.shape())?)?;
    let neg_inf =
        Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as(logits.shape())?;
    mask.where_cond(logits, &neg_inf)
}

fn top_p_filter(logits: &Tensor, p: f64) -> candle::Result<Tensor> {
    let vocab = logits.dim(1)?;
    let (sorted, _) = logits.sort_last_dim(false)?;
    let sorted_probs = candle_nn::ops::softmax_last_dim(&sorted)?;
    let cumsum = sorted_probs.cumsum(1)?;
    let shifted = cumsum.narrow(1, 0, vocab - 1)?;
    let zeros = Tensor::zeros((logits.dim(0)?, 1), DType::F32, logits.device())?;
    let shifted_cs = Tensor::cat(&[&zeros, &shifted], 1)?;
    let thr =
        Tensor::new(&[p as f32], logits.device())?.broadcast_as(shifted_cs.shape())?;
    let remove = shifted_cs.ge(&thr)?;
    let inf = Tensor::new(&[f32::INFINITY], logits.device())?.broadcast_as(sorted.shape())?;
    let kept = remove.where_cond(&inf, &sorted)?;
    let min_kept = kept.min(candle::D::Minus1)?.unsqueeze(1)?;
    let keep = logits.ge(&min_kept.broadcast_as(logits.shape())?)?;
    let neg_inf =
        Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as(logits.shape())?;
    keep.where_cond(logits, &neg_inf)
}

fn suppress_control_tokens(logits: &Tensor, eos: u32) -> candle::Result<Tensor> {
    let vocab = logits.dim(1)?;
    let start = vocab - 1024;
    let mut mask = vec![0.0f32; vocab];
    for i in start..vocab {
        if i as u32 != eos {
            mask[i] = 1.0;
        }
    }
    let m = Tensor::new(mask.as_slice(), logits.device())?
        .unsqueeze(0)?
        .broadcast_as(logits.shape())?;
    let zeros = Tensor::zeros(logits.shape(), DType::F32, logits.device())?;
    let neg_inf =
        Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as(logits.shape())?;
    m.gt(&zeros)?.where_cond(&neg_inf, logits)
}

fn apply_rep_penalty(logits: &Tensor, seen: &[u32], penalty: f64) -> candle::Result<Tensor> {
    if (penalty - 1.0).abs() < 1e-9 || seen.is_empty() {
        return Ok(logits.clone());
    }
    let vocab = logits.dim(1)?;
    let pf = penalty as f32;
    let mut mask = vec![0.0f32; vocab];
    for &t in seen {
        if (t as usize) < vocab {
            mask[t as usize] = 1.0;
        }
    }
    let pm = Tensor::new(mask.as_slice(), logits.device())?
        .unsqueeze(0)?
        .broadcast_as(logits.shape())?;
    let zeros = Tensor::zeros(logits.shape(), DType::F32, logits.device())?;
    let is_pos = logits.gt(&zeros)?;
    let pos_f = Tensor::new(&[1.0 / pf], logits.device())?.broadcast_as(logits.shape())?;
    let neg_f = Tensor::new(&[pf], logits.device())?.broadcast_as(logits.shape())?;
    let factor = is_pos.where_cond(&pos_f, &neg_f)?;
    let ones = Tensor::ones(logits.shape(), DType::F32, logits.device())?;
    let penalized = pm.gt(&zeros)?.where_cond(&factor, &ones)?;
    logits * penalized
}

/// Simple deterministic-seed multinomial sampler (PCG-based).
struct RngState(u64);
impl RngState {
    fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64)
                .unwrap_or(42)
        });
        Self(s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }

    fn next_f32(&mut self) -> f32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        let out = xs.rotate_right(rot);
        (out as f32) / (u32::MAX as f32)
    }
}

fn sample_token(
    logits: &Tensor,
    args: &Args,
    rng: &mut RngState,
    seen: &[u32],
    token_count: usize,
    min_new: usize,
    eos: u32,
) -> candle::Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = apply_temperature(&logits, args.temperature)?;
    let logits = apply_rep_penalty(&logits, seen, args.repetition_penalty)?;
    let logits = suppress_control_tokens(&logits, eos)?;

    // Suppress EOS for the first min_new tokens
    let logits = if token_count < min_new {
        let vocab = logits.dim(1)?;
        let mut m = vec![1.0f32; vocab];
        m[eos as usize] = 0.0;
        let mask = Tensor::new(m.as_slice(), logits.device())?
            .unsqueeze(0)?
            .broadcast_as(logits.shape())?;
        let zeros = Tensor::zeros(logits.shape(), DType::F32, logits.device())?;
        let neg_inf =
            Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as(logits.shape())?;
        mask.gt(&zeros)?.where_cond(&logits, &neg_inf)?
    } else {
        logits
    };

    let logits = if args.temperature < 0.01 {
        return Ok(logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0]);
    } else {
        logits
    };

    let logits = if args.top_k > 0 { top_k_filter(&logits, args.top_k)? } else { logits };
    let logits = if args.top_p < 1.0 { top_p_filter(&logits, args.top_p)? } else { logits };

    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let cumsum = probs.cumsum(1)?;
    let u = rng.next_f32();
    let u_t = Tensor::new(&[u], logits.device())?
        .unsqueeze(0)?
        .broadcast_as(cumsum.shape())?;
    let mask = cumsum.ge(&u_t)?;
    let vocab = probs.dim(1)?;
    let positions: Vec<f32> = (0..vocab).map(|i| i as f32 + 1.0).collect();
    let pos_t = Tensor::new(positions.as_slice(), logits.device())?
        .unsqueeze(0)?
        .broadcast_as(mask.shape())?;
    let large =
        Tensor::new(&[vocab as f32 + 1.0], logits.device())?.broadcast_as(mask.shape())?;
    Ok(mask.where_cond(&pos_t, &large)?.argmin(candle::D::Minus1)?.to_vec1::<u32>()?[0])
}

// ── WAV output ────────────────────────────────────────────────────────────────

fn save_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

// ── Loading reference audio (voice clone) ────────────────────────────────────

fn load_wav_f32(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    use hound::WavReader;
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?,
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32).map_err(|e| anyhow::anyhow!("{e}")))
            .collect::<anyhow::Result<Vec<_>>>()?,
        _ => anyhow::bail!(
            "Unsupported WAV format: {:?} {}‑bit",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };
    Ok((samples, spec.sample_rate))
}

// ── Build trailing-text tensor ────────────────────────────────────────────────

fn build_trailing_text(
    model: &TalkerModel,
    input_ids: &[u32],
) -> candle::Result<(Tensor, usize, Tensor)> {
    let trailing = if input_ids.len() > 1 {
        let rest = model.get_projected_text_embeddings(&input_ids[1..])?;
        let eos = model.get_tts_eos_embed()?;
        Tensor::cat(&[&rest, &eos], 1)?
    } else {
        model.get_tts_eos_embed()?
    };
    let len = trailing.dim(1)?;
    let pad = model.get_tts_pad_embed()?;
    Ok((trailing, len, pad))
}

// ── Main generation loop ──────────────────────────────────────────────────────

fn generate(
    model: &TalkerModel,
    cp: &CodePredictor,
    decoder: &Decoder12Hz,
    input_ids: &[u32],
    prefill_fn: impl Fn(&mut Vec<AnyKVCache>) -> candle::Result<(Tensor, Tensor)>,
    args: &Args,
    device: &Device,
) -> anyhow::Result<Vec<f32>> {
    let eos = codec_tokens::CODEC_EOS;
    const MIN_NEW: usize = 2;
    const REPEAT_STOP: usize = 10;

    let (trailing, trailing_len, tts_pad) = build_trailing_text(model, input_ids)?;

    let mut kv_caches = model.new_kv_caches(args.max_frames + 256);
    let (hidden, logits) = prefill_fn(&mut kv_caches)?;
    let prefill_len = hidden.dim(1)?;
    let mut offset = prefill_len;
    let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

    let logits_2d = logits.squeeze(1)?;
    let mut rng = RngState::new(args.seed);
    let mut seen: Vec<u32> = Vec::new();
    let mut sem_token = sample_token(&logits_2d, args, &mut rng, &seen, 0, MIN_NEW, eos)?;
    seen.push(sem_token);

    let mut all_codes: Vec<Vec<u32>> = Vec::new();
    let mut cp_caches = cp.new_kv_caches();
    let mut last_token = sem_token;
    let mut repeat_count = 0usize;

    for frame in 0..args.max_frames {
        if sem_token == eos {
            eprintln!("EOS at frame {frame}");
            break;
        }

        // Repetition detection
        if sem_token == last_token {
            repeat_count += 1;
            if repeat_count >= REPEAT_STOP && seen.len() >= MIN_NEW {
                eprintln!("Stopping: token {sem_token} repeated {repeat_count} times");
                break;
            }
        } else {
            repeat_count = 0;
        }
        last_token = sem_token;

        let sem_embed = model.get_codec_embedding(sem_token)?;
        let acoustic_codes =
            cp.generate_acoustic_codes(&last_hidden, &sem_embed, &mut cp_caches)?;

        // Collect frame
        let acoustics: Vec<u32> = acoustic_codes.to_vec1()?;
        let mut frame_codes = vec![sem_token];
        frame_codes.extend(&acoustics);
        all_codes.push(frame_codes);

        if frame == args.max_frames - 1 {
            break;
        }

        // Build next step embedding
        let acoustic_sum =
            cp.get_acoustic_embeddings_sum_from_tensor(&acoustic_codes)?;
        let summed = sem_embed.add(&acoustic_sum)?;
        let text_add = if frame < trailing_len {
            trailing.i((.., frame..frame + 1, ..))?
        } else {
            tts_pad.clone()
        };
        let step_in = summed.add(&text_add)?;

        let (h, new_logits) =
            model.generate_step_with_embed(&step_in, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        let l2d = new_logits.squeeze(1)?;
        let next = sample_token(&l2d, args, &mut rng, &seen, seen.len(), MIN_NEW, eos)?;
        seen.push(next);
        sem_token = next;
    }

    eprintln!("Generated {} codec frames", all_codes.len());

    let codes_tensor = codes_to_tensor(&all_codes, device)?;
    let waveform = decoder.decode(&codes_tensor)?;
    let audio: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    Ok(audio)
}

/// Sum codec embeddings for all 16 groups from reference codes.
///
/// Group 0 (semantic) → talker codec embedding table.
/// Groups 1–15 (acoustic) → code-predictor per-group embedding tables.
/// Result: `[1, T_frames, talker.hidden_size]`.
fn sum_ref_codec_embeddings(
    talker: &TalkerModel,
    cp: &CodePredictor,
    ref_codes: &Tensor, // [T_frames, 16] u32
) -> candle::Result<Tensor> {
    let mut summed = talker.embed_ref_semantic_codes(ref_codes)?; // [1, T, H]
    for group in 1usize..16 {
        let group_codes = ref_codes.i((.., group))?.contiguous()?;
        let group_embed = cp.embed_codes_for_group(group - 1, &group_codes)?;
        summed = summed.broadcast_add(&group_embed)?;
    }
    Ok(summed)
}

/// Full generation loop for ICL voice cloning.
///
/// Does a two-phase prefill:
/// 1. `prefill_voice_clone(icl_mode=true)` — installs role + codec structure in KV cache.
/// 2. `prefill_icl(icl_embed)` — runs reference audio + text through the transformer.
/// Then generates tokens with the same loop as `generate`.
fn generate_icl(
    model: &TalkerModel,
    cp: &CodePredictor,
    decoder: &Decoder12Hz,
    input_ids: &[u32],
    ref_text_ids: &[u32],
    ref_codec_embeds: &Tensor,
    language: Language,
    args: &Args,
    device: &Device,
) -> anyhow::Result<Vec<f32>> {
    let eos = codec_tokens::CODEC_EOS;
    const MIN_NEW: usize = 2;
    const REPEAT_STOP: usize = 10;

    let (trailing, trailing_len, tts_pad) = build_trailing_text(model, input_ids)?;

    let mut kv_caches = model.new_kv_caches(args.max_frames + 512);

    // Phase 1: voice-clone structure prefill (icl_mode=true — omits first text token)
    let speaker_embed = Tensor::zeros((1, model.config().hidden_size), candle::DType::F32, device)?;
    let (_phase1_hidden, _phase1_logits) =
        model.prefill_voice_clone(input_ids, &speaker_embed, language, true, &mut kv_caches)?;
    let phase1_len = _phase1_hidden.dim(1)?;
    let mut offset = phase1_len;

    // Phase 2: ICL prefill — reference audio + text
    let (icl_embed, _icl_trailing) =
        model.build_icl_prompt(input_ids, ref_text_ids, ref_codec_embeds)?;
    let icl_len = icl_embed.dim(1)?;
    eprintln!("ICL embed: {} frames ({} phase1 + {} icl)", offset + icl_len, offset, icl_len);

    let (last_hidden, logits) = model.prefill_icl(&icl_embed, &mut kv_caches, offset)?;
    offset += icl_len;

    // Generation loop (identical to generate())
    let logits_2d = logits.squeeze(1)?;
    let mut rng = RngState::new(args.seed);
    let mut seen: Vec<u32> = Vec::new();
    let mut sem_token = sample_token(&logits_2d, args, &mut rng, &seen, 0, MIN_NEW, eos)?;
    seen.push(sem_token);

    let mut all_codes: Vec<Vec<u32>> = Vec::new();
    let mut cp_caches = cp.new_kv_caches();
    let mut last_hidden = last_hidden;
    let mut last_token = sem_token;
    let mut repeat_count = 0usize;

    for frame in 0..args.max_frames {
        if sem_token == eos {
            eprintln!("EOS at frame {frame}");
            break;
        }
        if sem_token == last_token {
            repeat_count += 1;
            if repeat_count >= REPEAT_STOP && seen.len() >= MIN_NEW {
                eprintln!("Stopping: token {sem_token} repeated {repeat_count} times");
                break;
            }
        } else {
            repeat_count = 0;
        }
        last_token = sem_token;

        let sem_embed = model.get_codec_embedding(sem_token)?;
        let acoustic_codes = cp.generate_acoustic_codes(&last_hidden, &sem_embed, &mut cp_caches)?;

        let acoustics: Vec<u32> = acoustic_codes.to_vec1()?;
        let mut frame_codes = vec![sem_token];
        frame_codes.extend(&acoustics);
        all_codes.push(frame_codes);

        if frame == args.max_frames - 1 { break; }

        let acoustic_sum = cp.get_acoustic_embeddings_sum_from_tensor(&acoustic_codes)?;
        let summed = sem_embed.add(&acoustic_sum)?;
        let text_add = if frame < trailing_len {
            trailing.i((.., frame..frame + 1, ..))?  
        } else {
            tts_pad.clone()
        };
        let step_in = summed.add(&text_add)?;

        let (h, new_logits) = model.generate_step_with_embed(&step_in, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        let l2d = new_logits.squeeze(1)?;
        let next = sample_token(&l2d, args, &mut rng, &seen, seen.len(), MIN_NEW, eos)?;
        seen.push(next);
        sem_token = next;
    }

    eprintln!("Generated {} codec frames", all_codes.len());

    let codes_tensor = codes_to_tensor(&all_codes, device)?;
    let waveform = decoder.decode(&codes_tensor)?;
    let audio: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    Ok(audio)
}

/// Streaming generation loop.
///
/// Identical to `generate` but decodes each codec frame incrementally via
/// [`Decoder12Hz::decode_frame`] instead of batching all frames at the end.
/// Suitable for piping audio to a playback device frame-by-frame.
fn generate_streaming(
    model: &TalkerModel,
    cp: &CodePredictor,
    decoder: &Decoder12Hz,
    input_ids: &[u32],
    prefill_fn: impl Fn(&mut Vec<AnyKVCache>) -> candle::Result<(Tensor, Tensor)>,
    args: &Args,
    device: &Device,
) -> anyhow::Result<Vec<f32>> {
    let eos = codec_tokens::CODEC_EOS;
    const MIN_NEW: usize = 2;
    const REPEAT_STOP: usize = 10;

    let (trailing, trailing_len, tts_pad) = build_trailing_text(model, input_ids)?;

    let mut kv_caches = model.new_kv_caches(args.max_frames + 256);
    let (hidden, logits) = prefill_fn(&mut kv_caches)?;
    let prefill_len = hidden.dim(1)?;
    let mut offset = prefill_len;
    let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

    let logits_2d = logits.squeeze(1)?;
    let mut rng = RngState::new(args.seed);
    let mut seen: Vec<u32> = Vec::new();
    let mut sem_token = sample_token(&logits_2d, args, &mut rng, &seen, 0, MIN_NEW, eos)?;
    seen.push(sem_token);

    let mut dec_state = decoder.new_streaming_state();
    let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
    let mut cp_caches = cp.new_kv_caches();
    let mut last_token = sem_token;
    let mut repeat_count = 0usize;
    let mut total_frames = 0usize;

    for frame in 0..args.max_frames {
        if sem_token == eos {
            eprintln!("EOS at frame {frame}");
            break;
        }
        if sem_token == last_token {
            repeat_count += 1;
            if repeat_count >= REPEAT_STOP && seen.len() >= MIN_NEW {
                eprintln!("Stopping: token {sem_token} repeated {repeat_count} times");
                break;
            }
        } else {
            repeat_count = 0;
        }
        last_token = sem_token;

        let sem_embed = model.get_codec_embedding(sem_token)?;
        let acoustic_codes = cp.generate_acoustic_codes(&last_hidden, &sem_embed, &mut cp_caches)?;

        // Build [1, 16, 1] frame tensor for streaming decode
        let mut row: Vec<u32> = vec![sem_token];
        let acoustics: Vec<u32> = acoustic_codes.to_vec1()?;
        row.extend(&acoustics);
        let frame_codes = Tensor::from_vec(row, (1usize, 16usize, 1usize), device)?;

        // Decode this single frame immediately
        let chunk = decoder.decode_frame(&frame_codes, &mut dec_state)?;
        let samples: Vec<f32> = chunk.flatten_all()?.to_vec1()?;
        audio_chunks.push(samples);
        total_frames += 1;

        if frame == args.max_frames - 1 { break; }

        // Build next talker step input
        let acoustic_sum = cp.get_acoustic_embeddings_sum_from_tensor(&acoustic_codes)?;
        let summed = sem_embed.add(&acoustic_sum)?;
        let text_add = if frame < trailing_len {
            trailing.i((.., frame..frame + 1, ..))?  
        } else {
            tts_pad.clone()
        };
        let step_in = summed.add(&text_add)?;

        let (h, new_logits) = model.generate_step_with_embed(&step_in, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        let l2d = new_logits.squeeze(1)?;
        let next = sample_token(&l2d, args, &mut rng, &seen, seen.len(), MIN_NEW, eos)?;
        seen.push(next);
        sem_token = next;
    }

    eprintln!("Generated {} codec frames (streaming)", total_frames);

    let audio: Vec<f32> = audio_chunks.into_iter().flatten().collect();
    Ok(audio)
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = select_device(args.cpu)?;
    let dtype = compute_dtype_for_device(&device);
    eprintln!("Device: {:?}, dtype: {:?}", device, dtype);

    // ── Resolve files ──────────────────────────────────────────────────────
    eprintln!("Resolving model files for '{}'…", args.model_id);
    let files = resolve_files(&args.model_id)?;

    // ── Config ─────────────────────────────────────────────────────────────
    let parsed_cfg = if let Some(cfg_path) = &files.config {
        let txt = std::fs::read_to_string(cfg_path)?;
        match ParsedModelConfig::from_json(&txt) {
            Ok(c) => {
                eprintln!("Model variant: {}", c.label());
                Some(c)
            }
            Err(e) => {
                eprintln!("Warning: could not parse config.json: {e}");
                None
            }
        }
    } else {
        None
    };

    let (talker_cfg, cp_cfg) = if let Some(ref c) = parsed_cfg {
        (TalkerConfig::from_parsed(c), CodePredictorConfig::from_parsed(c))
    } else {
        (TalkerConfig::default(), CodePredictorConfig::default())
    };

    // ── Load weights ───────────────────────────────────────────────────────
    eprintln!("Loading model weights…");
    let weights = load_flat_weights(&files.model_weights, &device, dtype)?;

    // ── Tokenizer ──────────────────────────────────────────────────────────
    eprintln!("Loading tokenizer…");
    let tokenizer = load_tokenizer(&files.model_dir)?;
    let input_ids = tokenize(&tokenizer, &args.text)?;
    eprintln!("Tokenized {} → {:?}", args.text, input_ids);

    // ── Build models ───────────────────────────────────────────────────────
    eprintln!("Building TalkerModel (hidden={})…", talker_cfg.hidden_size);
    let talker = TalkerModel::from_weights_dtype(&weights, talker_cfg, &device, dtype)?;

    eprintln!("Building CodePredictor (hidden={})…", cp_cfg.hidden_size);
    let cp_weights = filter_prefix(&weights, "talker.code_predictor.");
    let cp_vb = VarBuilder::from_tensors(cp_weights, dtype, &device);
    let cp = CodePredictor::new(cp_cfg, cp_vb)?;

    // ── Decoder (speech tokenizer) ─────────────────────────────────────────
    eprintln!("Loading speech tokenizer decoder…");
    let decoder_path = {
        let st_path = files.model_dir.join("speech_tokenizer/model.safetensors");
        if st_path.exists() {
            st_path
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(args.model_id.clone(), RepoType::Model));
            repo.get("speech_tokenizer/model.safetensors")?
        }
    };
    let decoder_weights = load_flat_weights(&decoder_path, &device, DType::F32)?;
    let decoder = Decoder12Hz::from_weights(&decoder_weights, Decoder12HzConfig::default())?;

    // ── Language / speaker ─────────────────────────────────────────────────
    let language: Language = args.language.parse()?;
    let speaker: Speaker = args.speaker.parse().unwrap_or(Speaker::Ryan);

    // ── Synthesis ─────────────────────────────────────────────────────────
    eprintln!("Synthesizing…");

    let audio = if let Some(ref instruct_text) = args.instruct {
        // VoiceDesign
        let instruct_fmt = format!("<|im_start|>user\n{}<|im_end|>\n", instruct_text);
        let instruct_ids = tokenize(&tokenizer, &instruct_fmt)?;
        let prefill = |kv: &mut Vec<AnyKVCache>| talker.prefill_voice_design(&input_ids, &instruct_ids, language, kv);
        if args.streaming {
            generate_streaming(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
        } else {
            generate(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
        }
    } else if let Some(ref ref_path) = args.ref_audio {
        // VoiceClone — two sub-modes:
        //   a) ICL: --ref-codes + --ref-text supplied → full in-context-learning
        //   b) Fallback: zero speaker embedding (low quality without encoder)
        let (ref_samples, ref_sr) = load_wav_f32(ref_path)?;
        eprintln!("Reference audio: {} samples @ {} Hz", ref_samples.len(), ref_sr);
        let _ = ref_samples;

        if let (Some(ref codes_path), Some(ref ref_text)) = (&args.ref_codes, &args.ref_text) {
            // ICL path: load pre-encoded codec codes
            eprintln!("ICL voice clone: loading ref codes from {}", codes_path.display());
            let raw = candle::safetensors::load(codes_path, &device)?;
            let ref_codes = raw
                .get("codes")
                .ok_or_else(|| anyhow::anyhow!("ref codes file must contain key \"codes\""))?
                .to_dtype(DType::U32)?;
            eprintln!("Ref codes shape: {:?}", ref_codes.shape());

            let ref_text_ids = tokenize(&tokenizer, ref_text)?;
            let ref_codec_embeds = sum_ref_codec_embeddings(&talker, &cp, &ref_codes)?;

            generate_icl(
                &talker, &cp, &decoder,
                &input_ids,
                &ref_text_ids,
                &ref_codec_embeds,
                language,
                &args, &device,
            )?
        } else {
            // Fallback: no encoder available — zero speaker embed
            eprintln!("Warning: no --ref-codes/--ref-text supplied; using zero speaker embed");
            eprintln!("For real voice cloning, encode the reference WAV with Mimi and pass --ref-codes.");
            let speaker_embed = Tensor::zeros((1, talker.config().hidden_size), dtype, &device)?;
            let prefill = |kv: &mut Vec<AnyKVCache>| talker.prefill_voice_clone(&input_ids, &speaker_embed, language, false, kv);
            if args.streaming {
                generate_streaming(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
            } else {
                generate(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
            }
        }
    } else {
        // CustomVoice
        let prefill = |kv: &mut Vec<AnyKVCache>| talker.prefill_custom_voice(&input_ids, speaker, language, kv);
        if args.streaming {
            generate_streaming(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
        } else {
            generate(&talker, &cp, &decoder, &input_ids, prefill, &args, &device)?
        }
    };

    eprintln!(
        "Generated {:.2}s of audio ({} samples @ 24 kHz)",
        audio.len() as f64 / 24_000.0,
        audio.len()
    );

    save_wav(&args.output, &audio, 24_000)?;
    eprintln!("Saved to {:?}", args.output);
    Ok(())
}
