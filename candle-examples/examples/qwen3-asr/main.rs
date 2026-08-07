// Qwen3-ASR: Multilingual Automatic Speech Recognition
//
// Based on the Qwen3-ASR architecture from alan890104/qwen3-asr-rs (MIT).
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use tokenizers::Tokenizer;

use candle_transformers::models::qwen3_asr;
use candle_transformers::models::qwen3_asr::kv_cache::KVCache;
use candle_transformers::models::qwen3_asr::model::get_rope_index;

mod tokenizer;

const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const NUM_MEL_BINS: usize = 128;

const IM_END: u32 = 151_645;
const ENDOFTEXT: u32 = 151_643;
const ASR_TEXT_SEP: u32 = 151_704;
const IM_START: u32 = 151_644;
const NEWLINE: u32 = 198;

fn compute_power_stft(
    signal: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> (Vec<f32>, usize, usize) {
    let n_freqs = n_fft / 2 + 1;
    let n_frames = (signal.len().saturating_sub(n_fft)) / hop_length + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut power = vec![0.0f32; n_freqs * n_frames];
    let mut fft_buf = vec![Complex::new(0.0, 0.0); n_fft];
    for i in 0..n_frames {
        let offset = i * hop_length;
        let available = n_fft.min(signal.len().saturating_sub(offset));
        for j in 0..available {
            fft_buf[j] = Complex::new(signal[offset + j] * window[j], 0.0);
        }
        for item in fft_buf.iter_mut().take(n_fft).skip(available) {
            *item = Complex::new(0.0, 0.0);
        }
        fft.process(&mut fft_buf);
        let base = i * n_freqs;
        for j in 0..n_freqs {
            power[base + j] = fft_buf[j].norm_sqr();
        }
    }
    (power, n_freqs, n_frames)
}

// ─── Mel filterbank (Slaney scale) ───────────────────────────────

fn hz_to_mel(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn create_mel_filters(num_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let f_max = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..(num_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let mut filters = vec![0.0f32; num_mels * n_freqs];
    for i in 0..num_mels {
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
        for j in 0..n_freqs {
            let freq = j as f64 * sample_rate as f64 / n_fft as f64;
            let lower = hz_points[i];
            let center = hz_points[i + 1];
            let upper = hz_points[i + 2];
            let v = if freq >= lower && freq <= center {
                (freq - lower) / (center - lower)
            } else if freq > center && freq <= upper {
                (upper - freq) / (upper - center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = (v * enorm) as f32;
        }
    }
    filters
}

// ─── Mel spectrogram ─────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|i| 0.5 - 0.5 * (two_pi * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * n - 2;
    let r = i % period;
    if r < n {
        r
    } else {
        period - r
    }
}

fn reflection_pad(samples: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return vec![0.0; pad_left + pad_right];
    }
    let mut padded = Vec::with_capacity(n + pad_left + pad_right);
    for i in (1..=pad_left).rev() {
        padded.push(samples[reflect_index(i, n)]);
    }
    padded.extend_from_slice(samples);
    for i in 0..pad_right {
        padded.push(samples[n - 1 - reflect_index(i + 1, n)]);
    }
    padded
}

fn compute_mel_spectrogram(samples: &[f32]) -> (Vec<f32>, usize) {
    let mel_filters = create_mel_filters(NUM_MEL_BINS, N_FFT, SAMPLE_RATE);
    let n_freqs = N_FFT / 2 + 1;
    let hann = hann_window(N_FFT);

    let pad = N_FFT / 2;
    let padded = reflection_pad(samples, pad, pad);
    let padded_len = padded.len();
    let n_frames_total = (padded_len.saturating_sub(N_FFT)) / HOP_LENGTH + 1;
    let n_frames = n_frames_total.saturating_sub(1);

    let (power, _, _) = compute_power_stft(&padded, N_FFT, HOP_LENGTH, &hann);

    // Store as [mel_bins, frames] (row-major) to match tensor shape (1, NUM_MEL_BINS, n_frames)
    let mut mel = vec![0.0f32; NUM_MEL_BINS * n_frames];
    for m in 0..NUM_MEL_BINS {
        let filter_base = m * n_freqs;
        for i in 0..n_frames {
            let power_base = i * n_freqs;
            let mut sum = 0.0f32;
            for k in 0..n_freqs {
                sum += power[power_base + k] * mel_filters[filter_base + k];
            }
            mel[m * n_frames + i] = sum.max(1e-10).log10();
        }
    }

    // Qwen3-ASR normalization: max(log10, max_val - 8) / 4 + 1
    let max_log = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = max_log - 8.0;
    for v in mel.iter_mut() {
        *v = v.max(min_val) / 4.0 + 1.0;
    }

    (mel, n_frames)
}

// ─── Resampling (simple linear, no rubato needed) ────────────────

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
        out.push(samples[idx0] + (samples[idx1] - samples[idx0]) * t);
    }
    out
}

// ─── Prompt building ─────────────────────────────────────────────

fn encode(tokenizer: &Tokenizer, s: &str) -> anyhow::Result<Vec<u32>> {
    tokenizer
        .encode(s, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))
}

fn capital_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn normalize_language(lang: &str) -> String {
    let lower = lang.to_lowercase();
    match lower.as_str() {
        "en" => "English".into(),
        "es" => "Spanish".into(),
        "zh" => "Chinese".into(),
        "fr" => "French".into(),
        "de" => "German".into(),
        "ja" => "Japanese".into(),
        "ko" => "Korean".into(),
        "ru" => "Russian".into(),
        "pt" => "Portuguese".into(),
        "it" => "Italian".into(),
        "ar" => "Arabic".into(),
        "hi" => "Hindi".into(),
        "nl" => "Dutch".into(),
        "tr" => "Turkish".into(),
        "vi" => "Vietnamese".into(),
        "th" => "Thai".into(),
        "id" => "Indonesian".into(),
        other => capital_first(other),
    }
}

fn build_prompt(
    tokenizer: &Tokenizer,
    audio_token_id: u32,
    audio_start_id: u32,
    audio_end_id: u32,
    num_audio_tokens: usize,
    language: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    let mut tokens: Vec<u32> = Vec::new();

    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    tokens.push(IM_START);
    tokens.extend_from_slice(&encode(tokenizer, "system")?);
    tokens.push(NEWLINE);
    tokens.push(IM_END);
    tokens.push(NEWLINE);
    tokens.push(IM_START);
    tokens.extend_from_slice(&encode(tokenizer, "user")?);
    tokens.push(NEWLINE);
    tokens.push(audio_start_id);

    // audio placeholders
    tokens.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
    tokens.push(audio_end_id);

    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push(IM_END);
    tokens.push(NEWLINE);
    tokens.push(IM_START);
    tokens.extend_from_slice(&encode(tokenizer, "assistant")?);
    tokens.push(NEWLINE);

    if let Some(lang) = language {
        tokens.extend_from_slice(&encode(
            tokenizer,
            &format!("language {}", normalize_language(lang)),
        )?);
    }

    Ok(tokens)
}

// ─── Position IDs for decode steps ── (from PR #3376) ────────────

fn position_ids_for_step(prompt_len: i64, step: usize, device: &Device) -> anyhow::Result<Tensor> {
    let pos = prompt_len + step as i64;
    Ok(Tensor::from_vec(
        vec![pos; 3],
        (3usize, 1usize, 1usize),
        device,
    )?)
}

// ─── Decode text ─────────────────────────────────────────────────

fn decode_text_tokens(tokenizer: &Tokenizer, generated: &[u32]) -> anyhow::Result<String> {
    let sep_pos = generated.iter().position(|&id| id == ASR_TEXT_SEP);
    let text_ids = match sep_pos {
        Some(pos) => &generated[pos + 1..],
        None => generated,
    };
    let filtered: Vec<u32> = text_ids
        .iter()
        .copied()
        .filter(|&id| id != IM_END && id != ENDOFTEXT)
        .collect();
    if filtered.is_empty() {
        return Ok(String::new());
    }
    tokenizer
        .decode(&filtered, true)
        .map_err(|e| anyhow::anyhow!(e))
}

// ─── CLI ─────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    input: Option<String>,

    #[arg(long)]
    language: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long)]
    flash_attn: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    // ── Download / load ────────────────────────────────────────
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    println!("Loading config...");
    let config_path = repo.get("config.json")?;
    let cfg: qwen3_asr::Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    let audio_token_id = cfg.thinker_config.audio_token_id.ok_or_else(|| {
        anyhow::anyhow!("thinker_config.audio_token_id is required in config.json")
    })?;
    let audio_start_id = cfg.thinker_config.audio_start_token_id.unwrap_or(151_669);
    let audio_end_id = cfg.thinker_config.audio_end_token_id.unwrap_or(151_670);

    println!("Loading tokenizer...");
    // Try tokenizer.json first; fall back to vocab.json + merges.txt
    let tokenizer = if let Ok(path) = repo.get("tokenizer.json") {
        Tokenizer::from_file(&path).map_err(anyhow::Error::msg)?
    } else {
        let vocab_path = repo.get("vocab.json")?;
        let merges_path = repo.get("merges.txt")?;
        let config_path = repo.get("tokenizer_config.json").ok();
        tokenizer::build_qwen_bpe_tokenizer(&vocab_path, &merges_path, config_path.as_deref())?
    };

    println!("Loading safetensors...");
    let model_paths = if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("model.safetensors.index.json missing weight_map"))?;
        let mut files: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for file in weight_map.values() {
            if let Some(f) = file.as_str() {
                files.insert(f.to_string());
            }
        }
        let mut paths = Vec::new();
        for f in &files {
            paths.push(repo.get(f)?);
        }
        paths
    } else {
        vec![repo.get("model.safetensors")?]
    };
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&model_paths, qwen3_asr::DTYPE, &device)? };

    println!("Loading model...");
    let model = qwen3_asr::model::Model::load(&cfg, vb, &device, args.flash_attn)?;

    // ── Load / resample audio ──────────────────────────────────
    let input = match &args.input {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let dataset = api.dataset("Narsil/candle-examples".to_string());
            println!("No input specified, downloading sample...");
            dataset.get("samples_jfk.wav")?
        }
    };
    let (pcm_data, sr) = candle_examples::audio::pcm_decode(&input)?;
    println!("decoded: {} samples at {} Hz", pcm_data.len(), sr);

    let samples = if sr != SAMPLE_RATE {
        println!("resampling {} Hz -> {} Hz", sr, SAMPLE_RATE);
        linear_resample(&pcm_data, sr, SAMPLE_RATE)
    } else {
        pcm_data
    };

    // ── Mel spectrogram + audio encoder ─────────────────────────
    println!("computing mel spectrogram...");
    let (mel, n_frames) = compute_mel_spectrogram(&samples);
    let mel_tensor =
        Tensor::from_vec(mel, (1, NUM_MEL_BINS, n_frames), &device)?.to_dtype(qwen3_asr::DTYPE)?;

    println!("encoding audio...");
    let audio_features = model.get_audio_features(&mel_tensor)?;
    let num_audio_tokens = audio_features.dims2()?.0;
    println!("audio tokens: {num_audio_tokens}");

    // ── Build prompt ───────────────────────────────────────────
    let prompt = build_prompt(
        &tokenizer,
        audio_token_id,
        audio_start_id,
        audio_end_id,
        num_audio_tokens,
        args.language.as_deref(),
    )?;
    let prompt_len = prompt.len();
    println!("prompt tokens: {prompt_len}");

    // ── Prefill with KV cache ─── (pattern from PR #3376) ──────
    let audio_placeholder_count = prompt.iter().filter(|&&id| id == audio_token_id).count();
    let input_ids = Tensor::from_vec(prompt, (1, prompt_len), &device)?;
    let attention_mask = Tensor::ones((1usize, prompt_len), DType::U32, &device)?;

    let inputs_embeds = model.inputs_embeds_with_audio_features(
        &input_ids,
        Some(&audio_features),
        audio_placeholder_count,
    )?;
    let position_ids = get_rope_index(&attention_mask)?;

    let mut kv_cache = KVCache::new();
    let logits = model.forward_with_kv_cache(
        &attention_mask,
        &position_ids,
        &inputs_embeds,
        &mut kv_cache,
    )?;

    let mut next_logits = logits.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;

    let prompt_len_i64 = prompt_len as i64;

    // ── Autoregressive decode ─── (pattern from PR #3376) ──────
    let ones_col = Tensor::ones((1usize, 1usize), DType::U32, &device)?;
    let mut attention_mask_total = attention_mask;
    let mut generated: Vec<u32> = Vec::new();

    for step in 0..args.max_tokens {
        let next_token = next_logits.argmax(1)?.to_vec1::<u32>()?[0];

        if next_token == IM_END || next_token == ENDOFTEXT {
            break;
        }
        generated.push(next_token);

        // Extend attention mask: [1, total] -> [1, total + 1]
        attention_mask_total = Tensor::cat(&[&attention_mask_total, &ones_col], 1)?;

        let next_id = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
        let next_emb = model.embed_tokens(&next_id)?;
        let pos_ids = position_ids_for_step(prompt_len_i64, step, &device)?;

        let step_logits = model.forward_with_kv_cache(
            &attention_mask_total,
            &pos_ids,
            &next_emb,
            &mut kv_cache,
        )?;
        // step_logits: [1, 1, vocab_size] -- only the new token
        next_logits = step_logits.squeeze(1)?;
    }

    let text = decode_text_tokens(&tokenizer, &generated)?;
    println!("\n{text}");

    Ok(())
}
