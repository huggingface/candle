#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use clap::{ArgAction, Parser};
use tokenizers::Tokenizer;

use candle_transformers::models::qwen3_tts::{
    GenerationParams, Qwen3Tts, Qwen3TtsTokenizerV2, Qwen3TtsTokenizerV2Config,
    VoiceClonePromptItem,
};

#[derive(Parser)]
struct Args {
    /// Path to a local HF-style model directory.
    #[arg(long)]
    model_dir: PathBuf,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// TTS mode: custom, voice-design, or voice-clone.
    #[arg(long, default_value = "custom")]
    mode: String,

    /// Text to synthesize (repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    prompt: Vec<String>,

    /// Speaker name (CustomVoice models, repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    speaker: Vec<String>,

    /// Language (e.g., "English", "Chinese", or "Auto", repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    language: Vec<String>,

    /// Optional instruction text (repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    instruct: Vec<String>,

    /// Reference audio for voice-clone (repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    ref_audio: Vec<PathBuf>,

    /// Reference text for voice-clone ICL mode (repeat for batching).
    #[arg(long, action = ArgAction::Append)]
    ref_text: Vec<String>,

    /// Use x-vector-only mode (ignore ref text/code) for voice-clone.
    #[arg(long, default_value_t = false)]
    x_vector_only: bool,

    /// Output wav file (use {i} for batch index).
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    /// Temperature.
    #[arg(long, default_value_t = 0.9)]
    temperature: f64,

    /// Top-k sampling.
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p sampling.
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,

    /// Repetition penalty.
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,

    /// Max new tokens (codec steps).
    #[arg(long, default_value_t = 4096)]
    max_new_tokens: usize,

    /// Seed for sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Subtalker temperature.
    #[arg(long, default_value_t = 0.9)]
    subtalker_temperature: f64,

    /// Subtalker top-k.
    #[arg(long, default_value_t = 50)]
    subtalker_top_k: usize,

    /// Subtalker top-p.
    #[arg(long, default_value_t = 1.0)]
    subtalker_top_p: f64,
}

fn load_model_paths(model_dir: &Path) -> Result<(PathBuf, Vec<PathBuf>, PathBuf)> {
    let config = model_dir.join("config.json");
    let tokenizer = model_dir.join("tokenizer.json");
    let index = model_dir.join("model.safetensors.index.json");
    let weights = if index.exists() {
        candle_examples::hub_load_local_safetensors(model_dir, "model.safetensors.index.json")?
    } else {
        vec![model_dir.join("model.safetensors")]
    };
    Ok((config, weights, tokenizer))
}

fn load_speech_tokenizer_paths(model_dir: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    let tokenizer_dir = model_dir.join("speech_tokenizer");
    let config = tokenizer_dir.join("config.json");
    let index = tokenizer_dir.join("model.safetensors.index.json");
    let weights = if index.exists() {
        candle_examples::hub_load_local_safetensors(&tokenizer_dir, "model.safetensors.index.json")?
    } else {
        vec![tokenizer_dir.join("model.safetensors")]
    };
    Ok((config, weights))
}

fn broadcast<T: Clone>(mut vals: Vec<T>, batch: usize, default: T) -> Result<Vec<T>> {
    if vals.is_empty() {
        return Ok(vec![default; batch]);
    }
    if vals.len() == 1 && batch > 1 {
        return Ok(vec![vals.remove(0); batch]);
    }
    if vals.len() != batch {
        anyhow::bail!("value count {} does not match batch {}", vals.len(), batch);
    }
    Ok(vals)
}

fn broadcast_opt(mut vals: Vec<String>, batch: usize) -> Result<Vec<Option<String>>> {
    if vals.is_empty() {
        return Ok(vec![None; batch]);
    }
    if vals.len() == 1 && batch > 1 {
        let v = vals.remove(0);
        return Ok(vec![Some(v); batch]);
    }
    if vals.len() != batch {
        anyhow::bail!("value count {} does not match batch {}", vals.len(), batch);
    }
    Ok(vals.into_iter().map(Some).collect())
}

fn out_path_for_index(base: &str, idx: usize, batch: usize) -> String {
    if batch == 1 {
        return base.to_string();
    }
    if base.contains("{i}") {
        return base.replace("{i}", &idx.to_string());
    }
    if let Some((stem, ext)) = base.rsplit_once('.') {
        format!("{stem}_{idx}.{ext}")
    } else {
        format!("{base}_{idx}.wav")
    }
}

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str, device: &candle::Device) -> Result<Tensor> {
    let assistant_text = format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    let input_ids = tokenizer
        .encode(assistant_text, true)
        .map_err(anyhow::Error::msg)?;
    let input_ids: Vec<i64> = input_ids.get_ids().iter().map(|&v| v as i64).collect();
    Ok(Tensor::from_vec(input_ids, (1, input_ids.len()), device)?)
}

fn encode_instruct(
    tokenizer: &Tokenizer,
    instruct: &str,
    device: &candle::Device,
) -> Result<Tensor> {
    let instruct_text = format!("<|im_start|>user\n{instruct}<|im_end|>\n");
    let enc = tokenizer
        .encode(instruct_text, true)
        .map_err(anyhow::Error::msg)?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
    Ok(Tensor::from_vec(ids, (1, ids.len()), device)?)
}

fn encode_ref_text(
    tokenizer: &Tokenizer,
    ref_text: &str,
    device: &candle::Device,
) -> Result<Tensor> {
    let ref_text = format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n");
    let enc = tokenizer
        .encode(ref_text, true)
        .map_err(anyhow::Error::msg)?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
    Ok(Tensor::from_vec(ids, (1, ids.len()), device)?)
}

#[cfg(feature = "symphonia")]
fn load_audio_for_rate(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    let (pcm, sr) = candle_examples::audio::pcm_decode(path)?;
    if sr == target_sr {
        return Ok(pcm);
    }
    #[cfg(feature = "rubato")]
    {
        return candle_examples::audio::resample(&pcm, sr, target_sr).context("resample failed");
    }
    #[cfg(not(feature = "rubato"))]
    anyhow::bail!("rubato feature required for resampling");
}

#[cfg(not(feature = "symphonia"))]
fn load_audio_for_rate(_path: &Path, _target_sr: u32) -> Result<Vec<f32>> {
    anyhow::bail!("symphonia feature required for voice-clone audio decoding");
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    if args.prompt.is_empty() {
        anyhow::bail!("--prompt is required (repeat for batching)");
    }

    let (config_path, model_files, tokenizer_path) = load_model_paths(&args.model_dir)?;
    let (speech_config_path, speech_files) = load_speech_tokenizer_paths(&args.model_dir)?;

    let config_file = std::fs::File::open(config_path)?;
    let config: candle_transformers::models::qwen3_tts::Qwen3TtsConfig =
        serde_json::from_reader(config_file)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device)? };
    let mut model = Qwen3Tts::new(config, vb)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let speech_cfg_file = std::fs::File::open(speech_config_path)?;
    let speech_cfg: Qwen3TtsTokenizerV2Config = serde_json::from_reader(speech_cfg_file)?;
    let speech_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&speech_files, DType::F32, &device)? };
    let mut speech = Qwen3TtsTokenizerV2::new(speech_cfg.clone(), speech_vb)?;

    let batch = args.prompt.len();
    let languages = broadcast(args.language, batch, "Auto".to_string())?;
    let speakers = broadcast(args.speaker, batch, "".to_string())?;
    let instructs = broadcast_opt(args.instruct, batch)?;

    let mut input_ids = Vec::with_capacity(batch);
    for prompt in args.prompt.iter() {
        input_ids.push(encode_prompt(&tokenizer, prompt, &device)?);
    }

    let mut instruct_ids = Vec::with_capacity(batch);
    for ins in instructs.iter() {
        if let Some(text) = ins.as_ref() {
            if text.is_empty() {
                instruct_ids.push(None);
            } else {
                instruct_ids.push(Some(encode_instruct(&tokenizer, text, &device)?));
            }
        } else {
            instruct_ids.push(None);
        }
    }

    let params = GenerationParams {
        do_sample: true,
        top_k: Some(args.top_k),
        top_p: Some(args.top_p),
        temperature: args.temperature,
        repetition_penalty: args.repetition_penalty,
        subtalker_do_sample: true,
        subtalker_top_k: Some(args.subtalker_top_k),
        subtalker_top_p: Some(args.subtalker_top_p),
        subtalker_temperature: args.subtalker_temperature,
        max_new_tokens: args.max_new_tokens,
        seed: args.seed,
    };

    let mode = args.mode.to_lowercase();
    let mut ref_codes_for_decode: Option<Vec<Option<Tensor>>> = None;
    let codes_batch = if mode == "custom" {
        model.generate_custom_voice_codes_batch(
            &input_ids,
            &languages,
            &speakers,
            Some(&instruct_ids),
            true,
            &params,
        )?
    } else if mode == "voice-design" || mode == "voice_design" {
        model.generate_voice_design_codes_batch(
            &input_ids,
            &languages,
            &instruct_ids,
            true,
            &params,
        )?
    } else if mode == "voice-clone" || mode == "voice_clone" {
        let spk_cfg = model
            .config
            .speaker_encoder_config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("speaker encoder config missing for voice-clone"))?;
        let ref_audio = broadcast(args.ref_audio, batch, PathBuf::new())?;
        if ref_audio.iter().any(|p| p.as_os_str().is_empty()) {
            anyhow::bail!("--ref-audio is required for voice-clone");
        }
        let ref_texts = broadcast_opt(args.ref_text, batch)?;
        let mut ref_ids = Vec::with_capacity(batch);
        if args.x_vector_only {
            ref_ids.resize_with(batch, || None);
        } else {
            for rt in ref_texts.iter() {
                let rt = rt
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--ref-text required for ICL mode"))?;
                if rt.is_empty() {
                    ref_ids.push(None);
                } else {
                    ref_ids.push(Some(encode_ref_text(&tokenizer, rt, &device)?));
                }
            }
        }

        let mut prompts: Vec<VoiceClonePromptItem> = Vec::with_capacity(batch);
        for (idx, path) in ref_audio.iter().enumerate() {
            let pcm_for_spk =
                load_audio_for_rate(path, spk_cfg.sample_rate as u32).with_context(|| {
                    format!("load ref audio for speaker embedding: {}", path.display())
                })?;
            let spk_embed = model.extract_speaker_embedding(&pcm_for_spk, spk_cfg.sample_rate)?;

            let pcm_for_code = load_audio_for_rate(path, speech_cfg.input_sample_rate as u32)
                .with_context(|| {
                    format!("load ref audio for tokenizer encode: {}", path.display())
                })?;
            let audio_t = Tensor::from_vec(pcm_for_code, (1, pcm_for_code.len()), &device)?;
            let codes_list = speech.encode_audio(&audio_t, None)?;
            let ref_code = if args.x_vector_only {
                None
            } else {
                Some(codes_list[0].clone())
            };
            prompts.push(VoiceClonePromptItem {
                ref_code,
                ref_spk_embedding: spk_embed,
                x_vector_only_mode: args.x_vector_only,
                icl_mode: !args.x_vector_only,
            });
            if !args.x_vector_only && ref_ids[idx].is_none() {
                anyhow::bail!("ref_text required for ICL mode (batch index {idx})");
            }
        }
        ref_codes_for_decode = Some(prompts.iter().map(|p| p.ref_code.clone()).collect());

        model.generate_voice_clone_codes_batch(
            &input_ids,
            &languages,
            Some(&ref_ids),
            &prompts,
            true,
            &params,
        )?
    } else {
        anyhow::bail!("unsupported --mode {mode}");
    };

    let num_groups = model.config.talker_config.num_code_groups;
    for (idx, codes) in codes_batch.iter().enumerate() {
        let mut flat: Vec<i64> = Vec::with_capacity(codes.len() * num_groups);
        for row in codes.iter() {
            if row.len() != num_groups {
                anyhow::bail!("unexpected code group length {}", row.len());
            }
            flat.extend_from_slice(row);
        }
        let codes_t = Tensor::from_vec(flat, (codes.len(), num_groups), &device)?;
        let wav = if let Some(ref_codes) = ref_codes_for_decode.as_ref() {
            if let Some(ref_code) = ref_codes[idx].as_ref() {
                let ref_len = ref_code.dim(0)?;
                let total_len = ref_len + codes_t.dim(0)?;
                let cat = Tensor::cat(&[ref_code, &codes_t], 0)?;
                let wav = speech.decode_codes(&cat)?;
                let cut = ref_len * wav.len() / total_len.max(1);
                wav[cut..].to_vec()
            } else {
                speech.decode_codes(&codes_t)?
            }
        } else {
            speech.decode_codes(&codes_t)?
        };

        let out_path = out_path_for_index(&args.out_file, idx, codes_batch.len());
        let mut out = std::fs::File::create(&out_path)?;
        candle_examples::wav::write_pcm_as_wav(&mut out, &wav, speech.output_sample_rate())?;
    }

    Ok(())
}
