#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use std::io::Write;

use candle::{DType, Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::vibevoice::{
    AsrConfig, DpmSolverScheduler, StreamingConfig, VibeVoiceASR, VibeVoiceStreaming,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, ValueEnum)]
enum Task {
    /// Automatic speech recognition (audio → text)
    Asr,
    /// Text-to-speech (text → audio)
    Tts,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "VibeVoice ASR/TTS inference")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The task to perform.
    #[arg(long, value_enum, default_value_t = Task::Asr)]
    task: Task,

    /// Input audio file for ASR mode.
    #[arg(long)]
    input: Option<String>,

    /// Text prompt for TTS mode.
    #[arg(long, default_value = "Hello, this is a test of VibeVoice text to speech.")]
    prompt: String,

    /// The model repository on HuggingFace Hub.
    #[arg(long)]
    model_id: Option<String>,

    /// Revision (branch) of the model repository.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Path to a local model directory (overrides --model-id).
    #[arg(long)]
    weight_path: Option<String>,

    /// Path to a local tokenizer.json file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Path to a voice prompt safetensors file for TTS (converted from .pt).
    #[arg(long)]
    voice_prompt: Option<String>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1.0 means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 512)]
    sample_len: usize,

    /// Number of diffusion inference steps for TTS.
    #[arg(long, default_value_t = 20)]
    diffusion_steps: usize,

    /// Classifier-free guidance scale for TTS diffusion.
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f64,

    /// Output WAV file path for TTS mode.
    #[arg(long, default_value = "output.wav")]
    output: String,
}

fn load_audio_pcm(path: &str) -> Result<(Vec<f32>, u32)> {
    let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(path)?;
    Ok((pcm_data, sample_rate))
}

/// Run ASR: encode speech, build prompt with speech placeholders, inject features, then decode.
fn run_asr(args: &Args, device: &Device) -> Result<()> {
    let input = args
        .input
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--input audio file is required for ASR mode"))?;

    let (config_filename, tokenizer_filename, filenames) = load_model_files(args, false, true)?;
    let config: AsrConfig = serde_json::from_str(&std::fs::read_to_string(&config_filename)?)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(E::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer.clone());

    // F16 on CPU: gemm uses F32 accumulation internally, NaN-critical ops
    // (attention softmax, MLP multiply, residual adds) are upcast in qwen2.rs
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F16
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };

    println!("Loading VibeVoice ASR model (dtype={:?})...", dtype);
    let start = std::time::Instant::now();
    let mut model = VibeVoiceASR::new(&config, vb)?;
    println!("Model loaded in {:?}", start.elapsed());

    // Special token IDs from processor_config.json
    let speech_start_id = tokenizer
        .token_to_id("<|object_ref_start|>")
        .unwrap_or(151649);
    let speech_pad_id = tokenizer
        .token_to_id("<|box_start|>")
        .unwrap_or(151651);
    let speech_end_id = tokenizer
        .token_to_id("<|object_ref_end|>")
        .unwrap_or(151650);
    let im_start_id: u32 = tokenizer
        .token_to_id("<|im_start|>")
        .unwrap_or(151644);
    let im_end_id: u32 = tokenizer
        .token_to_id("<|im_end|>")
        .unwrap_or(151645);
    let eos_token_id: u32 = 151643; // <|endoftext|>
    let newline_id = tokenizer
        .token_to_id("\n")
        .unwrap_or(198);

    // Load and preprocess audio
    let (pcm_data, sample_rate) = load_audio_pcm(input)?;
    let target_sr: u32 = 24000;
    // Simple linear resampling if needed
    let pcm_data = if sample_rate != target_sr {
        println!("Resampling audio from {}Hz to {}Hz", sample_rate, target_sr);
        let ratio = target_sr as f64 / sample_rate as f64;
        let new_len = (pcm_data.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(pcm_data.len() - 1);
            let frac = src_idx - idx0 as f64;
            resampled.push(pcm_data[idx0] * (1.0 - frac as f32) + pcm_data[idx1] * frac as f32);
        }
        resampled
    } else {
        pcm_data
    };

    // Normalize audio to -25 dBFS (matching VibeVoiceAcousticTokenizerFeatureExtractor)
    let pcm_data = {
        let rms = (pcm_data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
            / pcm_data.len() as f64)
            .sqrt();
        if rms > 1e-10 {
            let current_db = 20.0 * rms.log10();
            let target_db = -25.0_f64;
            let gain = 10.0_f64.powf((target_db - current_db) / 20.0);
            pcm_data.iter().map(|&x| x * gain as f32).collect::<Vec<_>>()
        } else {
            pcm_data
        }
    };

    let audio_duration = pcm_data.len() as f64 / target_sr as f64;
    let audio_tensor = Tensor::new(&pcm_data[..], device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    println!(
        "Audio input shape: {:?} ({:.1}s at {}Hz)",
        audio_tensor.dims(),
        audio_duration,
        target_sr
    );

    // Encode speech → continuous features (B, num_tokens, hidden_dim)
    let speech_features = model.encode_speech(&audio_tensor)?;
    let num_speech_tokens = speech_features.dim(1)?;
    println!(
        "Speech encoded: {} tokens, shape {:?}",
        num_speech_tokens,
        speech_features.dims()
    );

    // Build prompt following Python VibeVoiceASRProcessor format:
    // <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
    // <|im_start|>user\n<speech_start><speech_pad>...<speech_pad><speech_end>\n{user_suffix}<|im_end|>\n
    // <|im_start|>assistant\n
    let system_prompt = "You are a helpful assistant that transcribes audio input into text output in JSON format.";
    let user_suffix = format!(
        "This is a {:.2} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content",
        audio_duration
    );

    // Encode system prompt text
    let system_text_enc = tokenizer
        .encode(format!("system\n{}", system_prompt).as_str(), false)
        .map_err(E::msg)?;
    let user_suffix_enc = tokenizer
        .encode(format!("\n{}", user_suffix).as_str(), false)
        .map_err(E::msg)?;
    // Build full prompt token sequence
    let mut prompt_ids: Vec<u32> = Vec::new();

    // System message: <|im_start|>system\n{system_prompt}<|im_end|>\n
    prompt_ids.push(im_start_id);
    prompt_ids.extend(system_text_enc.get_ids());
    prompt_ids.push(im_end_id);
    prompt_ids.push(newline_id);

    // User message: <|im_start|>user\n<speech_start><speech_pad>*N<speech_end>\n{user_suffix}<|im_end|>\n
    prompt_ids.push(im_start_id);
    let user_prefix_enc = tokenizer.encode("user\n", false).map_err(E::msg)?;
    prompt_ids.extend(user_prefix_enc.get_ids());
    prompt_ids.push(speech_start_id);
    let speech_pad_start = prompt_ids.len();
    for _ in 0..num_speech_tokens {
        prompt_ids.push(speech_pad_id);
    }
    let speech_pad_end = prompt_ids.len();
    prompt_ids.push(speech_end_id);
    prompt_ids.extend(user_suffix_enc.get_ids());
    prompt_ids.push(im_end_id);
    prompt_ids.push(newline_id);

    // No assistant generation prompt — matching the chat_template.jinja

    println!(
        "Prompt: {} tokens ({} speech placeholders at positions {}..{})",
        prompt_ids.len(),
        num_speech_tokens,
        speech_pad_start,
        speech_pad_end
    );

    // First pass: embed all tokens, replace speech positions with encoded features
    let prompt_tensor = Tensor::new(&prompt_ids[..], device)?.unsqueeze(0)?;
    let mut inputs_embeds = model.embed_tokens(&prompt_tensor)?;

    // Replace speech_pad positions with speech features
    // speech_features shape: (1, num_speech_tokens, hidden_dim)
    // inputs_embeds shape: (1, seq_len, hidden_dim)
    let hidden_dim = inputs_embeds.dim(2)?;
    for i in 0..num_speech_tokens {
        let speech_feat = speech_features.i((0, i))?.reshape((1, 1, hidden_dim))?;
        let pos = speech_pad_start + i;
        inputs_embeds = inputs_embeds.slice_scatter(&speech_feat, 1, pos)?;
    }

    // Forward pass with injected speech features
    let logits = model.forward_embeds(&inputs_embeds, 0)?;
    let seq_len = prompt_ids.len();

    // Greedy autoregressive text decoding
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);

    let start_gen = std::time::Instant::now();
    let mut generated_tokens: Vec<u32> = vec![];

    // Sample first token from the last position of the prompt
    let last_logits = logits.i((0, seq_len - 1, ..))?.to_dtype(DType::F32)?;
    let next_token = logits_processor.sample(&last_logits)?;
    generated_tokens.push(next_token);

    if let Some(text) = tos.next_token(next_token)? {
        print!("{text}");
        std::io::stdout().flush()?;
    }

    // Continue autoregressive decoding
    let mut seqlen_offset = seq_len;
    for _step in 1..args.sample_len {
        let last = *generated_tokens.last().unwrap();
        if last == im_end_id || last == eos_token_id {
            break;
        }

        let input_t = Tensor::new(&[last], device)?.unsqueeze(0)?;
        let logits = model.forward(&input_t, seqlen_offset)?;
        let logits = logits.i((0, 0, ..))?.to_dtype(DType::F32)?;

        let next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        seqlen_offset += 1;


        if let Some(text) = tos.next_token(next_token)? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tos.decode_rest()? {
        print!("{rest}");
    }

    let dt = start_gen.elapsed();
    let generated = generated_tokens.len();
    println!(
        "\n{generated} tokens generated ({:.2} token/s)",
        generated as f64 / dt.as_secs_f64(),
    );

    model.clear_kv_cache();
    Ok(())
}

// Window sizes matching Python implementation
const TTS_TEXT_WINDOW_SIZE: usize = 5;
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// Run TTS using the VibeVoice-Realtime streaming model with voice prompt and CFG diffusion.
fn run_tts(args: &Args, device: &Device) -> Result<()> {
    let (config_filename, tokenizer_filename, filenames) = load_model_files(args, true, false)?;
    let config: StreamingConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_filename)?)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(E::msg)?;

    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };

    println!("Loading VibeVoice-Realtime TTS model...");
    let start = std::time::Instant::now();
    let mut model = VibeVoiceStreaming::new(&config, vb)?;
    println!("Model loaded in {:?}", start.elapsed());

    // Load voice prompt if provided
    let voice_prompt_path = args.voice_prompt.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--voice-prompt is required for TTS mode. \
             Provide a .safetensors voice prompt file (convert .pt with convert_voice_prompt.py)."
        )
    })?;

    println!("Loading voice prompt from {voice_prompt_path}...");
    let vp_tensors = candle::safetensors::load(voice_prompt_path, device)?;
    let (lm_seq_offset, tts_lm_seq_offset) = model.load_voice_prompt(&vp_tensors, dtype)?;
    let (_lm_hidden, tts_lm_hidden, _neg_lm_hidden, neg_tts_lm_hidden) =
        VibeVoiceStreaming::get_voice_prompt_hidden_states(&vp_tensors, dtype)?;
    println!(
        "Voice prompt loaded: LM offset={lm_seq_offset}, TTS-LM offset={tts_lm_seq_offset}"
    );

    // Also need negative KV caches. Load them into saved state and swap as needed.
    let neg_lm_layers = config.decoder_config.num_hidden_layers
        - config.tts_backbone_num_hidden_layers;
    let neg_tts_lm_layers = config.tts_backbone_num_hidden_layers;

    // Load negative KV caches
    let mut neg_lm_kv: Vec<(Tensor, Tensor)> = Vec::with_capacity(neg_lm_layers);
    for i in 0..neg_lm_layers {
        let k = vp_tensors
            .get(&format!("neg_lm.kv.{i}.key"))
            .ok_or_else(|| anyhow::anyhow!("missing neg_lm.kv.{i}.key"))?
            .to_dtype(dtype)?;
        let v = vp_tensors
            .get(&format!("neg_lm.kv.{i}.value"))
            .ok_or_else(|| anyhow::anyhow!("missing neg_lm.kv.{i}.value"))?
            .to_dtype(dtype)?;
        neg_lm_kv.push((k, v));
    }
    let neg_lm_seq_offset = if let Some((k, _)) = neg_lm_kv.first() {
        k.dim(2)?
    } else {
        0
    };

    let mut neg_tts_lm_kv: Vec<(Tensor, Tensor)> = Vec::with_capacity(neg_tts_lm_layers);
    for i in 0..neg_tts_lm_layers {
        let k = vp_tensors
            .get(&format!("neg_tts_lm.kv.{i}.key"))
            .ok_or_else(|| anyhow::anyhow!("missing neg_tts_lm.kv.{i}.key"))?
            .to_dtype(dtype)?;
        let v = vp_tensors
            .get(&format!("neg_tts_lm.kv.{i}.value"))
            .ok_or_else(|| anyhow::anyhow!("missing neg_tts_lm.kv.{i}.value"))?
            .to_dtype(dtype)?;
        neg_tts_lm_kv.push((k, v));
    }
    let neg_tts_lm_seq_offset = if let Some((k, _)) = neg_tts_lm_kv.first() {
        k.dim(2)?
    } else {
        0
    };

    // Tokenize the TTS text (Python appends "\n" to the stripped text)
    let tts_text = format!("{}\n", args.prompt.trim());
    let encoding = tokenizer.encode(tts_text.as_str(), false).map_err(E::msg)?;
    let tts_text_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_text_tokens = tts_text_ids.len();
    println!("TTS text tokens: {total_text_tokens}");

    let dhcfg = model.diffusion_config();
    let mut scheduler = DpmSolverScheduler::new(dhcfg.ddpm_num_steps, &dhcfg.prediction_type);
    scheduler.set_timesteps(args.diffusion_steps);

    // Track sequence offsets for positive and negative paths
    let mut pos_lm_offset = lm_seq_offset;
    let mut pos_tts_lm_offset = tts_lm_seq_offset;
    let mut neg_tts_offset = neg_tts_lm_seq_offset;
    let _neg_lm_offset = neg_lm_seq_offset;

    // The last hidden states from voice prompt prefill
    let mut tts_lm_last_hidden = tts_lm_hidden.i((.., tts_lm_hidden.dim(1)? - 1.., ..))?;
    let mut neg_tts_lm_last_hidden =
        neg_tts_lm_hidden.i((.., neg_tts_lm_hidden.dim(1)? - 1.., ..))?;

    let mut audio_chunks: Vec<Tensor> = Vec::new();
    let mut acoustic_cache = model.new_acoustic_streaming_cache();
    let mut is_first_frame = true;
    let mut tts_text_window_index = 0;
    let mut step = 0;
    let max_steps = args.sample_len;
    let mut eos_detected = false;

    let start_gen = std::time::Instant::now();

    loop {
        if step >= max_steps {
            println!("Reached maximum generation length {max_steps}");
            break;
        }

        // --- Text window prefill ---
        let window_start = tts_text_window_index * TTS_TEXT_WINDOW_SIZE;
        let window_end = (window_start + TTS_TEXT_WINDOW_SIZE).min(total_text_tokens);
        let cur_text_ids = if window_start < total_text_tokens {
            &tts_text_ids[window_start..window_end]
        } else {
            &[]
        };
        let next_window_start = (tts_text_window_index + 1) * TTS_TEXT_WINDOW_SIZE;
        let next_window_end =
            (next_window_start + TTS_TEXT_WINDOW_SIZE).min(total_text_tokens);
        let _next_text_window_size = if next_window_start < total_text_tokens {
            next_window_end - next_window_start
        } else {
            0
        };
        tts_text_window_index += 1;

        if !cur_text_ids.is_empty() {
            let text_len = cur_text_ids.len();
            let text_tensor =
                Tensor::new(cur_text_ids, device)?.unsqueeze(0)?;

            // Forward pass through lower LM
            let lm_output = model.forward_lm(&text_tensor, pos_lm_offset)?;
            pos_lm_offset += text_len;

            // Forward pass through upper TTS LM with text tokens
            // Build input: embed text tokens, replace with LM hidden states, add type embedding
            let text_embeds = model.embed_tokens(&text_tensor)?;
            // Replace embeddings with LM hidden states (the tail)
            let lm_hs_len = lm_output.dim(1)?;
            let embed_len = text_embeds.dim(1)?;
            let start_idx = embed_len - lm_hs_len;
            let inputs_embeds = if start_idx == 0 {
                lm_output.clone()
            } else {
                let prefix = text_embeds.i((.., ..start_idx, ..))?;
                Tensor::cat(&[&prefix, &lm_output], 1)?
            };

            // Add type embedding: text=1
            let type_ids = Tensor::ones((1, embed_len), DType::U32, device)?;
            let inputs_embeds = model.add_tts_type_embedding(&inputs_embeds, &type_ids)?;

            let tts_output = model.forward_tts_lm(&inputs_embeds, pos_tts_lm_offset)?;
            pos_tts_lm_offset += text_len;
            tts_lm_last_hidden = tts_output.i((.., tts_output.dim(1)? - 1.., ..))?;

            // Negative path: just feed acoustic_embed through (no text for negative)
            // The negative path doesn't get text windows, it stays at its prefilled state.

            step += text_len;
            println!(
                "  Prefilled {text_len} text tokens (window {}, step {step})",
                tts_text_window_index - 1
            );
        }

        // --- Speech generation: produce TTS_SPEECH_WINDOW_SIZE speech frames ---
        for _speech_idx in 0..TTS_SPEECH_WINDOW_SIZE {
            if step >= max_steps {
                break;
            }

            // Get conditioning from last hidden states
            let positive_condition =
                tts_lm_last_hidden.i((.., tts_lm_last_hidden.dim(1)? - 1, ..))?;
            let negative_condition =
                neg_tts_lm_last_hidden
                    .i((.., neg_tts_lm_last_hidden.dim(1)? - 1, ..))?;

            // Save positive KV caches, swap to negative for neg forward pass
            let pos_tts_kv = model.save_tts_kv_cache();

            // CFG diffusion sampling for one frame
            // Reset scheduler state for each frame (step_index, model_outputs)
            scheduler.set_timesteps(args.diffusion_steps);
            let speech_latent = model.sample_speech_tokens(
                &positive_condition,
                &negative_condition,
                &mut scheduler,
                args.cfg_scale,
                device,
                dtype,
            )?;

            // Decode speech latent to audio chunk (streaming: use cache for continuity)
            let speech_for_decode = speech_latent.unsqueeze(1)?; // (1, 1, vae_dim)
            let audio_chunk = model.decode_speech_streaming(
                &speech_for_decode,
                &mut acoustic_cache,
                is_first_frame,
            )?;
            audio_chunks.push(audio_chunk);
            is_first_frame = false;

            // Get acoustic embedding for feedback
            let speech_for_connector = speech_latent.unsqueeze(1)?; // (1, 1, vae_dim)
            let acoustic_embed = model.acoustic_connector(&speech_for_connector)?;

            step += 1;

            // --- Positive TTS LM step with acoustic feedback ---
            // Restore positive KV cache
            model.restore_tts_kv_cache(&pos_tts_kv);

            // Add type embedding: speech=0
            let type_ids = Tensor::zeros((1, 1), DType::U32, device)?;
            let speech_embeds =
                model.add_tts_type_embedding(&acoustic_embed, &type_ids)?;

            let tts_output = model.forward_tts_lm(&speech_embeds, pos_tts_lm_offset)?;
            pos_tts_lm_offset += 1;
            tts_lm_last_hidden = tts_output.clone();

            // Save positive KV caches again
            let pos_tts_kv_updated = model.save_tts_kv_cache();

            // --- Negative TTS LM step with acoustic feedback ---
            model.restore_tts_kv_cache_raw(&neg_tts_lm_kv);

            let neg_speech_embeds =
                model.add_tts_type_embedding(&acoustic_embed, &type_ids)?;
            let neg_tts_output =
                model.forward_tts_lm(&neg_speech_embeds, neg_tts_offset)?;
            neg_tts_offset += 1;
            neg_tts_lm_last_hidden = neg_tts_output.clone();

            // Save negative KV caches
            neg_tts_lm_kv = model.save_tts_kv_cache_as_pairs();

            // Restore positive for next iteration
            model.restore_tts_kv_cache(&pos_tts_kv_updated);

            // Check EOS – Python stops immediately when sigmoid > 0.5
            let eos_hidden = tts_lm_last_hidden
                .i((.., tts_lm_last_hidden.dim(1)? - 1, ..))?;
            let eos_logit = model.eos_logit(&eos_hidden)?;
            let eos_prob = 1.0 / (1.0 + (-eos_logit).exp());
            if eos_prob > 0.5 {
                println!("  EOS detected at step {step} (prob={eos_prob:.3})");
                eos_detected = true;
                break;
            }

            if (step % 10) == 0 {
                println!("  Speech step {step} (eos_prob={eos_prob:.3})");
            }
        }

        if eos_detected {
            println!("EOS: stopping generation.");
            break;
        }
    }

    let dt = start_gen.elapsed();
    println!(
        "Generation completed in {dt:.2?}: {step} steps, {} audio chunks",
        audio_chunks.len()
    );

    // Concatenate all audio chunks
    if audio_chunks.is_empty() {
        anyhow::bail!("No audio generated");
    }

    let all_audio = Tensor::cat(&audio_chunks, audio_chunks[0].dims().len() - 1)?;
    println!("Generated audio shape: {:?}", all_audio.dims());

    let audio_data: Vec<f32> = all_audio
        .squeeze(0)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    write_wav(&args.output, &audio_data, 24000)?;
    println!("Audio saved to {}", args.output);

    model.clear_kv_cache();
    Ok(())
}

/// Write a mono 16-bit WAV file.
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let s = s.clamp(-1.0, 1.0);
        let val = (s * 32767.0) as i16;
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

fn load_model_files(
    args: &Args,
    is_streaming: bool,
    is_asr: bool,
) -> Result<(std::path::PathBuf, std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let default_model_id = if is_asr {
        "microsoft/VibeVoice-ASR-HF".to_string()
    } else if is_streaming {
        "microsoft/VibeVoice-Realtime-0.5B".to_string()
    } else {
        "microsoft/VibeVoice-1.5B".to_string()
    };
    let tokenizer_model_id = if is_asr {
        default_model_id.clone()  // ASR model has its own tokenizer.json
    } else if is_streaming {
        "Qwen/Qwen2.5-0.5B".to_string()
    } else {
        "Qwen/Qwen2.5-1.5B".to_string()
    };

    if let Some(ref weight_path) = args.weight_path {
        let path = std::path::Path::new(weight_path);
        let config = path.join("config.json");
        let tokenizer = match &args.tokenizer_file {
            Some(f) => std::path::PathBuf::from(f),
            None => {
                let local = path.join("tokenizer.json");
                if local.exists() {
                    local
                } else {
                    let api = Api::new()?;
                    let tok_repo = api.repo(Repo::new(
                        tokenizer_model_id.clone(),
                        RepoType::Model,
                    ));
                    tok_repo.get("tokenizer.json")?
                }
            }
        };
        let filenames = if path.join("model.safetensors.index.json").exists() {
            candle_examples::hub_load_local_safetensors(
                weight_path,
                "model.safetensors.index.json",
            )?
        } else {
            vec![path.join("model.safetensors")]
        };
        Ok((config, tokenizer, filenames))
    } else {
        let model_id = args.model_id.clone().unwrap_or(default_model_id);
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            args.revision.clone(),
        ));
        let config = repo.get("config.json")?;
        let tokenizer = match &args.tokenizer_file {
            Some(f) => std::path::PathBuf::from(f),
            None => {
                let tok_api = Api::new()?;
                let tok_repo = tok_api.repo(Repo::new(
                    tokenizer_model_id.clone(),
                    RepoType::Model,
                ));
                tok_repo.get("tokenizer.json")?
            }
        };
        let filenames =
            match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                Ok(f) => f,
                Err(_) => vec![repo.get("model.safetensors")?],
            };
        Ok((config, tokenizer, filenames))
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
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let device = candle_examples::device(args.cpu)?;
    println!(
        "Running on {}, to run on {}, build this example with `--features {}`",
        if device.is_cuda() {
            "GPU"
        } else if device.is_metal() {
            "Metal"
        } else {
            "CPU"
        },
        if device.is_cuda() {
            "CPU"
        } else {
            "GPU"
        },
        if device.is_cuda() {
            "cpu"
        } else {
            "cuda"
        },
    );

    match args.task {
        Task::Asr => run_asr(&args, &device),
        Task::Tts => run_tts(&args, &device),
    }
}
