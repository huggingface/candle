//! High-level ACE-Step pipeline with text encoder inside.
//!
//! Accepts raw text prompts — handles tokenization, text encoding,
//! conditioning, DiT denoising, and VAE decoding internally.

use anyhow::Result;
use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::ace_step::{
    lm::{LmConfig, LmPipeline, TokenIds},
    model::{AceStepConditionGenerationModel, ConditionInputs, GenerateOptions},
    sampling,
    vae::{AutoencoderOobleck, OobleckDecoder},
    AceStepConfig, VaeConfig,
};
use candle_transformers::models::qwen3;

/// Metadata from LM CoT output, used to enrich the DiT text prompt.
///
/// When the LM generates chain-of-thought metadata (BPM, keyscale, etc.),
/// these values should be fed back into the DiT caption — matching the Python
/// `_dict_to_meta_string` / `_extract_caption_and_language` flow.
#[derive(Default)]
pub struct LmMetadata {
    pub bpm: Option<String>,
    pub timesignature: Option<String>,
    pub keyscale: Option<String>,
    /// LM-generated caption — replaces the user prompt in the DiT text input.
    pub caption: Option<String>,
    /// Language code for lyrics (default: "en").
    pub language: Option<String>,
}

/// Generation parameters.
pub struct GenerationParams {
    pub duration_secs: f64,
    pub seed: Option<u64>,
    pub num_steps: Option<usize>,
    pub guidance_scale: f64,
    pub shift: f64,
    pub is_turbo: bool,
    /// Precomputed LM hints at 25Hz `(1, T, acoustic_dim)`.
    pub lm_hints_25hz: Option<Tensor>,
    /// LM-generated metadata to enrich the DiT text prompt.
    pub lm_metadata: Option<LmMetadata>,
    /// Fraction of denoising steps using cover condition (0.0–1.0).
    pub audio_cover_strength: f64,
    /// Timestep range for CFG/APG guidance.
    pub cfg_interval: (f64, f64),
    /// ODE (deterministic) or SDE (stochastic) sampling.
    pub infer_method: sampling::InferMethod,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            duration_secs: 10.0,
            seed: None,
            num_steps: None,
            guidance_scale: 5.0,
            shift: 1.0,
            is_turbo: false,
            lm_hints_25hz: None,
            lm_metadata: None,
            audio_cover_strength: 1.0,
            cfg_interval: (0.0, 1.0),
            infer_method: sampling::InferMethod::Ode,
        }
    }
}

/// Stereo audio output.
pub struct AudioOutput {
    /// Stereo waveform `(2, T)` in `[-1, 1]`, 48 kHz.
    pub audio: Tensor,
    pub sample_rate: u32,
}

/// High-level pipeline: text prompt → stereo audio.
pub struct AceStepPipeline {
    model: AceStepConditionGenerationModel,
    text_encoder: qwen3::Model,
    tokenizer: tokenizers::Tokenizer,
    vae_decoder: OobleckDecoder,
    vae_encoder: Option<AutoencoderOobleck>,
    lm: Option<LmState>,
    silence_latent: Tensor,
    dit_config: AceStepConfig,
    vae_config: VaeConfig,
    device: Device,
    dtype: DType,
}

struct LmState {
    pipeline: LmPipeline,
    tokenizer: tokenizers::Tokenizer,
}

impl AceStepPipeline {
    /// Load all components.
    ///
    /// - `dit_vb` / `vae_vb` / `text_vb`: VarBuilders for each model
    /// - `text_config`: Qwen3 config (for the embedding model, NOT the LM)
    /// - `tokenizer_path`: `tokenizer.json` for the text encoder
    /// - `silence_latent`: `(1, T, acoustic_dim)` pre-loaded
    /// - `load_encoder`: load VAE encoder for cover mode
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dit_config: &AceStepConfig,
        vae_config: &VaeConfig,
        text_config: &qwen3::Config,
        dit_vb: VarBuilder,
        vae_vb: VarBuilder,
        text_vb: VarBuilder,
        tokenizer_path: &std::path::Path,
        silence_latent: Tensor,
        load_encoder: bool,
    ) -> Result<Self> {
        let device = dit_vb.device().clone();
        let dtype = dit_vb.dtype();

        let text_encoder = qwen3::Model::new(text_config, text_vb)?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let model = AceStepConditionGenerationModel::new(dit_config, dit_vb)?;
        let vae_decoder = OobleckDecoder::new(vae_config, vae_vb.pp("decoder"))?;
        let vae_encoder = if load_encoder {
            Some(AutoencoderOobleck::new(vae_config, vae_vb)?)
        } else {
            None
        };

        Ok(Self {
            model,
            text_encoder,
            tokenizer,
            vae_decoder,
            vae_encoder,
            lm: None,
            silence_latent,
            dit_config: dit_config.clone(),
            vae_config: vae_config.clone(),
            device,
            dtype,
        })
    }

    /// Number of latent frames for a given duration.
    pub fn latent_frames(&self, duration_secs: f64) -> usize {
        let hop = self.vae_config.hop_length();
        (duration_secs * self.vae_config.sampling_rate as f64 / hop as f64).ceil() as usize
    }

    /// Generate music from a text prompt and optional lyrics.
    pub fn text2music(
        &mut self,
        prompt: &str,
        lyrics: &str,
        params: &GenerationParams,
    ) -> Result<AudioOutput> {
        let (text_hs, text_mask, lyric_hs, lyric_mask) = self.encode_text(
            prompt,
            lyrics,
            params.duration_secs,
            params.lm_metadata.as_ref(),
        )?;

        let latent_frames = self.latent_frames(params.duration_secs);
        let timbre_frames = 750.min(self.silence_latent.dim(1)?);

        self.generate_inner(
            &text_hs,
            &text_mask,
            &lyric_hs,
            &lyric_mask,
            &self.silence_latent.narrow(1, 0, latent_frames)?,
            &self.silence_latent.narrow(1, 0, timbre_frames)?,
            params,
            latent_frames,
        )
    }

    /// Generate a cover conditioned on reference audio `(1, 2, T)` at 48kHz.
    ///
    /// - `reference_audio`: structural source (melody/rhythm)
    /// - `timbre_audio`: optional separate timbre source; if `None`, uses
    ///   `reference_audio` for both structure and timbre (Python separates these
    ///   as `src_audio` vs `reference_audio`)
    ///
    /// When `params.lm_hints_25hz` is set, uses the LM-derived hints with
    /// `is_covers=true` (tokenize/detokenize path in prepare_condition).
    /// Otherwise, uses raw reference latents directly as context (no lossy
    /// FSQ round-trip).
    pub fn cover(
        &mut self,
        prompt: &str,
        lyrics: &str,
        reference_audio: &Tensor,
        timbre_audio: Option<&Tensor>,
        params: &GenerationParams,
    ) -> Result<AudioOutput> {
        let (text_hs, text_mask, lyric_hs, lyric_mask) = self.encode_text(
            prompt,
            lyrics,
            params.duration_secs,
            params.lm_metadata.as_ref(),
        )?;

        let vae = self
            .vae_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("VAE encoder not loaded (pass load_encoder=true)"))?;
        let ref_latent = vae.encode_mean(reference_audio)?.transpose(1, 2)?;

        let latent_frames = self.latent_frames(params.duration_secs);
        let ref_len = ref_latent.dim(1)?;

        let src_latents = if ref_len >= latent_frames {
            ref_latent.narrow(1, 0, latent_frames)?
        } else {
            let pad = self.silence_latent.narrow(1, 0, latent_frames - ref_len)?;
            Tensor::cat(&[&ref_latent, &pad], 1)?
        };

        // Timbre: use separate timbre audio if provided, otherwise reference audio
        let refer_audio_packed = if let Some(timbre) = timbre_audio {
            let timbre_latent = vae.encode_mean(timbre)?.transpose(1, 2)?;
            let timbre_len = timbre_latent.dim(1)?;
            let timbre_frames = 750.min(timbre_len);
            timbre_latent.narrow(1, 0, timbre_frames)?
        } else {
            let timbre_frames = 750.min(ref_len);
            ref_latent.narrow(1, 0, timbre_frames)?
        };

        // Only enable is_covers (tokenize/detokenize path) when LM hints
        // are provided. Without LM hints, raw reference latents go directly
        // into context — the lossy FSQ round-trip on raw VAE latents produces
        // poor results.
        self.generate_inner(
            &text_hs,
            &text_mask,
            &lyric_hs,
            &lyric_mask,
            &src_latents,
            &refer_audio_packed,
            params,
            latent_frames,
        )
    }

    /// Selectively re-generate a region of existing audio while preserving the rest.
    ///
    /// - `source_audio`: original stereo waveform `(1, 2, T)` at 48kHz
    /// - `start_secs` / `end_secs`: time range to regenerate (seconds)
    ///
    /// The preserved regions keep the original audio latents; the repaint region
    /// is replaced with silence and the chunk_mask signals the DiT to generate new
    /// content there.
    pub fn repaint(
        &mut self,
        prompt: &str,
        lyrics: &str,
        source_audio: &Tensor,
        start_secs: f64,
        end_secs: f64,
        params: &GenerationParams,
    ) -> Result<AudioOutput> {
        let (text_hs, text_mask, lyric_hs, lyric_mask) = self.encode_text(
            prompt,
            lyrics,
            params.duration_secs,
            params.lm_metadata.as_ref(),
        )?;

        let vae = self
            .vae_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("VAE encoder not loaded (pass load_encoder=true)"))?;
        let src_latent = vae.encode_mean(source_audio)?.transpose(1, 2)?; // (1, T_lat, D)

        let latent_frames = self.latent_frames(params.duration_secs);
        let hop = self.vae_config.hop_length() as f64;
        let sr = self.vae_config.sampling_rate as f64;

        // Convert time range to latent frame indices
        let start_frame = ((start_secs * sr / hop).floor() as usize).min(latent_frames);
        let end_frame = ((end_secs * sr / hop).ceil() as usize).min(latent_frames);

        anyhow::ensure!(
            end_frame > start_frame,
            "repaint region is empty: {start_secs}s..{end_secs}s → frames {start_frame}..{end_frame}"
        );

        // Crop/pad source latents to target length
        let src_len = src_latent.dim(1)?;
        let mut src_latents = if src_len >= latent_frames {
            src_latent.narrow(1, 0, latent_frames)?
        } else {
            let pad = self.silence_latent.narrow(1, 0, latent_frames - src_len)?;
            Tensor::cat(&[&src_latent, &pad], 1)?
        };

        // Replace repaint region with silence in src_latents
        let silence_region = self
            .silence_latent
            .narrow(1, start_frame, end_frame - start_frame)?;
        // Build: [original[:start], silence[start:end], original[end:]]
        let parts: Vec<Tensor> = if start_frame > 0 && end_frame < latent_frames {
            vec![
                src_latents.narrow(1, 0, start_frame)?,
                silence_region,
                src_latents.narrow(1, end_frame, latent_frames - end_frame)?,
            ]
        } else if start_frame == 0 {
            vec![
                silence_region,
                src_latents.narrow(1, end_frame, latent_frames - end_frame)?,
            ]
        } else {
            vec![src_latents.narrow(1, 0, start_frame)?, silence_region]
        };
        src_latents = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        // Timbre from original audio (first 750 frames)
        let timbre_frames = 750.min(src_len);
        let refer_audio_packed = src_latent.narrow(1, 0, timbre_frames)?;

        // Build chunk_mask: 0 = preserve, 1 = regenerate
        let acoustic_dim = self.dit_config.audio_acoustic_hidden_dim;
        let chunk_mask_dim = self.dit_config.in_channels - 2 * acoustic_dim;
        let mut mask_data = vec![0.0f32; latent_frames * chunk_mask_dim];
        for t in start_frame..end_frame {
            for c in 0..chunk_mask_dim {
                mask_data[t * chunk_mask_dim + c] = 1.0;
            }
        }
        let chunk_masks =
            Tensor::from_vec(mask_data, (1, latent_frames, chunk_mask_dim), &self.device)?
                .to_dtype(self.dtype)?;

        self.generate_inner_with_mask(
            &text_hs,
            &text_mask,
            &lyric_hs,
            &lyric_mask,
            &src_latents,
            &refer_audio_packed,
            &chunk_masks,
            params,
            latent_frames,
        )
    }

    /// Unload the LM model to free GPU/RAM memory before DiT denoising.
    ///
    /// The LM (~1.2 GB for 0.6B, ~3.5 GB for 1.7B) is only needed for
    /// `generate_lm_hints()`. Dropping it before the DiT pass frees memory
    /// for the denoising loop.
    pub fn unload_lm(&mut self) {
        self.lm = None;
    }

    /// Load the 5Hz LM model for audio code generation.
    ///
    /// Call once before `generate_lm_hints()`. The LM weights and tokenizer
    /// are loaded from a HuggingFace repo (e.g. `ACE-Step/acestep-5Hz-lm-0.6B`).
    pub fn load_lm(
        &mut self,
        lm_config: &qwen3::Config,
        lm_vb: VarBuilder,
        lm_tokenizer_path: &std::path::Path,
    ) -> Result<()> {
        let lm_model = qwen3::ModelForCausalLM::new(lm_config, lm_vb)?;
        let lm_tokenizer = tokenizers::Tokenizer::from_file(lm_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("LM tokenizer: {e}"))?;
        let token_ids = TokenIds::discover(|s| {
            let enc = lm_tokenizer
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("encode {s:?}: {e}")))?;
            Ok(enc.get_ids().to_vec())
        })?;
        let pipeline = LmPipeline::new(lm_model, token_ids, self.device.clone());
        self.lm = Some(LmState {
            pipeline,
            tokenizer: lm_tokenizer,
        });
        Ok(())
    }

    /// Generate LM audio codes and convert to 25Hz latent hints.
    ///
    /// Returns `(lm_hints_tensor, metadata)`. The hints can be passed as
    /// `GenerationParams.lm_hints_25hz` to `text2music()` or `cover()`.
    pub fn generate_lm_hints(
        &mut self,
        prompt: &str,
        lyrics: &str,
        duration_secs: f64,
        lm_config: &LmConfig,
    ) -> Result<(Tensor, std::collections::BTreeMap<String, String>)> {
        let lm = self
            .lm
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("LM not loaded — call load_lm() first"))?;

        let prompt_text = LmPipeline::format_prompt(prompt, lyrics);
        let prompt_enc = lm
            .tokenizer
            .encode(prompt_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("LM encode: {e}"))?;

        let decode_fn = |t: u32| -> candle::Result<String> {
            lm.tokenizer
                .decode(&[t], false)
                .map_err(|e| candle::Error::Msg(format!("{e}")))
        };
        let encode_fn = |s: &str| -> candle::Result<Vec<u32>> {
            let e = lm
                .tokenizer
                .encode(s, false)
                .map_err(|e| candle::Error::Msg(format!("{e}")))?;
            Ok(e.get_ids().to_vec())
        };

        let output = lm.pipeline.generate(
            prompt_enc.get_ids(),
            duration_secs,
            lm_config,
            &decode_fn,
            &encode_fn,
        )?;

        anyhow::ensure!(
            !output.audio_codes.is_empty(),
            "LM generated no audio codes"
        );
        let hints = self.audio_codes_to_latents(&output.audio_codes)?;
        Ok((hints, output.metadata))
    }

    /// Convert LM-generated audio codes to 25Hz latent hints.
    pub fn audio_codes_to_latents(&self, codes: &[i64]) -> Result<Tensor> {
        let code_tensor = Tensor::new(codes, &self.device)?
            .unsqueeze(0)?
            .unsqueeze(2)?;
        let quantized = self.model.tokenizer.get_output_from_indices(&code_tensor)?;
        let quantized = quantized.to_dtype(self.dtype)?;
        Ok(self.model.detokenizer.forward(&quantized)?)
    }

    // ---- Internals ----

    fn encode_text(
        &mut self,
        prompt: &str,
        lyrics: &str,
        duration_secs: f64,
        metadata: Option<&LmMetadata>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Use LM-generated caption if available, otherwise user prompt.
        let caption = metadata
            .and_then(|m| m.caption.as_deref())
            .unwrap_or(prompt);

        // Build metas string from LM metadata or defaults.
        let bpm = metadata.and_then(|m| m.bpm.as_deref()).unwrap_or("N/A");
        let timesig = metadata
            .and_then(|m| m.timesignature.as_deref())
            .unwrap_or("N/A");
        let keyscale = metadata
            .and_then(|m| m.keyscale.as_deref())
            .unwrap_or("N/A");

        let caption_text = format!(
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n\
             # Caption\n{caption}\n\n# Metas\n\
             - bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {} seconds\n\
             <|endoftext|>",
            duration_secs as u32
        );
        let encoding = self
            .tokenizer
            .encode(caption_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;
        let prompt_tokens = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let text_hidden_states = self.text_encoder.forward(&prompt_tokens, 0)?;
        self.text_encoder.clear_kv_cache();

        let language = metadata.and_then(|m| m.language.as_deref()).unwrap_or("en");
        let lyric_text = if lyrics.is_empty() {
            format!("# Languages\n{language}\n\n# Lyric\n[Instrumental]<|endoftext|>")
        } else {
            format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>")
        };
        let lyric_encoding = self
            .tokenizer
            .encode(lyric_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("lyric tokenization: {e}"))?;
        let lyric_tokens = Tensor::new(lyric_encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let lyric_hidden_states = self.text_encoder.embed_tokens(&lyric_tokens)?;

        let text_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;
        let lyric_mask = Tensor::new(lyric_encoding.get_attention_mask(), &self.device)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;

        Ok((
            text_hidden_states,
            text_mask,
            lyric_hidden_states,
            lyric_mask,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_inner(
        &mut self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        src_latents: &Tensor,
        refer_audio_packed: &Tensor,
        params: &GenerationParams,
        latent_frames: usize,
    ) -> Result<AudioOutput> {
        let acoustic_dim = self.dit_config.audio_acoustic_hidden_dim;
        let chunk_mask_dim = self.dit_config.in_channels - 2 * acoustic_dim;
        let chunk_masks =
            Tensor::ones((1, latent_frames, chunk_mask_dim), self.dtype, &self.device)?;

        self.generate_inner_with_mask(
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            src_latents,
            refer_audio_packed,
            &chunk_masks,
            params,
            latent_frames,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_inner_with_mask(
        &mut self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        src_latents: &Tensor,
        refer_audio_packed: &Tensor,
        chunk_masks: &Tensor,
        params: &GenerationParams,
        latent_frames: usize,
    ) -> Result<AudioOutput> {
        let is_cover = params.lm_hints_25hz.is_some();

        let condition = self.model.prepare_condition(&ConditionInputs {
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            refer_audio_packed,
            refer_audio_order_mask: &Tensor::zeros((1,), DType::I64, &self.device)?,
            src_latents,
            chunk_masks,
            silence_latent: &self.silence_latent.narrow(1, 0, latent_frames)?,
            is_covers: &Tensor::from_vec(vec![if is_cover { 1u8 } else { 0u8 }], 1, &self.device)?,
            precomputed_lm_hints_25hz: params.lm_hints_25hz.as_ref(),
            audio_codes: None,
            non_cover_text_hidden_states: None,
            non_cover_text_attention_mask: None,
        })?;

        let num_steps = params
            .num_steps
            .unwrap_or(if params.is_turbo { 8 } else { 50 });

        let xt = self.model.generate_latents(
            &condition,
            None,
            &GenerateOptions {
                seed: params.seed,
                num_steps,
                guidance_scale: params.guidance_scale,
                shift: params.shift,
                is_turbo: params.is_turbo,
                audio_cover_strength: params.audio_cover_strength,
                cfg_interval: params.cfg_interval,
                infer_method: params.infer_method,
                ..Default::default()
            },
            &self.device,
            self.dtype,
        )?;

        let audio = self.vae_decoder.forward(&xt.transpose(1, 2)?)?;
        let audio = audio.squeeze(0)?;

        Ok(AudioOutput {
            audio,
            sample_rate: self.vae_config.sampling_rate as u32,
        })
    }
}

/// Load silence_latent from PyTorch `.pt` file → `(1, T, acoustic_dim)`.
pub fn load_silence_latent(
    path: &std::path::Path,
    acoustic_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut archive.by_name("silence_latent/data/0")?, &mut raw)?;
    let n_elements = raw.len() / 4;
    let t_len = n_elements / acoustic_dim;
    let cpu = Device::Cpu;
    Ok(
        Tensor::from_raw_buffer(&raw, DType::F32, &[1, acoustic_dim, t_len], &cpu)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(dtype)?
            .to_device(device)?,
    )
}
