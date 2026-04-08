//! Top-level ACE-Step conditional generation model.
//!
//! Mirrors the subset of Python's `AceStepConditionGenerationModel` needed for
//! **single-sample DiT-only Text2Music** with base/sft models. Owns the
//! condition encoder, DiT decoder, audio tokenizer/detokenizer, and
//! null-condition embedding. Provides `prepare_condition` and
//! `generate_latents` as the primary API, operating in latent space.
//!
//! ## Current scope vs Python
//!
//! Implemented:
//! - Text2Music with base/sft (continuous schedule + APG guidance)
//! - Text2Music with turbo (discrete schedule, no CFG)
//! - Real tokenizer masks, encoder_attention_mask propagation
//! - Cover mode (`is_covers`, tokenize/detokenize round-trip in prepare_condition)
//! - `precomputed_lm_hints_25Hz` / `audio_codes` (LM-generated audio codes)
//! - Audio cover strength blending (switch condition mid-loop with KV cache reset)
//! - Custom timestep schedules for turbo models (mapped to VALID_TIMESTEPS)
//!
//! Not yet implemented (Python `generate_audio` features):
//! - Batched inference (batch > 1)
//! - LoRA support

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::condition::ConditionEncoder;
use super::dit::AceStepDiTModel;
use super::sampling;
use super::tokenizer::{AudioTokenDetokenizer, AudioTokenizer};
use super::AceStepConfig;

/// Inputs for the condition encoder.
pub struct ConditionInputs<'a> {
    /// Text encoder hidden states `(B, T_text, text_hidden_dim)`.
    pub text_hidden_states: &'a Tensor,
    /// Text attention mask `(B, T_text)`.
    pub text_attention_mask: &'a Tensor,
    /// Lyric token embeddings `(B, T_lyric, text_hidden_dim)`.
    pub lyric_hidden_states: &'a Tensor,
    /// Lyric attention mask `(B, T_lyric)`.
    pub lyric_attention_mask: &'a Tensor,
    /// Packed reference audio features `(N, T_audio, timbre_hidden_dim)`.
    pub refer_audio_packed: &'a Tensor,
    /// 1D order mask mapping segments to batch indices `(N,)`.
    pub refer_audio_order_mask: &'a Tensor,
    /// Source latents `(B, T, acoustic_dim)` — silence for text2music.
    pub src_latents: &'a Tensor,
    /// Chunk masks `(B, T, acoustic_dim)` — ones for full generation.
    pub chunk_masks: &'a Tensor,
    /// Silence latent for tokenizer padding `(1, T_max, acoustic_dim)`.
    pub silence_latent: &'a Tensor,
    /// Boolean tensor `(B,)` — true for cover mode items.
    pub is_covers: &'a Tensor,
    /// Precomputed LM hints at 25Hz `(B, T, acoustic_dim)`. Skips tokenize/detokenize.
    pub precomputed_lm_hints_25hz: Option<&'a Tensor>,
    /// Audio codes from LM `(B, N, 1)`. Skips tokenize, uses quantizer → detokenize.
    pub audio_codes: Option<&'a Tensor>,
    /// Text hidden states for the non-cover condition (used when `audio_cover_strength < 1.0`).
    pub non_cover_text_hidden_states: Option<&'a Tensor>,
    /// Attention mask for non-cover text.
    pub non_cover_text_attention_mask: Option<&'a Tensor>,
}

/// Prepared conditioning for the DiT decoder.
pub struct PreparedCondition {
    pub encoder_hidden_states: Tensor,
    pub encoder_attention_mask: Tensor,
    pub context_latents: Tensor,
}

/// Options for latent generation.
pub struct GenerateOptions {
    pub seed: Option<u64>,
    pub num_steps: usize,
    pub guidance_scale: f64,
    pub shift: f64,
    pub is_turbo: bool,
    /// ODE (deterministic) or SDE (stochastic) sampling.
    pub infer_method: sampling::InferMethod,
    /// Timestep range in which CFG/APG guidance is applied.
    /// Outside this range, only the conditional prediction is used.
    /// Default: `(0.0, 1.0)` — apply guidance at all timesteps.
    pub cfg_interval: (f64, f64),
    /// Custom timestep schedule for turbo models (1–20 values).
    /// Each value is mapped to the nearest valid timestep. When `None`,
    /// the shift-based schedule is used.
    pub timesteps: Option<Vec<f64>>,
    /// Fraction of denoising steps that use the cover condition (0.0–1.0).
    /// At `cover_steps = round(num_steps * audio_cover_strength)` the loop
    /// switches to the non-cover condition and clears the KV cache.
    /// Default: `1.0` (no blending — cover condition used for all steps).
    pub audio_cover_strength: f64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            seed: None,
            num_steps: 50,
            guidance_scale: 5.0,
            shift: 1.0,
            is_turbo: false,
            infer_method: sampling::InferMethod::Ode,
            cfg_interval: (0.0, 1.0),
            timesteps: None,
            audio_cover_strength: 1.0,
        }
    }
}

/// Top-level ACE-Step model that owns all sub-modules and provides
/// `prepare_condition` + `generate_latents` as the main API.
pub struct AceStepConditionGenerationModel {
    config: AceStepConfig,
    encoder: ConditionEncoder,
    decoder: AceStepDiTModel,
    pub tokenizer: AudioTokenizer,
    pub detokenizer: AudioTokenDetokenizer,
    null_condition_emb: Tensor,
}

impl AceStepConditionGenerationModel {
    /// Load all sub-modules from a single safetensors checkpoint.
    pub fn new(config: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = ConditionEncoder::new(config, vb.pp("encoder"))?;
        let decoder = AceStepDiTModel::new(config, vb.pp("decoder"))?;
        let tokenizer = AudioTokenizer::new(config, vb.pp("tokenizer"))?;
        let detokenizer = AudioTokenDetokenizer::new(config, vb.pp("detokenizer"))?;
        // XL models use encoder_hidden_size for null_condition_emb (it operates
        // in encoder space, not decoder space).
        let null_condition_emb =
            vb.get((1, 1, config.encoder_hidden_size()), "null_condition_emb")?;

        Ok(Self {
            config: config.clone(),
            encoder,
            decoder,
            tokenizer,
            detokenizer,
            null_condition_emb,
        })
    }

    /// Tokenize audio latents: pad to `pool_window_size` multiple, reshape, quantize.
    ///
    /// Returns `(quantized, indices, downsampled_mask)`.
    fn tokenize(
        &self,
        hidden_states: &Tensor,
        silence_latent: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, d) = hidden_states.dims3()?;
        let pws = self.tokenizer.pool_window_size();

        // Pad to multiple of pool_window_size using silence_latent
        let (hidden_states, attention_mask) = if t % pws != 0 {
            let pad_len = pws - (t % pws);
            let pad = silence_latent
                .narrow(1, 0, pad_len)?
                .broadcast_as((b, pad_len, d))?
                .contiguous()?;
            let hs = Tensor::cat(&[hidden_states, &pad], 1)?;
            let mask_pad = Tensor::zeros(
                (b, pad_len),
                attention_mask.dtype(),
                attention_mask.device(),
            )?;
            let am = Tensor::cat(&[attention_mask, &mask_pad], 1)?;
            (hs, am)
        } else {
            (hidden_states.clone(), attention_mask.clone())
        };

        let t_padded = hidden_states.dim(1)?;
        let num_patches = t_padded / pws;

        // Reshape: (B, T, D) -> (B, T/pws, pws, D)
        let reshaped = hidden_states.reshape((b, num_patches, pws, d))?;

        // Downsample attention_mask: reshape (B, T) -> (B, T/pws, pws) then max
        let mask_reshaped = attention_mask.reshape((b, num_patches, pws))?;
        let downsampled_mask = mask_reshaped.max(2)?;

        let (quantized, indices) = self.tokenizer.forward(&reshaped)?;
        Ok((quantized, indices, downsampled_mask))
    }

    /// Detokenize quantized representations back to acoustic space.
    fn detokenize(&self, quantized: &Tensor) -> Result<Tensor> {
        self.detokenizer.forward(quantized)
    }

    /// Prepare conditioning: encode text/lyrics/timbre into packed encoder
    /// hidden states, and build context latents from src_latents + chunk_masks.
    ///
    /// When cover mode is active (`is_covers > 0`), source latents are replaced
    /// with tokenized-then-detokenized LM hints.
    pub fn prepare_condition(&self, inputs: &ConditionInputs) -> Result<PreparedCondition> {
        let (encoder_hidden_states, encoder_attention_mask) = self.encoder.forward(
            inputs.text_hidden_states,
            inputs.text_attention_mask,
            inputs.lyric_hidden_states,
            inputs.lyric_attention_mask,
            inputs.refer_audio_packed,
            inputs.refer_audio_order_mask,
        )?;

        // Cover mode: compute LM hints and replace src_latents where is_covers > 0.
        // Skip entirely when is_covers is all-false (text2music) — no tokenize/detokenize needed.
        let latent_len = inputs.src_latents.dim(1)?;

        let any_covers = inputs
            .is_covers
            .gt(0u8)?
            .to_dtype(candle::DType::U32)?
            .sum_all()?
            .to_scalar::<u32>()?
            > 0;

        let src_latents = if any_covers {
            let lm_hints_25hz = if let Some(precomputed) = inputs.precomputed_lm_hints_25hz {
                precomputed.narrow(1, 0, latent_len)?
            } else if let Some(audio_codes) = inputs.audio_codes {
                let quantized = self.tokenizer.get_output_from_indices(audio_codes)?;
                let detokenized = self.detokenize(&quantized)?;
                let det_len = detokenized.dim(1)?;
                if det_len >= latent_len {
                    detokenized.narrow(1, 0, latent_len)?
                } else {
                    let pad = inputs.silence_latent.narrow(1, 0, latent_len - det_len)?;
                    Tensor::cat(&[&detokenized, &pad], 1)?
                }
            } else {
                let b = inputs.src_latents.dim(0)?;
                let attention_mask = Tensor::ones(
                    (b, latent_len),
                    inputs.src_latents.dtype(),
                    inputs.src_latents.device(),
                )?;
                let (quantized, _indices, _mask) =
                    self.tokenize(inputs.src_latents, inputs.silence_latent, &attention_mask)?;
                let detokenized = self.detokenize(&quantized)?;
                detokenized.narrow(1, 0, latent_len)?
            };

            let is_covers_3d = inputs
                .is_covers
                .gt(0u8)?
                .unsqueeze(1)?
                .unsqueeze(2)?
                .broadcast_as(inputs.src_latents.shape())?;
            is_covers_3d.where_cond(&lm_hints_25hz, inputs.src_latents)?
        } else {
            inputs.src_latents.clone()
        };

        // Context = [src_latents, chunk_masks] along feature dim
        let context_latents = Tensor::cat(&[&src_latents, inputs.chunk_masks], candle::D::Minus1)?;

        Ok(PreparedCondition {
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
        })
    }

    /// Clear cross-attention KV caches. Call between generations with different
    /// encoder states (different prompts).
    pub fn clear_kv_cache(&mut self) {
        self.decoder.clear_kv_cache();
    }

    /// Generate audio latents from prepared conditions.
    ///
    /// Returns the denoised latent `(B, T, acoustic_dim)` ready for VAE decode.
    /// Automatically caches cross-attention K/V on the first denoising step and
    /// reuses them for subsequent steps (encoder states are constant).
    ///
    /// When `non_cover_condition` is provided and `opts.audio_cover_strength < 1.0`,
    /// the loop switches from cover to non-cover condition partway through and
    /// clears the KV cache at the transition.
    pub fn generate_latents(
        &mut self,
        condition: &PreparedCondition,
        non_cover_condition: Option<&PreparedCondition>,
        opts: &GenerateOptions,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        // Clear KV cache so it gets rebuilt for this generation's encoder states.
        self.decoder.clear_kv_cache();

        let latent_frames = condition.context_latents.dim(1)?;
        let acoustic_dim = self.config.audio_acoustic_hidden_dim;

        if let Some(seed) = opts.seed {
            device.set_seed(seed)?;
        }
        let noise =
            Tensor::randn(0f32, 1f32, (1, latent_frames, acoustic_dim), device)?.to_dtype(dtype)?;

        let mut xt = noise;

        if opts.is_turbo {
            // Turbo: discrete schedule, no CFG, final step = x0 prediction
            let t_schedule = match &opts.timesteps {
                Some(ts) => sampling::get_custom_turbo_schedule(ts)?,
                None => sampling::get_turbo_schedule(opts.shift),
            };
            let num_steps = t_schedule.len();
            let cover_steps = (num_steps as f64 * opts.audio_cover_strength).round() as usize;
            let mut active = condition;

            for step in 0..num_steps {
                // Switch to non-cover condition at the transition point
                if let Some(nc) = non_cover_condition {
                    if step == cover_steps && step > 0 {
                        active = nc;
                        self.decoder.clear_kv_cache();
                    }
                }

                let t_curr = t_schedule[step];
                let t_tensor =
                    Tensor::from_vec(vec![t_curr as f32], (1,), device)?.to_dtype(dtype)?;

                let vt = self.decoder.forward(
                    &xt,
                    &t_tensor,
                    &t_tensor,
                    None,
                    &active.encoder_hidden_states,
                    Some(&active.encoder_attention_mask),
                    &active.context_latents,
                )?;

                if step == num_steps - 1 {
                    xt = sampling::get_x0_from_noise(&xt, &vt, t_curr)?;
                } else {
                    let t_next = t_schedule[step + 1];
                    match opts.infer_method {
                        sampling::InferMethod::Ode => {
                            xt = sampling::euler_step(&xt, &vt, t_curr - t_next)?;
                        }
                        sampling::InferMethod::Sde => {
                            xt = sampling::sde_step(&xt, &vt, t_curr, t_next)?;
                        }
                    }
                }
            }
        } else {
            // Base/SFT: continuous schedule with APG guidance
            let schedule = sampling::get_schedule(opts.num_steps, opts.shift);
            let mut momentum_buffer = sampling::MomentumBuffer::new(-0.75);
            let do_cfg = opts.guidance_scale > 1.0;
            let cover_steps = (opts.num_steps as f64 * opts.audio_cover_strength).round() as usize;

            // Build doubled tensors for CFG from cover condition
            let build_cfg_tensors =
                |cond: &PreparedCondition, null_emb: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
                    let null_enc = null_emb
                        .broadcast_as(cond.encoder_hidden_states.shape())?
                        .contiguous()?;
                    if do_cfg {
                        let enc = Tensor::cat(&[&cond.encoder_hidden_states, &null_enc], 0)?;
                        let mask = Tensor::cat(
                            &[&cond.encoder_attention_mask, &cond.encoder_attention_mask],
                            0,
                        )?;
                        let ctx = Tensor::cat(&[&cond.context_latents, &cond.context_latents], 0)?;
                        Ok((enc, mask, ctx))
                    } else {
                        Ok((
                            cond.encoder_hidden_states.clone(),
                            cond.encoder_attention_mask.clone(),
                            cond.context_latents.clone(),
                        ))
                    }
                };

            let (mut enc_cond, mut enc_mask_cond, mut ctx_doubled) =
                build_cfg_tensors(condition, &self.null_condition_emb)?;

            let (cfg_start, cfg_end) = opts.cfg_interval;

            for step in 0..opts.num_steps {
                // Switch to non-cover condition at the transition point
                if let Some(nc) = non_cover_condition {
                    if step == cover_steps && step > 0 {
                        self.decoder.clear_kv_cache();
                        let rebuilt = build_cfg_tensors(nc, &self.null_condition_emb)?;
                        enc_cond = rebuilt.0;
                        enc_mask_cond = rebuilt.1;
                        ctx_doubled = rebuilt.2;
                    }
                }

                let t_curr = schedule[step];
                let t_prev = schedule[step + 1];

                let x_input = if do_cfg {
                    Tensor::cat(&[&xt, &xt], 0)?
                } else {
                    xt.clone()
                };
                let batch_size = x_input.dim(0)?;
                let t_tensor =
                    Tensor::from_vec(vec![t_curr as f32; batch_size], (batch_size,), device)?
                        .to_dtype(dtype)?;

                let vt = self.decoder.forward(
                    &x_input,
                    &t_tensor,
                    &t_tensor,
                    None,
                    &enc_cond,
                    Some(&enc_mask_cond),
                    &ctx_doubled,
                )?;

                let vt = if do_cfg {
                    let v_cond = vt.narrow(0, 0, 1)?;
                    let v_uncond = vt.narrow(0, 1, 1)?;
                    let apply_guidance = t_curr >= cfg_start && t_curr <= cfg_end;
                    if apply_guidance {
                        sampling::apg_forward(
                            &v_cond,
                            &v_uncond,
                            opts.guidance_scale,
                            &mut momentum_buffer,
                            2.5,
                        )?
                    } else {
                        v_cond
                    }
                } else {
                    vt
                };

                match opts.infer_method {
                    sampling::InferMethod::Ode => {
                        xt = sampling::euler_step(&xt, &vt, t_curr - t_prev)?;
                    }
                    sampling::InferMethod::Sde => {
                        xt = sampling::sde_step(&xt, &vt, t_curr, t_prev)?;
                    }
                }
            }
        }

        Ok(xt)
    }
}
