//! Parler Model implementation for parler_tts text-to-speech synthesis
//!
//! Implements a transformer-based decoder architecture for generating audio tokens
//! from text using discrete tokens. The model converts text into audio segments
//! using multiple codebooks of quantized audio tokens.
//!
//! The model architecture includes:
//! - Multi-head attention layers for text and audio processing
//! - Feed-forward networks
//! - Layer normalization
//! - Positional embeddings
//! - Multiple codebook prediction heads
//!
//! The implementation follows the original parler_tts architecture while focusing
//! on audio token generation for text-to-speech synthesis.
//!

use crate::generation::LogitsProcessor;
use crate::models::t5;
use candle::{IndexOp, Result, Tensor};
use candle_nn::{layer_norm, linear_b as linear, Activation, LayerNorm, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub ffn_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub num_cross_attention_key_value_heads: Option<usize>,
    pub activation_function: Activation,
    pub hidden_size: usize,
    pub scale_embedding: bool,
    pub num_codebooks: usize,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub tie_word_embeddings: bool,
    pub rope_embeddings: bool,
    pub rope_theta: f64,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub decoder: DecoderConfig,
    pub text_encoder: t5::Config,
    pub vocab_size: usize,
    pub audio_encoder: crate::models::dac::Config,
}

#[derive(Debug, Clone)]
pub struct Attention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    is_causal: bool,
    kv_cache: Option<(Tensor, Tensor)>,
    scaling: f64,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl Attention {
    fn new(
        num_kv_heads: usize,
        is_causal: bool,
        cfg: &DecoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.rope_embeddings {
            candle::bail!("rope embeddings are not supported");
        }
        let embed_dim = cfg.hidden_size;
        let head_dim = embed_dim / cfg.num_attention_heads;
        let kv_out_dim = num_kv_heads * head_dim;
        let k_proj = linear(embed_dim, kv_out_dim, false, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, kv_out_dim, false, vb.pp("v_proj"))?;
        let q_proj = linear(embed_dim, embed_dim, false, vb.pp("q_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, false, vb.pp("out_proj"))?;
        Ok(Self {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            is_causal,
            kv_cache: None,
            scaling: (head_dim as f64).powf(-0.5),
            num_heads: cfg.num_attention_heads,
            num_kv_heads,
            num_kv_groups: cfg.num_attention_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?
            .reshape((b_sz, tgt_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = match key_value_states {
            Some(states) => states.apply(&self.k_proj)?,
            None => xs.apply(&self.k_proj)?,
        };
        let key_states = key_states
            .reshape((b_sz, (), self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = match key_value_states {
            Some(states) => states.apply(&self.v_proj)?,
            None => xs.apply(&self.v_proj)?,
        };
        let value_states = value_states
            .reshape((b_sz, (), self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        if self.is_causal {
            self.kv_cache = Some((key_states.clone(), value_states.clone()));
        }

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, ()))?
            .apply(&self.out_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: Activation,
}

impl DecoderLayer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        let kv_heads_cross = cfg.num_cross_attention_key_value_heads.unwrap_or(kv_heads);

        let self_attn = Attention::new(kv_heads, true, cfg, vb.pp("self_attn"))?;
        let encoder_attn = Attention::new(kv_heads_cross, false, cfg, vb.pp("encoder_attn"))?;
        let self_attn_layer_norm =
            layer_norm(cfg.hidden_size, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn_layer_norm =
            layer_norm(cfg.hidden_size, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear(cfg.hidden_size, cfg.ffn_dim, false, vb.pp("fc1"))?;
        let fc2 = linear(cfg.ffn_dim, cfg.hidden_size, false, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.hidden_size, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: cfg.activation_function,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_xs: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention
        let residual = xs;
        let xs = xs.apply(&self.self_attn_layer_norm)?;
        let xs = self.self_attn.forward(&xs, None, attention_mask)?;
        let xs = (residual + xs)?;

        // Cross attention
        let residual = &xs;
        let xs = xs.apply(&self.encoder_attn_layer_norm)?;
        let xs = self
            .encoder_attn
            .forward(&xs, Some(encoder_xs), encoder_attention_mask)?;
        let xs = (residual + xs)?;

        // Fully connected
        let residual = &xs;
        let xs = xs
            .apply(&self.final_layer_norm)?
            .apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
        self.encoder_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    embed_tokens: Vec<candle_nn::Embedding>,
    embed_positions: Tensor,
    layers: Vec<DecoderLayer>,
    layer_norm: LayerNorm,
    num_codebooks: usize,
    hidden_size: usize,
    lm_heads: Vec<Linear>,
    dtype: candle::DType,
    eos_token_id: u32,
    vocab_size: usize,
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.num_codebooks == 0 || cfg.eos_token_id >= cfg.vocab_size {
            candle::bail!("Parler requires nonempty codebooks and an in-vocabulary EOS token");
        }
        let eos_token_id = u32::try_from(cfg.eos_token_id)?;
        let vb_d = vb.pp("model.decoder");
        let mut embed_tokens = Vec::with_capacity(cfg.num_codebooks);
        let vb_e = vb_d.pp("embed_tokens");
        for embed_idx in 0..cfg.num_codebooks {
            let e = candle_nn::embedding(cfg.vocab_size + 1, cfg.hidden_size, vb_e.pp(embed_idx))?;
            embed_tokens.push(e)
        }
        let embed_positions = vb_d.get(
            (cfg.max_position_embeddings, cfg.hidden_size),
            "embed_positions.weights",
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_d.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let layer_norm = layer_norm(cfg.hidden_size, 1e-5, vb_d.pp("layer_norm"))?;

        let mut lm_heads = Vec::with_capacity(cfg.num_codebooks);
        let vb_l = vb.pp("lm_heads");
        for lm_idx in 0..cfg.num_codebooks {
            let lm_head = linear(cfg.hidden_size, cfg.vocab_size, false, vb_l.pp(lm_idx))?;
            lm_heads.push(lm_head)
        }
        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            num_codebooks: cfg.num_codebooks,
            lm_heads,
            hidden_size: cfg.hidden_size,
            dtype: vb.dtype(),
            eos_token_id,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        prompt_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_xs: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Vec<Tensor>> {
        let (b_sz, num_codebooks, seq_len) = input_ids.dims3()?;
        if num_codebooks != self.num_codebooks {
            candle::bail!("unexpected num codebooks in input {:?}", input_ids.shape())
        }
        let mut inputs_embeds = Tensor::zeros(
            (b_sz, seq_len, self.hidden_size),
            self.dtype,
            input_ids.device(),
        )?;
        for (idx, embs) in self.embed_tokens.iter().enumerate() {
            let e = input_ids.i((.., idx))?.apply(embs)?;
            inputs_embeds = (inputs_embeds + e)?
        }
        let inputs_embeds = match prompt_hidden_states {
            None => inputs_embeds,
            Some(pis) => Tensor::cat(&[pis, &inputs_embeds], 1)?,
        };
        let embed_positions = self
            .embed_positions
            .i(seqlen_offset..seqlen_offset + inputs_embeds.dim(1)?)?;
        let mut xs = (inputs_embeds + embed_positions.unsqueeze(0))?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask, encoder_xs, encoder_attention_mask)?;
        }
        let xs = xs.apply(&self.layer_norm)?;
        let mut lm_logits = Vec::with_capacity(self.num_codebooks);
        for lm_head in self.lm_heads.iter() {
            let logits = xs.apply(lm_head)?;
            lm_logits.push(logits)
        }
        Ok(lm_logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

/// EOS may be sampled only by the first unfinished codebook (or earlier
/// finished codebooks). This is the constraint in ParlerTTSLogitsProcessor:
/// allowing a later codebook to end early can truncate valid speech in every
/// codebook when the delayed streams are brought back to a common length.
#[derive(Debug)]
struct EosProcessor {
    first_unfinished: usize,
    codebooks: usize,
    eos_mask: Tensor,
    negative_infinity: Tensor,
}
impl EosProcessor {
    fn new(
        eos: u32,
        codebooks: usize,
        vocab: usize,
        dtype: candle::DType,
        device: &candle::Device,
    ) -> Result<Self> {
        if codebooks == 0 || eos as usize >= vocab {
            candle::bail!("invalid Parler EOS processor configuration");
        }
        let mask = (0..vocab)
            .map(|token| u8::from(token == eos as usize))
            .collect::<Vec<_>>();
        Ok(Self {
            first_unfinished: 0,
            codebooks,
            eos_mask: Tensor::new(mask.as_slice(), device)?,
            negative_infinity: Tensor::full(f32::NEG_INFINITY, vocab, device)?.to_dtype(dtype)?,
        })
    }
    /// Call once per generation step, before sampling any codebook. A new EOS
    /// in this step must not unlock the next codebook until the following step.
    fn advance(&mut self, finished: &[bool]) -> Result<()> {
        if finished.len() != self.codebooks {
            candle::bail!("Parler EOS state/codebook mismatch");
        }
        if finished[self.first_unfinished] && self.first_unfinished + 1 < self.codebooks {
            self.first_unfinished += 1;
        }
        Ok(())
    }
    fn apply(&self, book: usize, logits: &Tensor) -> Result<Tensor> {
        if book >= self.codebooks {
            candle::bail!("Parler EOS codebook index out of range");
        }
        if book > self.first_unfinished {
            self.eos_mask.where_cond(&self.negative_infinity, logits)
        } else {
            Ok(logits.clone())
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Audio codes on the CPU, with shape `(num_codebooks, frames)`.
    pub audio_codes: Tensor,
    pub finish_reason: GenerationFinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    EndOfSequence,
    Length,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub embed_prompts: candle_nn::Embedding,
    pub enc_to_dec_proj: Option<Linear>,
    pub decoder: Decoder,
    pub text_encoder: t5::T5EncoderModel,
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,
    pub audio_encoder: crate::models::dac::Model,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_encoder = t5::T5EncoderModel::load(vb.pp("text_encoder"), &cfg.text_encoder)?;
        let decoder = Decoder::new(&cfg.decoder, vb.pp("decoder"))?;
        let embed_prompts = candle_nn::embedding(
            cfg.vocab_size,
            cfg.decoder.hidden_size,
            vb.pp("embed_prompts"),
        )?;
        let enc_to_dec_proj = if cfg.text_encoder.d_model != cfg.decoder.hidden_size {
            let proj = linear(
                cfg.text_encoder.d_model,
                cfg.decoder.hidden_size,
                true,
                vb.pp("enc_to_dec_proj"),
            )?;
            Some(proj)
        } else {
            None
        };
        let audio_encoder =
            crate::models::dac::Model::new(&cfg.audio_encoder, vb.pp("audio_encoder.model"))?;
        Ok(Self {
            decoder,
            text_encoder,
            embed_prompts,
            enc_to_dec_proj,
            decoder_start_token_id: cfg.decoder_start_token_id,
            pad_token_id: cfg.pad_token_id,
            audio_encoder,
        })
    }

    /// Note that the returned tensor uses the CPU device.
    pub fn generate(
        &mut self,
        prompt_tokens: &Tensor,
        description_tokens: &Tensor,
        lp: LogitsProcessor,
        max_steps: usize,
    ) -> Result<Tensor> {
        self.generate_with_callback(prompt_tokens, description_tokens, lp, max_steps, |_| Ok(()))
    }

    /// As `generate`, with a fallible callback before encoding and every decode
    /// step. Applications can cooperatively cancel generation without waiting
    /// for the full utterance. A callback error is propagated to the caller.
    pub fn generate_with_callback(
        &mut self,
        prompt_tokens: &Tensor,
        description_tokens: &Tensor,
        lp: LogitsProcessor,
        max_steps: usize,
        before_step: impl FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        Ok(self
            .generate_with_callback_detailed(
                prompt_tokens,
                description_tokens,
                lp,
                max_steps,
                before_step,
            )?
            .audio_codes)
    }

    /// As `generate_with_callback`, also reporting whether every codebook
    /// reached EOS or the generation exhausted `max_steps`. EOS on the final
    /// allowed step is reported as `EndOfSequence`.
    pub fn generate_with_callback_detailed(
        &mut self,
        prompt_tokens: &Tensor,
        description_tokens: &Tensor,
        mut lp: LogitsProcessor,
        max_steps: usize,
        mut before_step: impl FnMut(usize) -> Result<()>,
    ) -> Result<GenerationResult> {
        before_step(0)?;
        self.decoder.clear_kv_cache();
        self.text_encoder.clear_kv_cache();
        let encoded = self.text_encoder.forward(description_tokens)?;
        let encoded = match self.enc_to_dec_proj.as_ref() {
            None => encoded,
            Some(proj) => encoded.apply(proj)?,
        };
        let prompt_hidden_states = prompt_tokens.apply(&self.embed_prompts)?;
        let num_codebooks = self.decoder.num_codebooks;
        let mut audio_tokens = vec![self.decoder_start_token_id; num_codebooks];
        let mut all_audio_tokens = vec![vec![]; num_codebooks];
        let eos = self.decoder.eos_token_id;
        let mut finished = vec![false; num_codebooks];
        let mut first_eos_steps = vec![None; num_codebooks];
        let mut steps_executed = 0;
        let mut eos_processor = EosProcessor::new(
            eos,
            num_codebooks,
            self.decoder.vocab_size,
            self.decoder.dtype,
            prompt_tokens.device(),
        )?;
        let prompt_len = prompt_hidden_states.dim(1)?;
        let mut finish_reason = GenerationFinishReason::Length;
        for step in 0..max_steps {
            before_step(step)?;
            steps_executed = step + 1;
            eos_processor.advance(&finished)?;
            let input_ids = Tensor::from_slice(
                audio_tokens.as_slice(),
                (1, num_codebooks, 1),
                prompt_tokens.device(),
            )?;
            let (prompt_hidden_states, pos) = if step == 0 {
                (Some(&prompt_hidden_states), 0)
            } else {
                (None, step + prompt_len)
            };
            let causal_mask = if pos == 0 {
                self.prepare_causal_mask(prompt_len + 1, prompt_len + 1, input_ids.device())?
            } else {
                self.prepare_causal_mask(1, pos + 1, input_ids.device())?
            };
            let logits = self.decoder.forward(
                &input_ids,
                prompt_hidden_states,
                Some(&causal_mask),
                &encoded,
                None,
                pos,
            )?;
            for (logit_idx, logit) in logits.iter().enumerate() {
                if logit_idx > step {
                    break;
                }
                if finished[logit_idx] {
                    audio_tokens[logit_idx] = self.pad_token_id;
                } else {
                    let logit = logit.i((0, logit.dim(1)? - 1))?;
                    let logit = eos_processor.apply(logit_idx, &logit)?;
                    let token = lp.sample(&logit)?;
                    audio_tokens[logit_idx] = token;
                    if token == eos {
                        finished[logit_idx] = true;
                        first_eos_steps[logit_idx] = Some(step);
                        tracing::debug!(
                            codebook = logit_idx,
                            step,
                            code_frames = all_audio_tokens[logit_idx].len(),
                            "Parler codebook reached first EOS"
                        );
                    }
                }
            }
            if finished.iter().all(|finished| *finished) {
                finish_reason = GenerationFinishReason::EndOfSequence;
                break;
            }
            for (cb_idx, &token) in audio_tokens.iter().enumerate() {
                if !finished[cb_idx]
                    && token != self.decoder_start_token_id
                    && token != self.pad_token_id
                {
                    all_audio_tokens[cb_idx].push(token)
                }
            }
        }

        let raw_lengths = all_audio_tokens.iter().map(Vec::len).collect::<Vec<_>>();
        let min_len = raw_lengths.iter().copied().min().unwrap_or(0);
        tracing::debug!(steps = steps_executed, ?finish_reason, ?first_eos_steps, ?raw_lengths, code_frames = min_len, trimmed_frames = ?raw_lengths.iter().map(|length|length-min_len).collect::<Vec<_>>(), "Parler generation completed");
        all_audio_tokens.iter_mut().for_each(|v| {
            v.resize(min_len, 0);
        });
        let all_audio_tokens = Tensor::new(all_audio_tokens, &candle::Device::Cpu)?;
        Ok(GenerationResult {
            audio_codes: all_audio_tokens,
            finish_reason,
        })
    }

    fn prepare_causal_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        device: &candle::Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if i + kv_len < j + q_len {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (q_len, kv_len), device)
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;
    #[test]
    fn eos_logits_processor_matches_official_torch_reference() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/parler_eos.json"))
                .map_err(candle::Error::wrap)?;
        let device = candle::Device::Cpu;
        let mut processor = EosProcessor::new(4, 3, 6, candle::DType::F32, &device)?;
        for step in fixture["steps"].as_array().unwrap() {
            let finished = step["finished"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>();
            processor.advance(&finished)?;
            for book in 0..3 {
                let logits = step["scores"][book]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                let masked = processor
                    .apply(book, &Tensor::new(logits.as_slice(), &device)?)?
                    .to_vec1::<f32>()?;
                for (actual, expected) in masked
                    .iter()
                    .zip(step["expected"][book].as_array().unwrap())
                {
                    if expected.is_null() {
                        assert_eq!(*actual, f32::NEG_INFINITY);
                    } else {
                        assert_eq!(*actual, expected.as_f64().unwrap() as f32);
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn later_codebook_eos_cannot_truncate_earlier_speech() -> Result<()> {
        let device = candle::Device::Cpu;
        let mut processor = EosProcessor::new(4, 3, 6, candle::DType::F32, &device)?;
        let mut sampler = LogitsProcessor::new(0, None, None);
        let mut finished = [false; 3];
        let mut eos_steps = [None; 3];
        let mut lengths = [0; 3];
        for step in 0..5 {
            processor.advance(&finished)?;
            for book in 0..3.min(step + 1) {
                if finished[book] {
                    continue;
                }
                let eos_score = if book == 0 && step < 2 { -10f32 } else { 10. };
                let scores = Tensor::new(&[1f32, 0., 0., 0., eos_score, -1.], &device)?;
                let token = sampler.sample(&processor.apply(book, &scores)?)?;
                if token == 4 {
                    finished[book] = true;
                    eos_steps[book] = Some(step);
                } else {
                    lengths[book] += 1;
                }
            }
        }
        assert_eq!(eos_steps, [Some(2), Some(3), Some(4)]);
        assert_eq!(
            lengths,
            [2, 2, 2],
            "an early EOS in an upper codebook must not remove valid speech frames"
        );
        Ok(())
    }
    /// Opt-in QA against logits recorded by the official Parler implementation.
    /// Loads only the text encoder, prompt embedding, and decoder on CPU.
    #[test]
    #[ignore = "requires the public Parler checkpoint and official Torch QA JSON"]
    fn checkpoint_prefill_logits_match_official_torch() -> Result<()> {
        let directory =
            std::env::var("CANDLE_PARLER_CHECKPOINT_DIR").map_err(candle::Error::wrap)?;
        let reference =
            std::env::var("CANDLE_PARLER_REFERENCE_JSON").map_err(candle::Error::wrap)?;
        let directory = std::path::Path::new(&directory);
        let cfg: Config = serde_json::from_slice(
            &std::fs::read(directory.join("config.json")).map_err(candle::Error::wrap)?,
        )
        .map_err(candle::Error::wrap)?;
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference).map_err(candle::Error::wrap)?)
                .map_err(candle::Error::wrap)?;
        let device = candle::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[directory.join("model.safetensors")],
                candle::DType::F32,
                &device,
            )?
        };
        let ids = |key: &str| -> Result<Tensor> {
            let tokens = reference[key][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)
        };
        let mut encoder = t5::T5EncoderModel::load(vb.pp("text_encoder"), &cfg.text_encoder)?;
        let encoded = encoder.forward(&ids("description_ids")?)?;
        let encoded = if cfg.text_encoder.d_model == cfg.decoder.hidden_size {
            encoded
        } else {
            encoded.apply(&linear(
                cfg.text_encoder.d_model,
                cfg.decoder.hidden_size,
                true,
                vb.pp("enc_to_dec_proj"),
            )?)?
        };
        let prompt = ids("prompt_ids")?.apply(&candle_nn::embedding(
            cfg.vocab_size,
            cfg.decoder.hidden_size,
            vb.pp("embed_prompts"),
        )?)?;
        let mut decoder = Decoder::new(&cfg.decoder, vb.pp("decoder"))?;
        let length = prompt.dim(1)? + 1;
        let mask = Tensor::from_vec(
            (0..length)
                .flat_map(|i| (0..length).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
                .collect::<Vec<_>>(),
            (length, length),
            &device,
        )?;
        let audio = Tensor::new(
            vec![vec![
                vec![cfg.decoder_start_token_id];
                cfg.decoder.num_codebooks
            ]],
            &device,
        )?;
        let logits = decoder.forward(&audio, Some(&prompt), Some(&mask), &encoded, None, 0)?;
        let actual = Tensor::stack(
            &logits
                .iter()
                .map(|head| head.i((0, length - 1)))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?;
        let expected = reference["first_decoder"]["last_logits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|book| {
                book.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect::<Vec<_>>();
        let expected = Tensor::from_vec(expected, actual.shape(), &device)?;
        let difference = (&actual - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let actual_tokens = actual.argmax(candle::D::Minus1)?.to_vec1::<u32>()?;
        let expected_tokens = expected.argmax(candle::D::Minus1)?.to_vec1::<u32>()?;
        eprintln!("Parler checkpoint prefill max_difference={difference} actual={actual_tokens:?} official={expected_tokens:?}");
        assert!(
            difference < 2e-3,
            "real checkpoint prefill logits differ by {difference}"
        );
        assert_eq!(actual_tokens, expected_tokens);
        Ok(())
    }
    #[test]
    fn decoder_prompt_cross_attention_and_incremental_cache_match_official_torch() -> Result<()> {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/parler_decoder.json"))
                .map_err(candle::Error::wrap)?;
        let device = candle::Device::Cpu;
        let values = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let cfg: DecoderConfig =
            serde_json::from_value(f["config"].clone()).map_err(candle::Error::wrap)?;
        let weights = f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                let shape = value["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                Ok((
                    name.clone(),
                    Tensor::from_vec(values(&value["values"]), shape, &device)?,
                ))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let mut decoder = Decoder::new(
            &cfg,
            VarBuilder::from_tensors(weights, candle::DType::F32, &device),
        )?;
        let ids = f["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (1, 3, 4), &device)?;
        let prompt = Tensor::from_vec(values(&f["prompt"]), (1, 2, 8), &device)?;
        let context = Tensor::from_vec(values(&f["context"]), (1, 3, 8), &device)?;
        let mask = |q: usize, kv: usize| {
            Tensor::from_vec(
                (0..q)
                    .flat_map(|i| {
                        (0..kv).map(move |j| {
                            if i + kv < j + q {
                                f32::NEG_INFINITY
                            } else {
                                0.
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
                (q, kv),
                &device,
            )
        };
        let pack =
            |values: Vec<Tensor>| Tensor::stack(&values, 1)?.reshape((3, values[0].dim(1)?, 6));
        let full =
            pack(decoder.forward(&ids, Some(&prompt), Some(&mask(6, 6)?), &context, None, 0)?)?;
        let expected = Tensor::from_vec(values(&f["expected"]), (3, 6, 6), &device)?;
        let difference =
            |a: &Tensor, b: &Tensor| (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>();
        let error = difference(&full, &expected)?;
        assert!(error < 2e-6, "full decoder logits difference={error}");
        decoder.clear_kv_cache();
        let mut parts = Vec::new();
        for step in 0..4 {
            let (q, kv, pos) = if step == 0 {
                (3, 3, 0)
            } else {
                (1, step + 3, step + 2)
            };
            parts.push(pack(decoder.forward(
                &ids.narrow(2, step, 1)?,
                if step == 0 { Some(&prompt) } else { None },
                Some(&mask(q, kv)?),
                &context,
                None,
                pos,
            )?)?);
        }
        let incremental = Tensor::cat(&parts, 1)?;
        let expected = Tensor::from_vec(values(&f["expected_incremental"]), (3, 6, 6), &device)?;
        let error = difference(&incremental, &expected)?;
        assert!(error < 2e-6, "cached decoder logits difference={error}");
        assert!(difference(&full, &incremental)? < 2e-6);
        Ok(())
    }
}
