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
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
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
            crate::models::dac::Model::new(&cfg.audio_encoder, vb.pp("audio_encoder"))?;
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
        mut lp: LogitsProcessor,
        max_steps: usize,
    ) -> Result<Tensor> {
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
        let prompt_len = prompt_hidden_states.dim(1)?;
        for step in 0..max_steps {
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
                if audio_tokens[logit_idx] != self.pad_token_id {
                    let logit = logit.i((0, logit.dim(1)? - 1))?;
                    let token = lp.sample(&logit)?;
                    audio_tokens[logit_idx] = token
                }
            }
            if audio_tokens.iter().all(|v| v == &self.pad_token_id) {
                break;
            }
            for (cb_idx, &token) in audio_tokens.iter().enumerate() {
                if token != self.decoder_start_token_id && token != self.pad_token_id {
                    all_audio_tokens[cb_idx].push(token)
                }
            }
        }

        let min_len = all_audio_tokens.iter().map(|v| v.len()).min().unwrap_or(0);
        all_audio_tokens.iter_mut().for_each(|v| {
            v.resize(min_len, 0);
        });
        let all_audio_tokens = Tensor::new(all_audio_tokens, &candle::Device::Cpu)?;
        Ok(all_audio_tokens)
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
