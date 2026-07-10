//! Microsoft Phi-3 model implementation
//!
//! See Phi model details at:
//! - [Phi-3 Model](https://huggingface.co/microsoft/phi-3)
//!
//! The Phi series are decoder-only transformers designed for code and language tasks.
//! Key characteristics:
//! - Decoder-only transformer architecture
//! - RoPE embeddings
//! - Layer normalization
//! - QK normalization
//! - Mixed activation functions
//! - Improved context window handling
//!
//! References:
//! - [Hugging Face Implementation](https://huggingface.co/microsoft/phi-3)
//! - [Alternative Implementation](https://huggingface.co/microsoft/phi-3/tree/main)
//!

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use crate::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub enum RopeScalingType {
    #[serde(rename = "longrope")]
    LongRope,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    pub short_factor: Vec<f32>,
    pub long_factor: Vec<f32>,
    #[serde(rename = "type")]
    pub type_: RopeScalingType,
}

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: Option<usize>,
    pub partial_rotary_factor: Option<f64>,
    /// Sliding attention window size from `config.json` (e.g. 2047 for Phi-3-mini-4k).
    /// When set, a query only attends keys within this many positions.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    partial_dim: Option<usize>,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let partial_dim = cfg
            .partial_rotary_factor
            .as_ref()
            .map(|v| (v * cfg.head_dim() as f64) as usize);
        let dim = partial_dim.unwrap_or(cfg.head_dim());
        // Build the phase table in F32 (like mistral/llama). Using the model dtype
        // (e.g. BF16) quantizes positions and phases on long axes and corrupts RoPE.
        let freqs = match cfg.rope_scaling.as_ref() {
            None => {
                let max_seq_len = cfg.max_position_embeddings;
                let inv_freq: Vec<_> = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                    .collect();
                let inv_freq = Tensor::from_vec(inv_freq, (1, ()), dev)?.to_dtype(DType::F32)?;
                let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((max_seq_len, 1))?;
                t.matmul(&inv_freq)?
            }
            Some(rope_scaling) => {
                let inv_freq_s: Vec<_> = (0..dim)
                    .step_by(2)
                    .zip(rope_scaling.short_factor.iter())
                    .map(|(i, &f)| f / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                    .collect();
                let inv_freq_s =
                    Tensor::from_vec(inv_freq_s, (1, ()), dev)?.to_dtype(DType::F32)?;
                let max_seq_len = cfg.max_position_embeddings;
                match cfg.original_max_position_embeddings {
                    None => {
                        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                            .to_dtype(DType::F32)?
                            .reshape((max_seq_len, 1))?;
                        t.matmul(&inv_freq_s)?
                    }
                    Some(original_max_seq_len) => {
                        let t_s = Tensor::arange(0u32, original_max_seq_len as u32, dev)?
                            .to_dtype(DType::F32)?
                            .reshape((original_max_seq_len, 1))?;
                        let freq_s = t_s.matmul(&inv_freq_s)?;
                        let inv_freq_l: Vec<_> = (0..dim)
                            .step_by(2)
                            .zip(rope_scaling.long_factor.iter())
                            .map(|(i, &f)| f / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
                            .collect();
                        let inv_freq_l =
                            Tensor::from_vec(inv_freq_l, (1, ()), dev)?.to_dtype(DType::F32)?;
                        let t_l =
                            Tensor::arange(original_max_seq_len as u32, max_seq_len as u32, dev)?
                                .to_dtype(DType::F32)?
                                .reshape(((), 1))?;
                        let freq_l = t_l.matmul(&inv_freq_l)?;
                        Tensor::cat(&[&freq_s, &freq_l], 0)?
                    }
                }
            }
        };
        // sin/cos are in [-1, 1]; casting to the model dtype after is safe.
        Ok(Self {
            partial_dim,
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn rope(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = match self.partial_dim {
            None => candle_nn::rotary_emb::rope(&xs.contiguous()?, cos, sin)?,
            Some(dim) => {
                let xs_rot = xs.i((.., .., .., ..dim))?.contiguous()?;
                let xs_pass = xs.i((.., .., .., dim..))?;
                let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, cos, sin)?;
                Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?
            }
        };
        Ok(x)
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = self.rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = self.rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Causal (+ optional sliding-window) attention mask.
    ///
    /// Rows are queries in the current step (absolute pos = `seqlen_offset + i`).
    /// Columns are all key positions `0..tgt_len + seqlen_offset`.
    /// Masks key `j` when `j > q_pos` (causal) or `q_pos - j >= sliding_window`.
    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let total_len = tgt_len + seqlen_offset;
        let sliding_window = self.sliding_window;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                let q_pos = seqlen_offset + i;
                (0..total_len).map(move |j| {
                    if j > q_pos {
                        f32::NEG_INFINITY
                    } else if sliding_window.is_some_and(|w| q_pos - j >= w) {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, total_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, total_len))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        // Always build a mask for multi-token steps. For single-token decode, still
        // apply a mask once the KV length exceeds the sliding window so generation
        // past the window stays in-distribution for the model.
        let need_mask = seq_len > 1
            || self
                .sliding_window
                .is_some_and(|w| seqlen_offset + seq_len > w);
        let attention_mask = if need_mask {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        } else {
            None
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn minimal_cfg(sliding_window: Option<usize>) -> Config {
        Config {
            vocab_size: 100,
            hidden_act: candle_nn::Activation::Silu,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 4096,
            original_max_position_embeddings: None,
            partial_rotary_factor: None,
            sliding_window,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn rope_sin_cos_cast_to_model_dtype() {
        let cfg = minimal_cfg(None);
        let rope = RotaryEmbedding::new(DType::BF16, &cfg, &Device::Cpu).unwrap();
        assert_eq!(rope.sin.dtype(), DType::BF16);
        assert_eq!(rope.cos.dtype(), DType::BF16);
        assert_eq!(rope.sin.dims()[0], 4096);
    }

    #[test]
    fn config_deserializes_sliding_window() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "sliding_window": 2047
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.sliding_window, Some(2047));
    }

    #[test]
    fn sliding_window_mask_blocks_distant_keys() {
        // Emulate mask rules used by prepare_decoder_attention_mask without building weights.
        let sliding_window = Some(4usize);
        let seqlen_offset = 0usize;
        let tgt_len = 6usize;
        let total_len = tgt_len + seqlen_offset;
        let mut values = Vec::new();
        for i in 0..tgt_len {
            let q_pos = seqlen_offset + i;
            for j in 0..total_len {
                let masked = j > q_pos || sliding_window.is_some_and(|w| q_pos - j >= w);
                values.push(!masked);
            }
        }
        // Query at pos 5 may attend keys 2..5 only when window=4 (q_pos - j < 4 => j > 1).
        let row5 = &values[5 * total_len..(5 + 1) * total_len];
        assert!(!row5[0]); // j=0: 5-0 >= 4 → masked
        assert!(!row5[1]); // j=1: 5-1 >= 4 → masked
        assert!(row5[2]); // j=2: 5-2 < 4 → allowed
        assert!(row5[5]); // self
        assert!(!row5.iter().skip(6).any(|&x| x)); // no future keys in total_len=6
    }
}
