//! Stella v5 model implementation.
//!
//! Stella is a dense text embedding model optimized for retrieval and similarity tasks.
//! This implementation provides support for multiple embedding dimensions.
//!
//! Key characteristics:
//! - Dense text embeddings optimized for similarity search
//! - Multiple output dimension support (256 to 8192)
//! - Grouped query attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//!
//! References:
//! - [MRL Framework](https://arxiv.org/abs/2205.13147)
//! - [Model Card](https://huggingface.co/dunzhang/stella_en_1.5B_v5)
//!

use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, Error, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, Activation, LayerNorm, VarBuilder};
use std::sync::Arc;

// internal representation for identifying which model is being used
#[derive(Debug, Copy, Clone, PartialEq, serde::Deserialize)]
pub enum ModelVariant {
    Large, // 1.5B
    Small, // 400M
}

impl Default for ModelVariant {
    fn default() -> Self {
        Self::Large
    }
}

// Same as `qwen2` family of models with the exception being the `embed_head`
// The final `output` causal modelling head is swapped with a learned `dense` layer, `embed_head`
#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub variant: ModelVariant,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub embed_head: EmbedHead,
    pub norm_eps: f64,             // RMSNorm for 1.5B || LayerNorm for 400M
    pub activation_fn: Activation, // Silu for 1.5B || Gelu for 400M
    // Unique to 1.5B
    pub num_key_value_heads: usize,
    // Unique to 400M
    pub type_vocab_size: usize,
    pub scaling_factor: f64,
}

// Excerpt from `stella` model card:
// `Stella_en_1.5B_v5` models have been trained on [MRL](https://arxiv.org/abs/2205.13147) enabling multiple output dimensions
// Embed head represents the config for various embedding dims supported
#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize)]
pub struct EmbedHead {
    pub in_features: usize,
    pub out_features: usize,
}

/// An enum variant representing the Embedding head dimensions `stella` is trained on
/// As the [model-card](https://huggingface.co/dunzhang/stella_en_1.5B_v5#introduction) suggests, D1024 is good enough for most cases
#[derive(Debug, Clone, Copy)]
pub enum EmbedDim {
    Dim256,
    Dim768,
    Dim1024,
    Dim2048,
    Dim4096,
    Dim6144,
    Dim8192,
}

impl Default for EmbedDim {
    fn default() -> Self {
        Self::Dim1024
    }
}

impl EmbedDim {
    pub fn config(&self, in_features: usize) -> EmbedHead {
        EmbedHead {
            in_features,
            out_features: match &self {
                Self::Dim256 => 256,
                Self::Dim768 => 768,
                Self::Dim1024 => 1024,
                Self::Dim2048 => 2048,
                Self::Dim4096 => 4096,
                Self::Dim6144 => 6144,
                Self::Dim8192 => 8192,
            },
        }
    }
}

// Initialize a new `stella_en` model - with 400M variant or 1.5B variant
impl Config {
    /// Initialize a new `stella_en_1.5B_v5`` model with given embedding dim
    pub fn new_1_5_b_v5(embed_dim: EmbedDim) -> Self {
        // Representing config.json at https://huggingface.co/dunzhang/stella_en_1.5B_v5/blob/main/config.json
        // Removed `sliding_window` related config which is basically being carried forward from `qwen2` but not used here
        Self {
            variant: ModelVariant::Large,
            activation_fn: candle_nn::Activation::Silu,
            vocab_size: 151646,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            max_position_embeddings: 131072,
            rope_theta: 1000000.,
            norm_eps: 1e-06,
            embed_head: embed_dim.config(1536),
            ..Default::default()
        }
    }

    /// Initialize new `stella_en_400M_v5`
    pub fn new_400_m_v5(embed_dim: EmbedDim) -> Self {
        Self {
            variant: ModelVariant::Small,
            vocab_size: 30528,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            norm_eps: 1e-12,
            scaling_factor: 2.0,
            rope_theta: 160000.0,
            activation_fn: Activation::Gelu,
            embed_head: embed_dim.config(1024),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        // Factoring in `scaling factor` for `400M` variant
        let max_seq_len = if cfg.scaling_factor == 0. {
            cfg.max_position_embeddings
        } else {
            ((cfg.max_position_embeddings as f64) * cfg.scaling_factor) as usize
        };

        // let rot_dim = if cfg.variant == ModelVariant::Small { dim / 2 } else { dim };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| {
                // Scaled rope_theta for 400M variant
                let rope_theta = if cfg.scaling_factor == 0. {
                    cfg.rope_theta
                } else {
                    cfg.rope_theta * cfg.scaling_factor
                };
                let mut freq = 1. / rope_theta.powf(i as f64 / dim as f64);

                if cfg.scaling_factor != 0. {
                    freq /= cfg.scaling_factor.powf(2.0 / (dim as f64))
                }

                freq as f32
            })
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;

        // Calculate position embeddings with scaled sequence length
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        // if cfg.variant == ModelVariant::Small {
        //     freqs = Tensor::cat(&[&freqs, &freqs], 1)?
        // }

        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    // TODO: re-visit this
    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    variant: ModelVariant,
    gate_proj: Linear,
    up_proj: Option<Linear>, // `up_proj` only for 1.5B variant
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let (gate_proj, up_proj, down_proj) = match cfg.variant {
            ModelVariant::Large => (
                linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?,
                Some(linear_no_bias(
                    hidden_sz,
                    intermediate_sz,
                    vb.pp("up_proj"),
                )?),
                linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?,
            ),
            ModelVariant::Small => (
                linear_no_bias(hidden_sz, intermediate_sz * 2, vb.pp("up_gate_proj"))?,
                None,
                linear(intermediate_sz, hidden_sz, vb.pp("down_proj"))?,
            ),
        };

        Ok(Self {
            variant: cfg.variant,
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.activation_fn,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = self.gate_proj.forward(xs)?;

        let (lhs, rhs) = match self.variant {
            ModelVariant::Large => {
                let lhs = up.apply(&self.act_fn)?;
                let rhs = xs.apply(self.up_proj.as_ref().unwrap())?;

                (lhs, rhs)
            }
            ModelVariant::Small => {
                // Get the dimensions
                let (_batch_size, _seq_len, hidden_dim) = up.dims3()?;
                let split_size = hidden_dim / 2;

                // Split along the last dimension (hidden_dim)
                let up_states = up.narrow(2, 0, split_size)?;
                let gate = up.narrow(2, split_size, split_size)?.apply(&self.act_fn)?;

                (up_states, gate)
            }
        };

        (lhs * rhs)?.apply(&self.down_proj)
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
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    variant: ModelVariant,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = if num_kv_heads > 0 {
            num_heads / num_kv_heads
        } else {
            0
        };
        let head_dim = hidden_sz / num_heads;

        let (qkv_proj, o_proj) = match cfg.variant {
            ModelVariant::Large => {
                // The 1.5B variant comes with separate `q, k, v` layers, let's merge it and standardize
                // Weights
                let q_w = vb
                    .pp("q_proj")
                    .get((num_heads * head_dim, hidden_sz), "weight")?;
                let k_w = vb
                    .pp("k_proj")
                    .get((num_kv_heads * head_dim, hidden_sz), "weight")?;
                let v_w = vb
                    .pp("v_proj")
                    .get((num_kv_heads * head_dim, hidden_sz), "weight")?;
                // Biases
                let q_b = vb.pp("q_proj").get(num_heads * head_dim, "bias")?;
                let k_b = vb.pp("k_proj").get(num_kv_heads * head_dim, "bias")?;
                let v_b = vb.pp("v_proj").get(num_kv_heads * head_dim, "bias")?;

                let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
                let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;

                (
                    Linear::from_weights(qkv_w, Some(qkv_b)),
                    linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?,
                )
            }
            ModelVariant::Small => (
                linear(hidden_sz, 3 * num_heads * head_dim, vb.pp("qkv_proj"))?,
                linear(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?,
            ),
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            variant: cfg.variant,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;

        let n_kv_heads = match self.variant {
            ModelVariant::Large => self.num_kv_heads,
            ModelVariant::Small => self.num_heads,
        };

        let (query_states, key_states, value_states) = match self.variant {
            ModelVariant::Large => {
                let q_sz = self.num_heads * self.head_dim;
                let kv_sz = n_kv_heads * self.head_dim;

                let q = qkv.narrow(D::Minus1, 0, q_sz)?.reshape((
                    b_sz,
                    q_len,
                    self.num_heads,
                    self.head_dim,
                ))?;
                let k = qkv.narrow(D::Minus1, q_sz, kv_sz)?.reshape((
                    b_sz,
                    q_len,
                    n_kv_heads,
                    self.head_dim,
                ))?;
                let v = qkv.narrow(D::Minus1, q_sz + kv_sz, kv_sz)?.reshape((
                    b_sz,
                    q_len,
                    n_kv_heads,
                    self.head_dim,
                ))?;

                (q, k, v)
            }
            ModelVariant::Small => {
                // Split into Q, K, V and reshape to match PyTorch shapes
                let qkv = qkv.reshape((b_sz, q_len, 3, self.num_heads, self.head_dim))?;

                (
                    qkv.i((.., .., 0, .., ..))?,
                    qkv.i((.., .., 1, .., ..))?,
                    qkv.i((.., .., 2, .., ..))?,
                )
            }
        };

        let query_states = query_states.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.transpose(1, 2)?.contiguous()?;

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states)?;

        // The 1.5B is expected to have grouped query attention
        let (key_states, value_states) = if self.variant == ModelVariant::Large {
            (
                crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?,
                crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?,
            )
        } else {
            (key_states, value_states)
        };

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = query_states.matmul(&key_states.transpose(2, 3)?)?;
            let attn_weights = (attn_weights * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

            attn_weights.matmul(&value_states)?
        };

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
enum NormType {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

#[derive(Debug, Clone)]
struct Layer {
    variant: ModelVariant,
    attention: Attention,
    mlp: MLP,
    // For 1.5B: this is `input_layernorm`
    // For 400M: this is `output_layernorm`
    layernorm: NormType,
    post_attention_layernorm: NormType,
}

impl Layer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = Attention::new(
            rotary_emb,
            cfg,
            vb.pp(if cfg.variant == ModelVariant::Large {
                "self_attn"
            } else {
                "attention"
            }),
        )?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let (layernorm, post_attention_layernorm) = match cfg.variant {
            ModelVariant::Large => (
                NormType::Rms(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    vb.pp("input_layernorm"),
                )?),
                NormType::Rms(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.norm_eps,
                    vb.pp("post_attention_layernorm"),
                )?),
            ),
            ModelVariant::Small => (
                NormType::Layer(layer_norm(
                    cfg.hidden_size,
                    candle_nn::LayerNormConfig {
                        eps: cfg.norm_eps,
                        ..Default::default()
                    },
                    vb.pp("mlp_ln"),
                )?),
                NormType::Layer(layer_norm(
                    cfg.hidden_size,
                    candle_nn::LayerNormConfig {
                        eps: cfg.norm_eps,
                        ..Default::default()
                    },
                    vb.pp("attn_ln"),
                )?),
            ),
        };

        Ok(Self {
            variant: cfg.variant,
            attention,
            mlp,
            layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Here, the application of normalizations and activation calculations differ
        // For Large [1.5B]:
        //  residual = x
        //  state = other_layernorm(xs)
        //  state = attention(state)
        //  state += residual
        //  residual = state
        //  state = mlp(attention_layernorm(state))
        //  -> residual + state
        // For Small [400M]:
        //  residual = x;
        //  state = attention(x)
        //  state += residual
        //  state = attention_layernorm(state)
        //  residual = state
        //  state = mlp(state)
        //  state += residual
        //  -> other_layernorm(state)
        let residual = xs;

        match self.variant {
            ModelVariant::Large => {
                let (attn_ln, input_ln) = if let (NormType::Rms(attn_ln), NormType::Rms(input_ln)) =
                    (&self.post_attention_layernorm, &self.layernorm)
                {
                    (attn_ln, input_ln)
                } else {
                    return Err(candle::error::Error::Msg(
                        "Stella 1.5B expects RMSNorm".to_string(),
                    ));
                };

                let xs = input_ln.forward(xs)?;
                let xs = (self.attention.forward(&xs, attention_mask)? + residual)?;

                let residual = &xs;
                let xs = xs.apply(attn_ln)?.apply(&self.mlp)?;

                residual + xs
            }
            ModelVariant::Small => {
                let (attn_ln, output_ln) =
                    if let (NormType::Layer(attn_ln), NormType::Layer(input_ln)) =
                        (&self.post_attention_layernorm, &self.layernorm)
                    {
                        (attn_ln, input_ln)
                    } else {
                        return Err(candle::error::Error::Msg(
                            "Stella 400M expects RMSNorm".to_string(),
                        ));
                    };

                let xs = (self.attention.forward(xs, attention_mask)? + residual)?;
                let xs = attn_ln.forward(&xs)?;

                let residual = &xs;
                let xs = (self.mlp.forward(&xs)? + residual)?;

                output_ln.forward(&xs)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Embeddings {
    variant: ModelVariant,
    // For 1.5B: this is the `embed_tokens`
    // For 400M: this is the `word_embeddings`
    embeddings: candle_nn::Embedding,
    // folloing are specifically for 400M
    token_type_embeddings: Option<candle_nn::Embedding>,
    layer_norm: Option<LayerNorm>,
    position_ids: Option<Tensor>,
}

impl Embeddings {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (embeddings, token_type_embeddings, layer_norm, position_ids) = match cfg.variant {
            ModelVariant::Large => (
                candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?,
                None,
                None,
                None,
            ),
            ModelVariant::Small => {
                let vb = vb.pp("embeddings");
                let weight = vb.pp("LayerNorm").get_with_hints(
                    cfg.hidden_size,
                    "weight",
                    candle_nn::Init::Const(1.0),
                )?;
                let bias = vb.pp("LayerNorm").get_with_hints(
                    cfg.hidden_size,
                    "bias",
                    candle_nn::Init::Const(0.0),
                )?;
                let dev = bias.device().clone();

                let layer_norm = candle_nn::LayerNorm::new(weight, bias, cfg.norm_eps);

                (
                    candle_nn::embedding(
                        cfg.vocab_size,
                        cfg.hidden_size,
                        vb.pp("word_embeddings"),
                    )?,
                    Some(candle_nn::embedding(
                        cfg.type_vocab_size,
                        cfg.hidden_size,
                        vb.pp("token_type_embeddings"),
                    )?),
                    Some(layer_norm),
                    Some(Tensor::arange(
                        0u32,
                        cfg.max_position_embeddings as u32,
                        &dev,
                    )?),
                )
            }
        };

        Ok(Self {
            variant: cfg.variant,
            embeddings,
            token_type_embeddings,
            layer_norm,
            position_ids,
        })
    }
}

impl Module for Embeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embd = self.embeddings.forward(xs)?;
        // For 1.5B just forward the embeddings
        if self.variant == ModelVariant::Large {
            return Ok(embd);
        }

        let (token_type_embed, layer_norm, pos_ids) =
            if let (Some(token_type_embd), Some(layer_norm), Some(position_ids)) = (
                &self.token_type_embeddings,
                &self.layer_norm,
                &self.position_ids,
            ) {
                (token_type_embd, layer_norm, position_ids)
            } else {
                return Err(Error::Msg(
                    "Stella 400M requires `token_type_embeddings`, `layer_norm` and `position_ids`"
                        .to_string(),
                ));
            };

        let (batch_size, seq_length) = xs.dims2()?;

        let pos_ids = pos_ids
            .as_ref()
            .narrow(0, 0, seq_length)?
            .expand((batch_size, seq_length))?;

        layer_norm.forward(&embd.add(&token_type_embed.forward(&pos_ids.zeros_like()?)?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    norm: Option<RmsNorm>,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = match cfg.variant {
            ModelVariant::Large => vb.pp("model"),
            ModelVariant::Small => vb.pp("new"),
        };
        // let embed_tokens =
        //     candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let embeddings = Embeddings::new(cfg, vb_m.clone())?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = match cfg.variant {
            ModelVariant::Large => vb_m.pp("layers"),
            ModelVariant::Small => vb_m.pp("encoder").pp("layer"),
        };
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = Layer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = match cfg.variant {
            ModelVariant::Large => Some(RmsNorm::new(
                cfg.hidden_size,
                cfg.norm_eps,
                vb_m.pp("norm"),
            )?),
            ModelVariant::Small => None,
        };
        Ok(Self {
            embeddings,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    pub fn forward(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            // This is not a `causal language modelling` task, we'll need to prepare a `non-causal` attention
            Some(self.prepare_attention_mask(mask)?)
        };

        let mut xs = self.embeddings.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref())?
        }

        if let Some(n) = &self.norm {
            xs.apply(n)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
pub struct EmbeddingModel {
    base_model: Model,
    lm_head: Linear,
}

impl EmbeddingModel {
    pub fn new(cfg: &Config, base_vb: VarBuilder, embed_vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, base_vb.clone())?;
        let lm_head = linear(
            cfg.embed_head.in_features,
            cfg.embed_head.out_features,
            embed_vb.pp("linear"),
        )?;

        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = self.base_model.forward(input_ids, mask)?;
        let x = self.pool(&x, mask)?;

        // No matter what keeping the final activations as F32 helps with the accuracy
        self.lm_head.forward(&x.to_dtype(DType::F32)?) // [B_sz, dim_size]
    }

    /// Same as forward pass but normalizes the output
    pub fn forward_norm(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = self.forward(input_ids, mask)?;
        // Normalize
        x.broadcast_div(&x.sqr()?.sum_keepdim(1)?.sqrt()?)
    }

    fn pool(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mask = mask.to_dtype(x.dtype())?; // [B_Sz, Seq_len]
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        // expanding the shape of the mask from [B_Sz, Seq_len] -> [B_Sz, Seq_len, Hidden_size]
        let mask_expanded = mask
            .unsqueeze(2)?
            .broadcast_as((batch_size, seq_len, hidden_dim))?; // [B_Sz, Seq_len, Hidden_dim]

        let x = (x * &mask_expanded)?;

        // Sum
        let sum_mask = mask
            .sum(1)?
            .unsqueeze(1)?
            .expand((batch_size, hidden_dim))?;
        x.sum(1)? / sum_mask
    }
}
