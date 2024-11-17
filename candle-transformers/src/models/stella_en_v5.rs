use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

 // internal representation for identifying which model is being used
 #[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub enum ModelVariant {
    Large, // 1.5B
    Small  // 400M
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
    pub norm_eps: f64, // RMSNorm for 1.5B || LayerNorm for 400M
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
            // max_window_layers: 21,
            // tie_word_embeddings: false,
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

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| {
                // Scaled rope_theta for 400M variant
                let rope_theta = if cfg.scaling_factor == 0. { cfg.rope_theta } else { cfg.rope_theta * cfg.scaling_factor };
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
        let mut freqs = t.matmul(&inv_freq)?;
        if cfg.variant == ModelVariant::Small {
            freqs = Tensor::cat(&[&freqs, &freqs], 1)?
        }

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

// impl Module for EmbeddingLayer {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        
//     }
// }

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.activation_fn,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states)?;

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
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
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
        let norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            // sliding_window: 0,
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

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref())?
        }
        xs.apply(&self.norm)
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
            embed_vb.pp("linear")
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