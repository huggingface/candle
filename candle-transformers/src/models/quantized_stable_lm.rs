//! Module for quantized StableLM implementation.
//!
//! StableLM is a series of open-source large language models
//! optimized for performance and stability. This implementation
//! provides quantization support for efficient model deployment.
//!
//! Key characteristics:
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - [StableLM](https://github.com/Stability-AI/StableLM)
//!

use crate::quantized_nn::{layer_norm, linear, linear_no_bias, Embedding, Linear};
pub use crate::quantized_var_builder::VarBuilder;
use candle::{quantized::QuantizedBackend, DType, Module, Result, Tensor, D};
use candle_nn::{Activation, LayerNorm};
use std::sync::Arc;

pub use crate::models::stable_lm::Config;
use crate::models::stable_lm::RotaryEmbedding;

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP<QB: QuantizedBackend> {
    gate_proj: Linear<QB>,
    up_proj: Linear<QB>,
    down_proj: Linear<QB>,
    act_fn: Activation,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> MLP<QB> {
    fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for MLP<QB>
where
    Linear<QB>: Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let _enter = self.span.enter();
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

type KVCache<QB> = (
    Tensor<<QB as QuantizedBackend>::Storage>,
    Tensor<<QB as QuantizedBackend>::Storage>,
);
#[derive(Debug, Clone)]
struct Attention<QB: QuantizedBackend> {
    q_proj: Linear<QB>,
    k_proj: Linear<QB>,
    v_proj: Linear<QB>,
    o_proj: Linear<QB>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding<QB::Storage>>,
    kv_cache: Option<KVCache<QB>>,
    use_cache: bool,
    rotary_ndims: usize,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> Attention<QB> {
    fn new(
        rotary_emb: Arc<RotaryEmbedding<QB::Storage>>,
        cfg: &Config,
        vb: VarBuilder<QB>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let linear_layer = if cfg.use_qkv_bias {
            linear
        } else {
            linear_no_bias
        };
        let q_proj = linear_layer(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_layer(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_layer(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: cfg.num_kv_groups(),
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
            use_cache: cfg.use_cache,
            rotary_ndims: cfg.rotary_ndims(),
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor<QB::Storage>,
        attention_mask: Option<&Tensor<QB::Storage>>,
        seqlen_offset: usize,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let _enter = self.span.enter();
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

        let (rot_ndims, pass_ndims) = (self.rotary_ndims, self.head_dim - self.rotary_ndims);
        let query_rot = query_states.narrow(D::Minus1, 0, rot_ndims)?;
        let query_pass = query_states.narrow(D::Minus1, rot_ndims, pass_ndims)?;
        let key_rot = key_states.narrow(D::Minus1, 0, rot_ndims)?;
        let key_pass = key_states.narrow(D::Minus1, rot_ndims, pass_ndims)?;
        let (query_rot, key_rot) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_rot, &key_rot, seqlen_offset)?;
        let query_states = Tensor::cat(&[query_rot, query_pass], D::Minus1)?.contiguous()?;
        let key_states = Tensor::cat(&[key_rot, key_pass], D::Minus1)?.contiguous()?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        if self.use_cache {
            self.kv_cache = Some((key_states.clone(), value_states.clone()));
        }

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
struct DecoderLayer<QB: QuantizedBackend> {
    self_attn: Attention<QB>,
    mlp: MLP<QB>,
    input_layernorm: LayerNorm<QB::Storage>,
    post_attention_layernorm: LayerNorm<QB::Storage>,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> DecoderLayer<QB> {
    fn new(
        rotary_emb: Arc<RotaryEmbedding<QB::Storage>>,
        cfg: &Config,
        vb: VarBuilder<QB>,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor<QB::Storage>,
        attention_mask: Option<&Tensor<QB::Storage>>,
        seqlen_offset: usize,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let _enter = self.span.enter();
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model<QB: QuantizedBackend> {
    embed_tokens: Embedding<QB>,
    layers: Vec<DecoderLayer<QB>>,
    norm: LayerNorm<QB::Storage>,
    lm_head: Linear<QB>,
    device: QB::Device,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> Model<QB> {
    pub fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(DType::F32, cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor<QB::Storage>> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(DType::F32)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor<QB::Storage>,
        seqlen_offset: usize,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let _enter = self.span.enter();
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}
