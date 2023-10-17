#![allow(unused)]
use crate::models::with_tracing::{linear, Embedding as E, Linear};
/// MPT model used by replit-code-v1_5-3b
/// https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/modeling_mpt.py
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};

// https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/configuration_mpt.py
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_layers: usize,
    pub(crate) expansion_ratio: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) vocab_size: usize,
    pub(crate) kv_n_heads: usize,
    // pub(crate) attn_config: AttnConfig,
}

#[derive(Debug)]
struct GroupedQueryAttention {
    wqkv: Linear,
    out_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    softmax_scale: f64,
    head_dim: usize,
    d_model: usize,
    n_heads: usize,
    kv_n_heads: usize,
    span: tracing::Span,
}

impl GroupedQueryAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wqkv_size = cfg.d_model + 2 * cfg.kv_n_heads;
        let wqkv = linear(cfg.d_model, wqkv_size, vb.pp("Wqkv"))?;
        let head_dim = cfg.d_model / cfg.n_heads;
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        let out_proj = linear(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;
        Ok(Self {
            wqkv,
            out_proj,
            kv_cache: None,
            softmax_scale,
            head_dim,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            kv_n_heads: cfg.kv_n_heads,
            span: tracing::span!(tracing::Level::TRACE, "gqa"),
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let qkv = self.wqkv.forward(xs)?;
        let query = qkv.narrow(2, 0, self.d_model)?;
        let kv_size = self.kv_n_heads * self.head_dim;
        let key = qkv.narrow(2, self.d_model, kv_size)?;
        let value = qkv.narrow(2, self.d_model + kv_size, kv_size)?;
        // scaled_multihead_dot_product_attention
        let query = query
            .reshape((b_size, seq_len, self.n_heads, ()))?
            .transpose(1, 2)?;
        let key = key
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .permute((0, 2, 3, 1))?;
        let value = value
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .transpose(1, 2)?;
        let (key, value) = match &self.kv_cache {
            None => (key, value),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key], 3)?;
                let v = Tensor::cat(&[prev_v, &value], 2)?;
                (k, v)
            }
        };
        // TODO:
        // k = k.repeat_interleave(n_heads // kv_n_heads, dim=1)
        // v = v.repeat_interleave(n_heads // kv_n_heads, dim=1)
        let attn_weights = (query.matmul(&key)? * self.softmax_scale)?;
        // TODO: attn_bias, alibi
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&value)?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        attn_output.apply(&self.out_proj)
    }
}
