use crate::quantized_nn::{layer_norm_no_bias, linear_no_bias, Embedding, Linear};
pub use crate::quantized_var_builder::VarBuilder;
/// MPT model used by replit-code-v1_5-3b
/// https://huggingface.co/replit/replit-code-v1_5-3b/blob/main/modeling_mpt.py
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::LayerNorm;

pub use super::mpt::Config;

#[derive(Debug, Clone)]
struct GroupedQueryAttention {
    wqkv: Linear,
    out_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    softmax_scale: f64,
    head_dim: usize,
    d_model: usize,
    n_heads: usize,
    kv_n_heads: usize,
    attn_bias: Tensor,
    span: tracing::Span,
}

impl GroupedQueryAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.d_model / cfg.n_heads;
        let wqkv_size = cfg.d_model + 2 * cfg.kv_n_heads * head_dim;
        let wqkv = linear_no_bias(cfg.d_model, wqkv_size, vb.pp("Wqkv"))?;
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        let out_proj = linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;
        let attn_bias = super::mpt::build_alibi_bias(cfg)?.to_device(vb.device())?;
        Ok(Self {
            wqkv,
            out_proj,
            kv_cache: None,
            softmax_scale,
            head_dim,
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            kv_n_heads: cfg.kv_n_heads,
            attn_bias,
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
            .transpose(1, 2)?; // b,h,s,d
        let key = key
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .permute((0, 2, 3, 1))?; // b,h,d,s
        let value = value
            .reshape((b_size, seq_len, self.kv_n_heads, ()))?
            .transpose(1, 2)?; // b,h,s,d
        let (key, value) = match &self.kv_cache {
            None => (key, value),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key], 3)?;
                let v = Tensor::cat(&[prev_v, &value], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key.clone(), value.clone()));
        let query = query.contiguous()?;
        let key = crate::utils::repeat_kv(key, self.n_heads / self.kv_n_heads)?.contiguous()?;
        let value = crate::utils::repeat_kv(value, self.n_heads / self.kv_n_heads)?.contiguous()?;
        let attn_weights = (query.matmul(&key)? * self.softmax_scale)?;
        let attn_bias = {
            let s_q = query.dim(D::Minus2)?;
            let s_k = key.dim(D::Minus1)?;
            let (_, _, a_q, a_k) = self.attn_bias.dims4()?;
            let start_q = a_q.saturating_sub(s_q);
            let start_k = a_k.saturating_sub(s_k);
            self.attn_bias.i((.., .., start_q.., start_k..))?
        };
        let attn_weights = attn_weights.broadcast_add(&attn_bias)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => super::mpt::masked_fill(
                &attn_weights,
                &mask.broadcast_as(attn_weights.shape())?,
                f32::NEG_INFINITY,
            )?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&value)?
            .transpose(1, 2)?
            .flatten_from(D::Minus2)?;
        let out = attn_output.apply(&self.out_proj)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Ffn {
    up_proj: Linear,
    down_proj: Linear,
}

impl Ffn {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.d_model * cfg.expansion_ratio;
        let up_proj = linear_no_bias(cfg.d_model, hidden, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(hidden, cfg.d_model, vb.pp("down_proj"))?;
        Ok(Self { up_proj, down_proj })
    }
}

impl Module for Ffn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.up_proj)?.gelu_erf()?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct MPTBlock {
    norm1: LayerNorm, // Do we need the low-precision variant?
    attn: GroupedQueryAttention,
    norm2: LayerNorm,
    ffn: Ffn,
}

impl MPTBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm_no_bias(cfg.d_model, 1e-5, vb.pp("norm_1"))?;
        let norm2 = layer_norm_no_bias(cfg.d_model, 1e-5, vb.pp("norm_2"))?;
        let attn = GroupedQueryAttention::new(cfg, vb.pp("attn"))?;
        let ffn = Ffn::new(cfg, vb.pp("ffn"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            ffn,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.norm1)?;
        let xs = self.attn.forward(&xs, mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.norm2)?.apply(&self.ffn)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    wte: Embedding,
    blocks: Vec<MPTBlock>,
    norm_f: LayerNorm,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = Embedding::new(cfg.vocab_size, cfg.d_model, vb.pp("wte"))?;
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let block = MPTBlock::new(cfg, vb_b.pp(i))?;
            blocks.push(block)
        }
        let norm_f = layer_norm_no_bias(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        Ok(Self {
            wte,
            blocks,
            norm_f,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.wte)?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(super::mpt::get_mask(seq_len, xs.device())?)
        };
        for block in self.blocks.iter_mut() {
            xs = block.forward(&xs, mask.as_ref())?;
        }
        let xs = xs.apply(&self.norm_f)?;
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .matmul(&self.wte.embeddings().t()?)?
            .squeeze(1)?;
        Ok(logits)
    }
}
