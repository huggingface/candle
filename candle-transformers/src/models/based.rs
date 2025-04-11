//! Based from the Stanford Hazy Research group.
//!
//! See "Simple linear attention language models balance the recall-throughput tradeoff", Arora et al. 2024
//! - Simple linear attention language models balance the recall-throughput tradeoff. [Arxiv](https://arxiv.org/abs/2402.18668)
//! - [Github Rep](https://github.com/HazyResearch/based)
//! - [Blogpost](https://hazyresearch.stanford.edu/blog/2024-03-03-based)

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    conv1d_no_bias, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Conv1d, Conv1dConfig,
    Func, Linear, RmsNorm, VarBuilder,
};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LinearAttentionFeatureMapConfig {
    input_dim: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LinearAttentionConfig {
    num_heads: usize,
    feature_dim: usize,
    feature_map: LinearAttentionFeatureMapConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SlidingWindowAttentionConfig {
    num_heads: usize,
    window_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    vocab_size: usize,
    #[serde(rename = "n_embd")]
    hidden_size: usize,
    #[serde(rename = "n_inner")]
    intermediate_size: usize,
    #[serde(rename = "n_layer")]
    num_hidden_layers: usize,
    #[serde(rename = "n_head")]
    num_attention_heads: usize,

    layer_norm_epsilon: f64,
    #[serde(default = "default_rope", rename = "rotary_emb_base")]
    rope_theta: f64,

    alt_mixer_layers: Vec<usize>,
    alt_mixer_2_layers: Vec<usize>,
    #[serde(rename = "alt_mixer")]
    la: LinearAttentionConfig,
    #[serde(rename = "alt_mixer_2")]
    swa: SlidingWindowAttentionConfig,
}

fn default_rope() -> f64 {
    10_000.0
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_no_bias(cfg.hidden_size, cfg.hidden_size * 4, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

// Swiglu implementation.
// Not using Activation::Swiglu because this has the gate and y arguments switched compared to the version in candle-nn/src/ops.rs
fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[1].silu()? * &xs[0]
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.fc1)?;
        let xs = swiglu(&xs)?;
        let xs = xs.apply(&self.fc2)?;
        Ok(xs)
    }
}

// A gated convolutional block.
#[derive(Debug, Clone)]
struct BasedConv {
    in_proj: Linear,
    out_proj: Linear,
    conv: Conv1d,
    state: Tensor,
}

impl BasedConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size * 2;

        let conv1d_cfg = Conv1dConfig {
            groups: dim,
            padding: 2,
            ..Default::default()
        };

        let in_proj = linear(cfg.hidden_size, cfg.hidden_size * 4, vb.pp("in_proj"))?;
        let out_proj = linear(dim, cfg.hidden_size, vb.pp("out_proj"))?;
        let conv = conv1d_no_bias(dim, dim, 3, conv1d_cfg, vb.pp("conv.conv"))?;
        let state = Tensor::zeros((1, dim, 3), vb.dtype(), vb.device())?;
        Ok(Self {
            in_proj,
            out_proj,
            conv,
            state,
        })
    }

    fn step(&mut self, xs: &Tensor) -> Result<Tensor> {
        self.state = self.state.roll(-1, D::Minus1)?;
        let (_, _, l) = self.state.dims3()?;
        self.state = self.state.narrow(D::Minus1, 0, l - 1)?;
        self.state = Tensor::cat(&[&self.state, &xs.transpose(1, 2)?], 2)?;

        let xs = (&self.state * self.conv.weight().permute((1, 0, 2))?)?
            .sum_keepdim(0)?
            .sum(D::Minus1)?;

        let xs = xs.unsqueeze(1)?;

        Ok(xs)
    }

    fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let xs = xs.apply(&self.in_proj)?;
        let us = xs.chunk(2, D::Minus1)?;
        let (_b, l, _d) = us[0].dims3()?;
        let u_conv = if seqlen_offset > 0 {
            self.step(&us[0])?
        } else {
            let k = std::cmp::min(3, l);
            self.state = self.state.narrow(D::Minus1, 0, 3 - k)?;
            let xs = us[0].narrow(1, l - k, k)?.transpose(1, 2)?;
            self.state = Tensor::cat(&[&self.state, &xs], 2)?;

            us[0]
                .transpose(1, 2)?
                .apply(&self.conv)?
                .narrow(D::Minus1, 0, l)?
                .transpose(1, 2)?
        };

        let u_conv = u_conv.silu()?;
        let v = u_conv.broadcast_mul(&us[1])?;
        let xs = v.apply(&self.out_proj)?;

        Ok(xs)
    }
}

// Linear attention approximating softmax using second order Taylor polynomials.
#[derive(Debug, Clone)]
struct LinearAttention {
    proj_q: Linear,
    proj_k: Linear,
    proj_v: Linear,
    out_proj: Linear,
    feature_dim: usize,
    num_heads: usize,
    input_dim: usize,
    k_state: Tensor,
    kv_state: Tensor,
}

impl LinearAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_dim = cfg.la.feature_map.input_dim;
        let out_proj = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;
        let proj_k = linear_no_bias(
            cfg.hidden_size,
            cfg.la.num_heads * cfg.la.feature_dim,
            vb.pp("proj_k"),
        )?;
        let proj_q = linear_no_bias(
            cfg.hidden_size,
            cfg.la.num_heads * cfg.la.feature_dim,
            vb.pp("proj_q"),
        )?;

        let proj_v = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("proj_v"))?;
        let expanded_size = cfg.la.feature_dim.pow(2) + cfg.la.feature_dim + 1;
        let k_state = Tensor::zeros(
            (1, cfg.la.num_heads, 1, 1, expanded_size),
            vb.dtype(),
            vb.device(),
        )?;
        let kv_state = Tensor::zeros(
            (1, cfg.la.num_heads, cfg.la.feature_dim, expanded_size),
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            proj_q,
            proj_k,
            proj_v,
            out_proj,
            feature_dim: cfg.la.feature_dim,
            num_heads: cfg.la.num_heads,
            input_dim,
            k_state,
            kv_state,
        })
    }

    fn taylor_expansion(&self) -> Result<Func<'static>> {
        let r2 = std::f64::consts::SQRT_2;
        let rd = (self.input_dim as f64).sqrt();
        let rrd = rd.sqrt();

        Ok(Func::new(move |xs| {
            let dims = xs.dims();
            let mut d = dims.to_vec();
            if let Some(last) = d.last_mut() {
                *last = 1;
            };

            let x = xs
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&xs.unsqueeze(D::Minus2)?)?;
            let x = (x.flatten_from(D::Minus2)? / r2)?;
            let o = Tensor::ones(d, xs.dtype(), xs.device())?;
            let x = Tensor::cat(&[o, (xs / rrd)?, (&x / rd)?], D::Minus1)?;

            Ok(x)
        }))
    }

    fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let eps = 1e-12;

        let feature_map = self.taylor_expansion()?;

        let (b, l, d) = xs.dims3()?;
        let q = xs.apply(&self.proj_q)?;
        let k = xs.apply(&self.proj_k)?;
        let v = xs.apply(&self.proj_v)?;

        let q = q
            .reshape((b, l, self.num_heads, self.feature_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, l, self.num_heads, self.feature_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_heads, d / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = feature_map.forward(&q)?;
        let k = feature_map.forward(&k)?;

        let y = if seqlen_offset > 0 {
            let (_b, _h, l, _d) = k.dims4()?;
            let q = q.unsqueeze(D::Minus2)?;
            let k = k.unsqueeze(D::Minus2)?;
            let v = v.unsqueeze(D::Minus1)?;
            let kn = k.narrow(D::Minus1, l - 1, 1)?;
            let vn = v.narrow(D::Minus1, l - 1, 1)?;

            self.k_state = self.k_state.broadcast_add(&kn)?;
            self.kv_state = self.kv_state.broadcast_add(&kn.broadcast_mul(&vn)?)?;

            let num = q.broadcast_mul(&self.kv_state)?.sum(D::Minus1)?;
            let den = (q.broadcast_mul(&self.k_state)?.sum(D::Minus1)? + eps)?;
            num.broadcast_div(&den)?
        } else {
            self.k_state = k.sum(2)?.unsqueeze(2)?.unsqueeze(3)?;
            self.kv_state = k
                .transpose(2, 3)?
                .matmul(&v)?
                .transpose(2, 3)?
                .unsqueeze(2)?;
            let aqk = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
            let tril = Tensor::tril2(l, aqk.dtype(), aqk.device())?;
            let aqk = aqk.broadcast_mul(&tril)?.matmul(&v)?;

            let z = (1f64 / (q.mul(&k.cumsum(2)?)?.sum(D::Minus1)? + eps)?)?;
            aqk.broadcast_mul(&z.unsqueeze(D::Minus1)?)?
        };

        let (b, h, l, d) = y.dims4()?;
        let y = y.permute((0, 2, 1, 3))?.reshape((b, l, h * d))?;
        let y = self.out_proj.forward(&y)?;

        Ok(y)
    }
}

// Rotary embeddings used in local attention.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = 2048; // Hardcoded, missing from config.
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// Local attention using a small sliding window.
#[derive(Debug, Clone)]
struct SlidingWindowAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl SlidingWindowAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.swa.num_heads;
        let head_dim = hidden_size / num_heads;
        let out_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))?;
        let wqkv = linear_no_bias(hidden_size, hidden_size * 3, vb.pp("Wqkv"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        Ok(Self {
            wqkv,
            out_proj,
            hidden_size,
            num_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = xs.apply(&self.wqkv)?;
        let qkv = qkv.reshape((b_sz, q_len, 3, (), self.head_dim))?;

        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let out = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.out_proj)?;

        Ok(out)
    }
}

// The model layers use three types of mixers.
#[derive(Debug, Clone)]
enum SequenceMixer {
    Based(BasedConv),
    Linear(LinearAttention),
    Sliding(SlidingWindowAttention),
}

impl SequenceMixer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Based(b) => b.forward(xs, pos),
            Self::Linear(b) => b.forward(xs, pos),
            Self::Sliding(b) => b.forward(xs, attention_mask, pos),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    mlp: MLP,
    norm1: RmsNorm,
    norm2: RmsNorm,
    mixer: SequenceMixer,
}

impl DecoderLayer {
    fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let norm1 = rms_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm1"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm2"))?;

        let l_attn = cfg.alt_mixer_layers.contains(&layer_idx);
        let sw_attn = cfg.alt_mixer_2_layers.contains(&layer_idx);

        let mixer = if l_attn {
            SequenceMixer::Linear(LinearAttention::new(cfg, vb.pp("mixer"))?)
        } else if sw_attn {
            SequenceMixer::Sliding(SlidingWindowAttention::new(cfg, vb.pp("mixer"))?)
        } else {
            SequenceMixer::Based(BasedConv::new(cfg, vb.pp("mixer"))?)
        };

        Ok(Self {
            mlp,
            norm1,
            norm2,
            mixer,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = self.mixer.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.norm2)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: super::with_tracing::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vocab_size = cfg.vocab_size + (8 - cfg.vocab_size % 8) % 8;
        let lm_head = linear_no_bias(cfg.hidden_size, vocab_size, vb.pp("lm_head"))?;
        let embed_tokens = super::with_tracing::Embedding::from_weights(lm_head.weight().clone())?;
        let vb_m = vb.pp("transformer");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(layer_idx, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb_m.pp("ln_f"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.swa.window_size,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let sliding_window = self.sliding_window / 2;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
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
