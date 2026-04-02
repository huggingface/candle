//! Standalone Llama-style causal decoder for Granite-Docling.
//!
//! Self-contained implementation (GQA, RoPE, SiLU-gated MLP, RMSNorm, KV cache)
//! to allow future CPU flash attention and interlaced weight optimizations
//! without coupling to the general-purpose Llama model.

use super::config::TextConfig;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias as linear, embedding, Linear, VarBuilder};

// ---------------------------------------------------------------------------
// RMS Norm
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
        let positions = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?;
        // (max_seq, head_dim/2)
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    fn new() -> Self {
        Self { k: None, v: None }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (self.k.as_ref(), self.v.as_ref()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, k], 2)?;
                let v = Tensor::cat(&[prev_v, v], 2)?;
                (k, v)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    fn current_len(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dim(2).unwrap_or(0))
    }

    fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ---------------------------------------------------------------------------
// Attention (Grouped Query Attention)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: KvCache,
}

impl Attention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        let q_proj = linear(h, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(h, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(h, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, h, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: KvCache::new(),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let offset = self.kv_cache.current_len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads to match Q heads
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.t()?)? / scale;
        let attn = match attention_mask {
            Some(mask) => attn?.broadcast_add(mask)?,
            None => attn?,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // (B, heads, seq, head_dim) -> (B, seq, hidden)
        out.transpose(1, 2)?
            .reshape((b, seq_len, ()))?
            .apply(&self.o_proj)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// Feed-forward (SiLU-gated MLP)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear(h, i, vb.pp("gate_proj"))?;
        let up_proj = linear(h, i, vb.pp("up_proj"))?;
        let down_proj = linear(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
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
        rotary: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ---------------------------------------------------------------------------
// Text Model (full decoder)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    rotary: RotaryEmbedding,
    dtype: DType,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?)
        };

        let rotary = RotaryEmbedding::new(cfg, dtype, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            dtype,
        })
    }

    pub fn embed_tokens(&self) -> &candle_nn::Embedding {
        &self.embed_tokens
    }

    fn causal_mask(&self, seq_len: usize, past_kv_len: usize, device: &Device) -> Result<Tensor> {
        let total_len = past_kv_len + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > past_kv_len + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?.to_dtype(self.dtype)
    }

    /// Forward pass from token IDs. Returns logits (B, seq, vocab).
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        self.forward_embeds(&xs)
    }

    /// Forward pass from already-embedded inputs. Returns logits (B, seq, vocab).
    pub fn forward_embeds(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let past_kv_len = self.layers[0].self_attn.kv_cache.current_len();

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, past_kv_len, xs.device())?)
        };

        let mut hidden = xs.clone();
        for layer in self.layers.iter_mut() {
            hidden = layer.forward(&hidden, &self.rotary, mask.as_ref())?;
        }

        let hidden = self.norm.forward(&hidden)?;

        match &self.lm_head {
            Some(lm_head) => hidden.apply(lm_head),
            None => {
                // Tied embeddings: use embed_tokens weight as lm_head
                let w = self.embed_tokens.embeddings();
                hidden.broadcast_matmul(&w.t()?)
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.self_attn.kv_cache.clear();
        }
    }
}
