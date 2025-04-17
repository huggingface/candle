use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
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

    /// RoPE 적용 (q, k shape: B x H x L x D)
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Qwen3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let var = (xs.clone() * &xs)?.mean_keepdim(D::Minus1)?;
        let rms = (var + self.eps)?.powf(-0.5)?;
        let xs = xs.broadcast_mul(&rms)?;
        let ws = self.weight.reshape((1, -1))?;
        Ok((xs * &ws)?.to_dtype(orig_dtype))
    }
}

#[derive(Debug, Clone)]
struct Qwen3HeadRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3HeadRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), DType::F32)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B*L*H, D)
        let orig_dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let var = (xs.clone() * &xs)?.mean_keepdim(D::Minus1)?;
        let rms = var.add(self.eps)?.powf(-0.5)?;
        let xs = xs.broadcast_mul(&rms)?;
        let ws = self.weight.reshape((1, -1))?;
        Ok((xs * &ws)?.to_dtype(orig_dtype))
    }
}

fn repeat_kv(kv: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    let (b, h_kv, l, d) = kv.dims4()?;
    kv.unsqueeze(2)?
        .expand((b, h_kv, n_rep, l, d))?
        .reshape((b, h_kv * n_rep, l, d))
}

#[derive(Debug, Clone)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Qwen3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // norms
    q_norm: Qwen3HeadRmsNorm,
    k_norm: Qwen3HeadRmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // sliding window
    sliding_window: Option<usize>,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3Attention {
    fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let (q_proj, k_proj, v_proj, o_proj) = if cfg.attention_bias {
            (
                linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
                linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
                linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
                linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            )
        } else {
            (
                linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
                linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
                linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
                linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            )
        };

        let q_norm = Qwen3HeadRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3HeadRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let sliding_window = if cfg.use_sliding_window && layer_idx >= cfg.max_window_layers {
            cfg.sliding_window
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            sliding_window,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per‑head RMSNorm
        let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. KV 캐시 누적
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[prev_k, &k], 2)?,
                Tensor::cat(&[prev_v, &v], 2)?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // 6. GQA repeat_kv
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;

        // 7. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 8. Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: Qwen3RmsNorm,
    ln2: Qwen3RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::new(cfg, rotary, idx, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(cfg, vb.pp("mlp"))?,
            ln1: Qwen3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln1"))?,
            ln2: Qwen3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln2"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = x + h;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        Ok(x + h2)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen3RmsNorm,
    rotary: Arc<Qwen3RotaryEmbedding>,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), i, vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: Qwen3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            rotary,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}
