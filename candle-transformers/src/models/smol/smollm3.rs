use crate::{
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    // Optional fields
    pub attention_bias: Option<bool>,
    pub attention_dropout: Option<f64>,
    pub mlp_bias: Option<bool>,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: Option<bool>,
    pub rope_scaling: Option<serde_json::Value>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub max_window_layers: Option<usize>,
    // SmolLM3-specific: NoPE configuration
    pub no_rope_layers: Option<Vec<usize>>,
    pub no_rope_layer_interval: Option<usize>,
}

impl Config {
    pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
        // Method 1: Explicit array (some model variants may provide this)
        if let Some(ref no_rope_layers) = self.no_rope_layers {
            if layer_idx < no_rope_layers.len() {
                // 0 = skip RoPE (NoPE), 1 = use RoPE
                return no_rope_layers[layer_idx] == 0;
            }
        }

        // Method 2: Interval pattern (SmolLM3-3B uses this)
        // With interval=4: layers 0,1,2 use RoPE; layer 3 skips RoPE (NoPE)
        // Pattern: every 4th layer (3,7,11...) skips RoPE
        if let Some(interval) = self.no_rope_layer_interval {
            return (layer_idx + 1) % interval == 0;
        }

        // Default: use RoPE on all layers (standard Llama behavior)
        false
    }

    /// Calculates head_dim from hidden_size and num_attention_heads
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SmolLM3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl SmolLM3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SmolLM3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl SmolLM3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mlp_bias = cfg.mlp_bias.unwrap_or(false);
        Ok(Self {
            gate_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp_bias,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp_bias,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                mlp_bias,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for SmolLM3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SmolLM3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Option<Arc<SmolLM3RotaryEmbedding>>,
    kv_cache: KvCache,
    // NoPE flag
    skip_rope: bool,
}

impl SmolLM3Attention {
    pub(crate) fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: Option<Arc<SmolLM3RotaryEmbedding>>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let use_sliding_window = cfg.use_sliding_window.unwrap_or(false);
        if use_sliding_window {
            candle::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
        )?;

        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
        )?;

        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            attention_bias,
            vb.pp("o_proj"),
        )?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

        // Check if this layer should skip RoPE (NoPE)
        let skip_rope = cfg.should_skip_rope(layer_idx);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            skip_rope,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
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

        // 3. RoPE - only apply if this layer should use RoPE (not NoPE)
        let (q, k) = if self.skip_rope {
            // NoPE: Skip rotary embeddings, but ensure tensors are contiguous
            (q.contiguous()?, k.contiguous()?)
        } else {
            // Apply RoPE
            if let Some(ref rope) = self.rotary_emb {
                rope.apply(&q, &k, offset)?
            } else {
                (q, k)
            }
        };

        // 4. Accumulate KV cache
        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // 5. GQA repeat_kv
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        // 6. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 7. Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DecoderLayer {
    self_attn: SmolLM3Attention,
    mlp: SmolLM3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary: Option<Arc<SmolLM3RotaryEmbedding>>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = SmolLM3Attention::new(cfg, layer_idx, rotary, vb.pp("self_attn"))?;
        let mlp = SmolLM3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub(crate) embed_tokens: candle_nn::Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        // Only create rotary embedding if at least one layer uses RoPE
        let needs_rope = (0..cfg.num_hidden_layers).any(|i| !cfg.should_skip_rope(i));
        let rotary = if needs_rope {
            Some(Arc::new(SmolLM3RotaryEmbedding::new(
                vb.dtype(),
                cfg,
                vb.device(),
            )?))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
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
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
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
