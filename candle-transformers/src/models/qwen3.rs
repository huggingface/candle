use crate::{
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, Activation, VarBuilder};
use std::sync::Arc;

#[cfg(feature = "flash-attn")]
use candle_flash_attn;

#[cfg(not(feature = "flash-attn"))]
use candle_nn::attention::{flash_attn, AttnMask};

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
    #[serde(default)]
    pub use_flash_attn: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
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
pub(crate) struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
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
pub(crate) struct Qwen3Attention {
    // projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    // norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    use_flash_attn: bool,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Qwen3Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            candle::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        // dim=2 because we concatenate along the sequence dimension
        // For tensors of shape [batch, heads, seq, head_dim]
        let kv_cache = ConcatKvCache::new(2);

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
            hidden_size,
            use_flash_attn: cfg.use_flash_attn,
            rotary_emb,
            kv_cache,
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

        // 3. Per-head RMSNorm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // 6. Attention dispatch based on device and features
        let on_cpu = x.device().is_cpu();

        if on_cpu {
            if self.use_flash_attn {
                // CPU with flash flag: use optimized CPU flash attention
                self.forward_cpu_flash_attn(&q, &k, &v, offset, b, l)
            } else {
                // CPU without flash flag: use standard matmul (for comparison/testing)
                self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
            }
        } else if self.use_flash_attn {
            // GPU with flash-attn flag: use GPU flash attention
            self.forward_flash_attn(&q, &k, &v, offset, b, l)
        } else {
            // GPU without flash-attn: use standard matmul attention
            self.forward_standard_attn(&q, &k, &v, attn_mask, b, l)
        }
    }

    /// GPU flash attention path (requires flash-attn feature)
    #[cfg(feature = "flash-attn")]
    fn forward_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // Flash attention expects (B, S, H, D) format
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let causal = l > 1;
        let ctx = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?;

        // Output: (B, S, H, D) -> (B, L, hidden_size)
        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    /// Fallback when flash-attn feature not enabled but use_flash_attn was requested
    #[cfg(not(feature = "flash-attn"))]
    fn forward_flash_attn(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _offset: usize,
        _b: usize,
        _l: usize,
    ) -> Result<Tensor> {
        candle::bail!(
            "use_flash_attn=true requires compiling with --features flash-attn. \
             For CPU, omit --use-flash-attn flag to use optimized CPU attention."
        )
    }

    /// CPU flash attention - optimized fused kernel for CPU
    #[cfg(not(feature = "flash-attn"))]
    fn forward_cpu_flash_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // CPU flash attention expects (B, S, H, D) format
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let ctx = match q.dtype() {
            DType::F32 => flash_attn::<f32>(
                &q,
                &k,
                &v,
                scale,
                AttnMask::causal_with_offset(offset),
                None,
                None,
            )?,
            DType::F64 => flash_attn::<f64>(
                &q,
                &k,
                &v,
                scale,
                AttnMask::causal_with_offset(offset),
                None,
                None,
            )?,
            DType::BF16 => {
                let q_f32 = q.to_dtype(DType::F32)?;
                let k_f32 = k.to_dtype(DType::F32)?;
                let v_f32 = v.to_dtype(DType::F32)?;
                let ctx_f32 = flash_attn::<f32>(
                    &q_f32,
                    &k_f32,
                    &v_f32,
                    scale,
                    AttnMask::causal_with_offset(offset),
                    None,
                    None,
                )?;
                ctx_f32.to_dtype(DType::BF16)?
            }
            dtype => candle::bail!("Unsupported dtype for CPU flash attention: {:?}", dtype),
        };

        // Output from CPU flash attention is (B, H, S, D), transpose to (B, S, H, D)
        let ctx = ctx.transpose(1, 2)?;

        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }

    /// Stub for when flash-attn is enabled (CPU path not needed)
    #[cfg(feature = "flash-attn")]
    fn forward_cpu_flash_attn(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _offset: usize,
        _b: usize,
        _l: usize,
    ) -> Result<Tensor> {
        candle::bail!(
            "CPU inference with use_flash_attn=true requires building without --features flash-attn"
        )
    }

    /// Standard matmul-based attention (works on any device)
    fn forward_standard_attn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        // GQA repeat_kv
        let k = repeat_kv(k.clone(), self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v.clone(), self.num_kv_groups)?.contiguous()?;

        // Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
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

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    use_flash_attn: bool,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            use_flash_attn: cfg.use_flash_attn,
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

        // Build causal mask for standard attention path
        // Flash attention (CPU or GPU) handles masking internally
        let needs_mask = !self.use_flash_attn && l > 1;
        let causal = if needs_mask {
            Some(self.causal_mask(b, l, offset, None)?)
        } else {
            None
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
