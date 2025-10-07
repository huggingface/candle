//! GPT-OSS (MoE + sink-attn) inference implementation for Candle.
//!
//! Ported from HuggingFace Transformers' `GptOss` reference (Python).
//! Key differences vs LLaMA/Gemma:
//! - MoE MLP with top-k router (token-wise expert routing, weighted combine)
//! - Per-head `sinks` logits concatenated to attention logits
//! - Layer-wise attention type: {full, sliding}-window causal mask
//!
//! This follows the structure of `gemma3.rs`.

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    // Core sizes
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,

    // Attention
    pub attention_bias: bool,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64, // simple RoPE (θ). If you need YaRN, extend this.

    // Layer types: "full_attention" | "sliding_attention" per layer.
    pub layer_types: Vec<String>,
    // Sliding window extent used when layer_types[i] == "sliding_attention".
    pub sliding_window: usize,

    // MoE parameters
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize, // top-k per token

    // Optional
    pub use_flash_attn: Option<bool>,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}
impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let d = x.dim(D::Minus1)? as f64;
        let x32 = x.to_dtype(internal)?;
        let var = (x32.sqr()?.sum_keepdim(D::Minus1)? / d)?;
        let xhat = x32.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // GPT-OSS norm multiplies by weight (no +1)
        xhat.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}
impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        // inv_freq[i] = 1 / (theta^(2i/d))
        let inv: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_len = inv.len();
        let inv = Tensor::from_vec(inv, (1, inv_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }
    fn apply_qk(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, qlen, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, qlen)?;
        let sin = self.sin.narrow(0, seqlen_offset, qlen)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
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
    rotary: Arc<RotaryEmbedding>,
    sinks: Tensor, // (num_heads,)
    kv_cache: KvCache,
    use_flash_attn: bool,
}
impl Attention {
    fn new(
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        sliding: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nk = cfg.num_key_value_heads;
        let gh = nh / nk;
        let dh = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(h, nh * dh, bias, vb.pp("q_proj"))?;
        let k_proj = linear(h, nk * dh, bias, vb.pp("k_proj"))?;
        let v_proj = linear(h, nk * dh, bias, vb.pp("v_proj"))?;
        let o_proj = linear(nh * dh, h, bias, vb.pp("o_proj"))?;
        let sinks = vb.get(nh, "sinks")?; // nn.Parameter of size (num_heads,)
        let kv_cache = if sliding {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.sliding_window,
            ))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: nh,
            num_kv_heads: nk,
            num_kv_groups: gh,
            head_dim: dh,
            rotary,
            sinks,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn.unwrap_or(false),
        })
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b, qlen, _) = xs.dims3()?;

        // Projections.
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, qlen, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (b, nH, q, d)
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, qlen, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, qlen, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // RoPE.
        let (q, k) = self.rotary.apply_qk(&q, &k, seqlen_offset)?;

        // KV-cache.
        let (k, v) = match &mut self.kv_cache {
            KvCache::Normal(c) => c.append(&k, &v)?,
            KvCache::Rotating(c) => c.append(&k, &v)?,
        };

        // Repeat kv to groups.
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?; // (b, nH, t, d)
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?; // (b, nH, t, d)

        // Flash-attn path.
        if self.use_flash_attn {
            let qf = q.transpose(1, 2)?; // (b, q, h, d)
            let kf = k.transpose(1, 2)?;
            let vf = v.transpose(1, 2)?;
            let scale = 1f32 / (self.head_dim as f32).sqrt();
            // We cannot easily append sinks with flash-attn. Fall back to eager if sinks are required.
            // For strict parity, use eager path when sinks exist.
            // If you still want flash-attn, remove sinks or approximate via bias.
            let attn = flash_attn(&qf, &kf, &vf, scale, true)?.transpose(1, 2)?; // (b, h, q, d)
            return attn
                .transpose(1, 2)?
                .reshape((b, qlen, self.num_heads * self.head_dim))?
                .apply(&self.o_proj);
        }

        // Eager attention with sinks.
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let mut logits = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // (b, h, q, t)

        // Mask (causal/sliding).
        if let Some(m) = attn_mask {
            // m: (b, 1, q, t_total) or broadcastable
            logits = logits.broadcast_add(m)?;
        }

        // Append sinks: concat an extra column per head with `sinks` value (broadcast).
        // sinks: (h,) -> (1,h,1,1) -> expand (b,h,q,1)
        let sinks =
            self.sinks
                .reshape((1, self.num_heads, 1, 1))?
                .expand((b, self.num_heads, qlen, 1))?;
        let combined = Tensor::cat(&[&logits, &sinks], D::Minus1)?; // (b,h,q,t+1)

        // Numerically-stable softmax over last dim.
        let maxv = combined.max_keepdim(D::Minus1)?;
        let probs = (combined - maxv)?.exp()?;
        let z = probs.sum_keepdim(D::Minus1)?;
        let probs = probs.broadcast_div(&z)?; // (b,h,q,t+1)

        // Drop the sink probs and renormalize remaining scores implicitly (by slicing).
        let scores = probs.narrow(D::Minus1, 0, probs.dim(D::Minus1)? - 1)?; // (b,h,q,t)

        let attn = scores.matmul(&v)?; // (b,h,q,d)
        attn.transpose(1, 2)?
            .reshape((b, qlen, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}
#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// -------------------- MoE: Router + Experts --------------------

#[derive(Debug, Clone)]
struct TopKRouter {
    w: Linear, // (hidden_size -> num_experts)
    num_experts: usize,
    top_k: usize,
}
impl TopKRouter {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: linear(
                cfg.hidden_size,
                cfg.num_local_experts,
                true,
                vb.pp("router"),
            )?,
            num_experts: cfg.num_local_experts,
            top_k: cfg.num_experts_per_tok,
        })
    }

    // Returns (router_scores, topk_indices)
    // router_scores: (B*T, E) sparse-softmax (prob on top-k positions, zeros elsewhere)
    // topk_indices : (B*T, K)
    fn route(&self, xs_bt_h: &Tensor) -> Result<(Tensor, Tensor)> {
        // logits: (N, E)
        let logits = self.w.forward(xs_bt_h)?;
        // topk over last dim
        let (top_vals, top_idx) = logits.topk(
            self.top_k,
            D::Minus1,
            true,  /*largest*/
            false, /*sorted*/
        )?;
        // softmax over top-k slice
        let top_probs = candle_nn::ops::softmax_last_dim(&top_vals)?;
        
        // Simplified: return top_probs and top_idx directly
        // The expert routing logic will handle the sparse indexing
        Ok((top_probs, top_idx))
    }
}


#[derive(Debug, Clone)]
struct Experts {
    // Parameters packed per expert to mirror HF layout:
    // gate_up: (E, H, 2*D), gate_up_bias: (E, 2*D)
    // down   : (E, D, H),   down_bias    : (E, H)
    gate_up: Tensor,
    gate_up_bias: Tensor,
    down: Tensor,
    down_bias: Tensor,
    hidden: usize,
    expert_dim: usize,
}
impl Experts {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let e = cfg.num_local_experts;
        let h = cfg.hidden_size;
        let d = cfg.intermediate_size;
        let gate_up = vb.get((e, h, 2 * d), "gate_up_proj")?;
        let gate_up_bias = vb.get((e, 2 * d), "gate_up_proj_bias")?;
        let down = vb.get((e, d, h), "down_proj")?;
        let down_bias = vb.get((e, h), "down_proj_bias")?;
        Ok(Self {
            gate_up,
            gate_up_bias,
            down,
            down_bias,
            hidden: h,
            expert_dim: d,
        })
    }

    // xs: (B,T,H). router_scores: (B*T,K). top_idx: (B*T,K)
    // Returns: (B,T,H)
    fn forward(&self, xs: &Tensor, router_scores: &Tensor, top_idx: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let n = b * t;

        // Flatten tokens: (N,H)
        let x = xs.reshape((n, h))?;

        // We'll process only the tokens that route to each expert, in a lightweight loop over experts hit.
        // Build a mask of (N,K,E) -> which tokens go to which expert; then extract token indices per expert.
        // More efficient impls can precompute per-expert index lists on CPU.

        // For simplicity (and performance good enough in Rust), do a per-expert gather:
        let dev = xs.device();
        let mut out = Tensor::zeros((n, h), xs.dtype(), dev)?;

        // Simplified approach: process each token and its top-k experts
        let (n_, k_) = top_idx.dims2()?;
        debug_assert_eq!(n, n_);
        
        for token_idx in 0..n {
            let x_tok = x.i(token_idx)?; // (H,)
            let mut token_output = Tensor::zeros(h, xs.dtype(), dev)?;
            
            for k in 0..k_ {
                let expert_id = top_idx.i((token_idx, k))?.to_scalar::<i64>()? as usize;
                let expert_weight = router_scores.i((token_idx, k))?;
                
                // Expert computation
                let w = self.gate_up.i(expert_id)?;
                let b = self.gate_up_bias.i(expert_id)?;
                let w_down = self.down.i(expert_id)?;
                let b_down = self.down_bias.i(expert_id)?;
                
                let gate_up = x_tok.matmul(&w)? + b?;
                let gate = gate_up.narrow(D::Minus1, 0, self.expert_dim)?;
                let up = gate_up.narrow(D::Minus1, self.expert_dim, self.expert_dim)?;
                
                // GLU activation
                let alpha = 1.702f64;
                let limit = 7.0f64;
                let gate = gate.clamp(-limit, limit)?;
                let up = up.clamp(-limit, limit)?;
                let glu = (&gate * (&gate * alpha)?.sigmoid()?)?;
                let ff = ((&up + 1.0)? * glu)?;
                
                let expert_out = ff.matmul(&w_down)? + b_down?;
                let weighted_out = expert_out.broadcast_mul(&expert_weight)?;
                
                token_output = (&token_output + &weighted_out)?;
            }
            
            // Update output for this token
            out = out.slice_assign(&[token_idx..token_idx+1], &token_output.unsqueeze(0)?)?;
        }

        out.reshape((b, t, h))
    }
}


// -------------------- Decoder Layer --------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp_router: TopKRouter,
    mlp_experts: Experts,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}
impl DecoderLayer {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        layer_idx: usize,
        rotary: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let attn_type = cfg
            .layer_types
            .get(layer_idx)
            .map(|s| s.as_str())
            .unwrap_or("full_attention");
        let sliding = attn_type == "sliding_attention";
        let self_attn = Attention::new(cfg, rotary.clone(), sliding, vb.pp("self_attn"))?;
        let mlp_router = TopKRouter::new(cfg, vb.pp("mlp.router"))?;
        let mlp_experts = Experts::new(cfg, vb.pp("mlp.experts"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp_router,
            mlp_experts,
            input_layernorm,
            post_attention_layernorm,
            sliding_window: if sliding {
                Some(cfg.sliding_window)
            } else {
                None
            },
        })
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask_full: Option<&Tensor>,
        attn_mask_sliding: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Pre-attn LN + Attn + residual
        let residual = xs;
        let x = self.input_layernorm.forward(xs)?;
        let mask = if self.sliding_window.is_some() {
            attn_mask_sliding
        } else {
            attn_mask_full
        };
        let x = self.self_attn.forward(&x, mask, seqlen_offset)?;
        let x = x + residual?;

        // Post-attn LN + MoE + residual
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        // Router
        let (b, t, h) = x.dims3()?;
        let x_flat = x.reshape((b * t, h))?;
        let (router_scores, top_idx) = self.mlp_router.route(&x_flat)?;
        let x_moe = self.mlp_experts.forward(&x, &router_scores, &top_idx)?;
        x_moe + residual
    }
}

// -------------------- Model --------------------

fn prepare_decoder_attention_mask(
    b: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let mask_vec: Vec<f32> = if let Some(win) = sliding {
        // sliding-window causal: allow j in [i-win, i]
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || (j + win) < i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    } else {
        // standard causal
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect()
    };
    let local = Tensor::from_slice(&mask_vec, (tgt_len, tgt_len), dev)?;
    let full = if seqlen_offset > 0 {
        let left = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, dev)?;
        Tensor::cat(&[&left, &local], D::Minus1)?
    } else {
        local
    };
    full.expand((b, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
}
impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_l.pp(i), i, rotary.clone())?);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    fn build_masks(
        &self,
        b: usize,
        q: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if q <= 1 {
            return Ok((None, None));
        }
        let full =
            prepare_decoder_attention_mask(b, q, seqlen_offset, None, self.dtype, &self.device)?;
        let slide = prepare_decoder_attention_mask(
            b,
            q,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(full), Some(slide)))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b, q) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;
        x = x * (self.hidden_size as f64).sqrt()?;

        let (full_mask, sliding_mask) = self.build_masks(b, q, seqlen_offset)?;

        for layer in self.layers.iter_mut() {
            x = layer.forward(&x, full_mask.as_ref(), sliding_mask.as_ref(), seqlen_offset)?;
        }

        // final norm + tied lm head over last position
        let logits = x
            .narrow(1, q - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_kv_cache()
        }
    }
}
