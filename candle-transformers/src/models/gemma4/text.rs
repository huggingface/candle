//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, linear_no_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm ─────────────────────────────────────────────────────────────────
//
// Gemma 4 stores the full scale in `weight` (initialized to ones) and multiplies
// by it directly. This is not the Gemma 3 `(1 + weight)` offset.

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
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let weight = self.weight.to_dtype(internal_dtype)?;
        x_normed.broadcast_mul(&weight)?.to_dtype(x_dtype)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
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

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
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

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
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

// ── Flash attention ─────────────────────────────────────────────────────────

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

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    /// Absent on KV-shared layers; those reuse an earlier layer's K/V.
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    is_kv_shared: bool,
    store_full_length_kv: bool,
    sliding_window: usize,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: Option<KvCache>,
    use_flash_attn: bool,
}

/// Full-length K/V published by the last non-shared layer of each attention type.
#[derive(Debug, Clone, Default)]
struct SharedKvStates {
    sliding: Option<(Tensor, Tensor)>,
    full: Option<(Tensor, Tensor)>,
}

impl SharedKvStates {
    fn get(&self, sliding: bool) -> Option<&(Tensor, Tensor)> {
        if sliding {
            self.sliding.as_ref()
        } else {
            self.full.as_ref()
        }
    }

    fn set(&mut self, sliding: bool, kv: (Tensor, Tensor)) {
        if sliding {
            self.sliding = Some(kv);
        } else {
            self.full = Some(kv);
        }
    }
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let is_kv_shared = cfg.is_kv_shared_layer(layer_idx);
        let store_full_length_kv = cfg.stores_full_length_kv(layer_idx);
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        // Shared layers reuse an earlier layer's K/V. Those tensors may still
        // be present in the checkpoint, but Hugging Face ignores them.
        let (k_proj, v_proj, k_norm, kv_cache) = if is_kv_shared {
            (None, None, None, None)
        } else {
            let k_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
            let v_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
            let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            // The donor of each type must keep the full sequence, not a sliding window.
            let kv_cache = if is_sliding && !store_full_length_kv {
                KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                    2,
                    cfg.effective_sliding_window(),
                ))
            } else {
                KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                    2,
                    cfg.max_position_embeddings,
                ))
            };
            (Some(k_proj), Some(v_proj), Some(k_norm), Some(kv_cache))
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
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            is_kv_shared,
            store_full_length_kv,
            sliding_window: cfg.effective_sliding_window(),
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        shared_kv: &mut SharedKvStates,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;
        let (q, _) = if self.is_sliding {
            self.rotary_emb_local
                .apply_rotary_emb_qkv(&q, &q, seqlen_offset)?
        } else {
            self.rotary_emb_global
                .apply_rotary_emb_qkv(&q, &q, seqlen_offset)?
        };

        let (k, v) = if self.is_kv_shared {
            let (k, v) = shared_kv.get(self.is_sliding).ok_or_else(|| {
                candle::Error::Msg("missing shared KV for Gemma 4 layer".into())
            })?;
            (k.clone(), v.clone())
        } else {
            let k_proj = self.k_proj.as_ref().unwrap();
            let v_proj = self.v_proj.as_ref().unwrap();
            let k_norm = self.k_norm.as_ref().unwrap();
            let mut k = k_proj.forward(xs)?;
            let v = v_proj.forward(xs)?;
            k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            k = k_norm.forward(&k)?;
            let v = v_norm(&v, self.rms_norm_eps)?;
            let (_, k) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&k, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&k, &k, seqlen_offset)?
            };
            let (k, v) = match self.kv_cache.as_mut().unwrap() {
                KvCache::Normal(cache) => cache.append(&k, &v)?,
                KvCache::Rotating(cache) => cache.append(&k, &v)?,
            };
            if self.store_full_length_kv {
                shared_kv.set(self.is_sliding, (k.clone(), v.clone()));
            }
            (k, v)
        };

        let (k, v) = if self.is_sliding && q_len == 1 {
            take_last_kv(&k, &v, self.sliding_window)?
        } else {
            (k, v)
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        // Gemma 4 sets attention scaling to 1. Position is already in RoPE.
        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1.0, mask.is_some())?.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match self.kv_cache.as_mut() {
            Some(KvCache::Normal(c)) => c.reset(),
            Some(KvCache::Rotating(c)) => c.reset(),
            None => {}
        }
    }
}

fn take_last_kv(k: &Tensor, v: &Tensor, window: usize) -> Result<(Tensor, Tensor)> {
    let len = k.dim(2)?;
    if len <= window {
        return Ok((k.clone(), v.clone()));
    }
    let start = len - window;
    Ok((k.narrow(2, start, window)?, v.narrow(2, start, window)?))
}

// ── Per-layer embeddings (PLE) ──────────────────────────────────────────────
//
// E2B/E4B feed a small residual into every decoder layer. Larger Gemma 4
// variants set `hidden_size_per_layer_input` to 0 and omit these weights.

#[derive(Debug, Clone)]
struct PerLayerEmbeddings {
    embed_tokens: candle_nn::Embedding,
    model_projection: Linear,
    projection_norm: RmsNorm,
    ple_dim: usize,
    num_layers: usize,
    embed_scale: f64,
    proj_scale: f64,
    combine_scale: f64,
}

impl PerLayerEmbeddings {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let ple_dim = cfg.hidden_size_per_layer_input;
        let num_layers = cfg.num_hidden_layers;
        let packed = num_layers * ple_dim;
        Ok(Self {
            embed_tokens: candle_nn::embedding(
                cfg.vocab_size_per_layer_input,
                packed,
                vb.pp("embed_tokens_per_layer"),
            )?,
            model_projection: linear_no_bias(cfg.hidden_size, packed, vb.pp("per_layer_model_projection"))?,
            projection_norm: RmsNorm::new(ple_dim, cfg.rms_norm_eps, vb.pp("per_layer_projection_norm"))?,
            ple_dim,
            num_layers,
            embed_scale: (ple_dim as f64).sqrt(),
            proj_scale: 1.0 / (cfg.hidden_size as f64).sqrt(),
            combine_scale: 1.0 / 2.0f64.sqrt(),
        })
    }

    /// `[batch, seq, num_layers, ple_dim]`
    fn forward(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let token = (self.embed_tokens.forward(input_ids)? * self.embed_scale)?;
        let token = token.reshape((b_sz, seq_len, self.num_layers, self.ple_dim))?;

        let context = (self.model_projection.forward(inputs_embeds)? * self.proj_scale)?;
        let context = context.reshape((b_sz, seq_len, self.num_layers, self.ple_dim))?;
        let context = self.projection_norm.forward(&context)?;
        (token + context)? * self.combine_scale
    }
}

#[derive(Debug, Clone)]
struct PerLayerInput {
    gate: Linear,
    projection: Linear,
    post_norm: RmsNorm,
    act: Activation,
}

impl PerLayerInput {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let ple_dim = cfg.hidden_size_per_layer_input;
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, ple_dim, vb.pp("per_layer_input_gate"))?,
            projection: linear_no_bias(ple_dim, cfg.hidden_size, vb.pp("per_layer_projection"))?,
            post_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_per_layer_input_norm"))?,
            act: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.gate)?.apply(&self.act)?;
        let xs = (xs * per_layer_input)?;
        let xs = xs.apply(&self.projection)?;
        let xs = self.post_norm.forward(&xs)?;
        residual + xs
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input: Option<PerLayerInput>,
    /// Learned per-layer scale. Absent only on non-HF exports.
    layer_scalar: Option<Tensor>,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_activation,
            false,
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let per_layer_input = if cfg.uses_per_layer_embeddings() {
            Some(PerLayerInput::new(cfg, vb.clone())?)
        } else {
            None
        };
        let layer_scalar = if vb.contains_tensor("layer_scalar") {
            Some(vb.get(1, "layer_scalar")?)
        } else {
            None
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input,
            layer_scalar,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        shared_kv: &mut SharedKvStates,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            shared_kv,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if let (Some(block), Some(per_layer_input)) = (&self.per_layer_input, per_layer_input) {
            xs = block.forward(&xs, per_layer_input)?;
        }
        if let Some(scale) = &self.layer_scalar {
            xs = xs.broadcast_mul(scale)?;
        }
        Ok(xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    per_layer_embeddings: Option<PerLayerEmbeddings>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
}

impl TextModel {
    /// Create a text model from weights rooted at the language-model module.
    ///
    /// For Hugging Face `Gemma4ForConditionalGeneration` checkpoints, pass a
    /// `VarBuilder` already scoped to `model.language_model`.
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let per_layer_embeddings = if cfg.uses_per_layer_embeddings() {
            Some(PerLayerEmbeddings::new(cfg, vb.clone())?)
        } else {
            None
        };

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            per_layer_embeddings,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        self.forward_embeds_with_ids(&xs, Some(input_ids), seqlen_offset, b_size, seq_len)
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward_embeds_with_ids(xs, None, seqlen_offset, batch_size, seq_len)
    }

    /// Decode from embeddings. `input_ids` is required only for models that use
    /// per-layer embeddings (E2B/E4B); other Gemma 4 variants ignore it.
    pub fn forward_embeds_with_ids(
        &mut self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let per_layer_inputs = match (&self.per_layer_embeddings, input_ids) {
            (Some(ple), Some(input_ids)) => Some(ple.forward(input_ids, xs)?),
            (Some(_), None) => {
                candle::bail!("Gemma 4 per-layer embeddings require input_ids")
            }
            (None, _) => None,
        };

        let mut xs = xs.clone();
        let mut shared_kv = SharedKvStates::default();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_ple = match &per_layer_inputs {
                Some(ple) => Some(ple.narrow(2, layer_idx, 1)?.squeeze(2)?),
                None => None,
            };
            xs = layer.forward(
                &xs,
                layer_ple.as_ref(),
                &mut shared_kv,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
            )?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
