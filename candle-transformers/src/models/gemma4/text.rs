//! Gemma 4 text decoder.
//!
//! Dense Gemma 4 decoder with per-layer inputs and shared key/value states.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// Gemma 4 uses a direct learned scale, unlike the +1 offset in Gemma 3.

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
        x_normed
            .broadcast_mul(&self.weight.to_dtype(internal_dtype)?)?
            .to_dtype(x_dtype)
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
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
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
        let lhs = xs.apply(&self.gate_proj)?;
        let lhs = lhs
            .to_dtype(DType::F32)?
            .apply(&self.act_fn)?
            .to_dtype(xs.dtype())?;
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

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
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
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    shared: bool,
    k_eq_v: bool,
    use_flash_attn: bool,
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
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let shared = layer_idx >= cfg.num_hidden_layers - cfg.num_kv_shared_layers;
        let k_eq_v = cfg.attention_k_eq_v && !is_sliding;
        let k_proj = if shared {
            None
        } else {
            Some(linear_bias(
                hidden_sz,
                num_kv_heads * head_dim,
                bias,
                vb.pp("k_proj"),
            )?)
        };
        let v_proj = if shared || k_eq_v {
            None
        } else {
            Some(linear_bias(
                hidden_sz,
                num_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
            )?)
        };
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = if shared {
            None
        } else {
            Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?)
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
            rotary_emb_global,
            rotary_emb_local,
            kv_cache: None,
            shared,
            k_eq_v,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
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
        let (k, v) = if self.shared {
            shared_kv
                .cloned()
                .ok_or_else(|| candle::Error::Msg("Gemma4 shared KV producer is missing".into()))?
        } else {
            let raw_k = xs.apply(self.k_proj.as_ref().unwrap())?;
            let raw_v = if self.k_eq_v {
                raw_k.clone()
            } else {
                xs.apply(self.v_proj.as_ref().unwrap())?
            };
            let k = raw_k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k.apply(self.k_norm.as_ref().unwrap())?;
            let v = raw_v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v_norm(&v, self.rms_norm_eps)?;
            let (k, _) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&k, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&k, &k, seqlen_offset)?
            };
            let (k, v) = match &self.kv_cache {
                Some((old_k, old_v)) => {
                    (Tensor::cat(&[old_k, &k], 2)?, Tensor::cat(&[old_v, &v], 2)?)
                }
                None => (k, v),
            };
            self.kv_cache = Some((k.clone(), v.clone()));
            (k, v)
        };
        let cache = (k.clone(), v.clone());
        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            flash_attn(&q, &k, &v, 1., mask.is_some())?.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?;

            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights =
                candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
                    .to_dtype(q.dtype())?;
            attn_weights.matmul(&v)?
        };
        let output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?;
        Ok((output, cache))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
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
    #[allow(dead_code)]
    is_sliding: bool,
    per_layer: Option<(Linear, Linear, RmsNorm, Activation)>,
    layer_scalar: Option<Tensor>,
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
            cfg.intermediate_size
                * if cfg.use_double_wide_mlp
                    && layer_idx >= cfg.num_hidden_layers - cfg.num_kv_shared_layers
                {
                    2
                } else {
                    1
                },
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
        let per_layer = if cfg.hidden_size_per_layer_input > 0 {
            Some((
                candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    cfg.hidden_size_per_layer_input,
                    vb.pp("per_layer_input_gate"),
                )?,
                candle_nn::linear_no_bias(
                    cfg.hidden_size_per_layer_input,
                    cfg.hidden_size,
                    vb.pp("per_layer_projection"),
                )?,
                RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                )?,
                cfg.hidden_activation,
            ))
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
            is_sliding,
            per_layer,
            layer_scalar,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        per_layer_input: Option<&Tensor>,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, kv) = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;
        if let (Some((gate, projection, norm, activation)), Some(input)) =
            (&self.per_layer, per_layer_input)
        {
            let residual = &xs;
            let gated = xs.apply(gate)?;
            let gated = gated
                .to_dtype(DType::F32)?
                .apply(activation)?
                .to_dtype(xs.dtype())?;
            let gated = (gated * input)?;
            xs = (residual + gated.apply(projection)?.apply(norm)?)?;
        }
        if let Some(scalar) = &self.layer_scalar {
            xs = xs.broadcast_mul(scalar)?;
        }
        Ok((xs, kv))
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
    let kv_len = tgt_len + seqlen_offset;
    let mask: Vec<_> = (0..tgt_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                let position = seqlen_offset + i;
                if j > position
                    || sliding_window.is_some_and(|window| j.saturating_add(window) <= position)
                {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (tgt_len, kv_len), device)?;
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    max_positions: usize,
    per_layer: Option<(candle_nn::Embedding, Linear, RmsNorm, usize)>,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.enable_moe_block {
            candle::bail!("Gemma4 MoE text decoder is not implemented");
        }
        if cfg.hidden_size == 0
            || cfg.head_dim == 0
            || !cfg.head_dim.is_multiple_of(2)
            || cfg.global_head_dim == 0
            || !cfg.global_head_dim.is_multiple_of(2)
            || cfg.num_attention_heads == 0
            || cfg.num_key_value_heads == 0
            || !cfg
                .num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads)
            || cfg
                .num_global_key_value_heads
                .is_some_and(|n| n == 0 || !cfg.num_attention_heads.is_multiple_of(n))
            || cfg.num_kv_shared_layers >= cfg.num_hidden_layers && cfg.num_kv_shared_layers != 0
            || cfg.layer_types.len() != cfg.num_hidden_layers
            || cfg.sliding_window == 0
            || cfg.max_position_embeddings == 0
            || !(0.0..=1.0).contains(&cfg.partial_rotary_factor())
        {
            candle::bail!("invalid Gemma4 text configuration");
        }
        let unshared = cfg.num_hidden_layers - cfg.num_kv_shared_layers;
        for layer_type in cfg.layer_types.iter().skip(unshared) {
            if !cfg.layer_types[..unshared].contains(layer_type) {
                candle::bail!("Gemma4 shared layer has no KV producer of type {layer_type}");
            }
        }
        // A standalone causal LM nests the decoder under `model`; the full
        // multimodal checkpoint already passes `model.language_model` here.
        let vb_m = if vb.contains_tensor("embed_tokens.weight") {
            vb.clone()
        } else {
            vb.pp("model")
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_global_base_freq(),
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
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
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let per_layer = if cfg.hidden_size_per_layer_input > 0 {
            let dim = cfg.hidden_size_per_layer_input;
            Some((
                candle_nn::embedding(
                    cfg.vocab_size_per_layer_input,
                    cfg.num_hidden_layers * dim,
                    vb_m.pp("embed_tokens_per_layer"),
                )?,
                candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_hidden_layers * dim,
                    vb_m.pp("per_layer_model_projection"),
                )?,
                RmsNorm::new(dim, cfg.rms_norm_eps, vb_m.pp("per_layer_projection_norm"))?,
                dim,
            ))
        } else {
            None
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            max_positions: cfg.max_position_embeddings,
            per_layer,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 && seqlen_offset < self.sliding_window {
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
        self.forward_embeds_inner(&xs, Some(input_ids), seqlen_offset, b_size, seq_len)
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward_embeds_inner(xs, None, seqlen_offset, batch_size, seq_len)
    }

    /// Multimodal callers supply the token identities used by per-layer inputs
    /// separately from the audio/image-injected main embeddings.
    pub fn forward_embeds_with_tokens(
        &mut self,
        xs: &Tensor,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        self.forward_embeds_inner(xs, Some(input_ids), seqlen_offset, batch, seq_len)
    }

    fn forward_embeds_inner(
        &mut self,
        xs: &Tensor,
        input_ids: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if seq_len == 0
            || seqlen_offset
                .checked_add(seq_len)
                .is_none_or(|n| n > self.max_positions)
        {
            candle::bail!("Gemma4 text context limit exceeded or input is empty");
        }
        let per_layer_inputs = match &self.per_layer {
            Some((embeddings, projection, norm, dim)) => {
                let shape = (batch_size, seq_len, self.layers.len(), *dim);
                let context = (xs.apply(projection)? / (self.hidden_size as f64).sqrt())?
                    .reshape(shape)?
                    .apply(norm)?;
                Some(match input_ids {
                    Some(ids) => {
                        let identity =
                            (ids.apply(embeddings)? * (*dim as f64).sqrt())?.reshape(shape)?;
                        ((context + identity)? / 2f64.sqrt())?
                    }
                    None => context,
                })
            }
            None => None,
        };
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let mut xs = xs.clone();
        let mut shared: [Option<(Tensor, Tensor)>; 2] = [None, None];
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let input = per_layer_inputs
                .as_ref()
                .map(|p| p.i((.., .., idx, ..)))
                .transpose()?;
            let kind = usize::from(layer.is_sliding);
            let (output, kv) = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                input.as_ref(),
                shared[kind].as_ref(),
            )?;
            shared[kind] = Some(kv);
            xs = output;
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
