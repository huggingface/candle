//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm ─────────────────────────────────────────────────────────────────

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
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
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

fn first_kv_shared_layer_idx(cfg: &Gemma4TextConfig) -> usize {
    cfg.num_hidden_layers
        .saturating_sub(cfg.num_kv_shared_layers)
}

fn kv_shared_layer_index(cfg: &Gemma4TextConfig, layer_idx: usize) -> Result<Option<usize>> {
    let first_shared = first_kv_shared_layer_idx(cfg);
    if first_shared == 0 || layer_idx < first_shared {
        return Ok(None);
    }

    let attention_type = cfg.layer_types.get(layer_idx).ok_or_else(|| {
        candle::Error::msg(format!("missing Gemma4 layer type for layer {layer_idx}"))
    })?;

    cfg.layer_types[..first_shared]
        .iter()
        .rposition(|ty| ty == attention_type)
        .map(Some)
        .ok_or_else(|| {
            candle::Error::msg(format!(
                "Gemma4 layer {layer_idx} is configured to share KV without a prior `{attention_type}` donor layer."
            ))
        })
}

fn kv_donor_flags(cfg: &Gemma4TextConfig) -> Result<Vec<bool>> {
    let first_shared = first_kv_shared_layer_idx(cfg);
    let mut donors = vec![false; cfg.num_hidden_layers];
    for shared_idx in first_shared..cfg.num_hidden_layers {
        if let Some(donor_idx) = kv_shared_layer_index(cfg, shared_idx)? {
            donors[donor_idx] = true;
        }
    }
    Ok(donors)
}

fn gemma4_disable_ple() -> bool {
    std::env::var_os("CANDLE_GEMMA4_DISABLE_PLE").is_some()
}

fn gemma4_disable_shared_kv() -> bool {
    std::env::var_os("CANDLE_GEMMA4_DISABLE_SHARED_KV").is_some()
}

fn gemma4_force_f32_attn() -> bool {
    std::env::var_os("CANDLE_GEMMA4_FORCE_F32_ATTN").is_some()
}

fn gemma4_disable_attn_softcap() -> bool {
    std::env::var_os("CANDLE_GEMMA4_DISABLE_ATTN_SOFTCAP").is_some()
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    is_gpt_neox: bool,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        Self::new_with_layout(dtype, head_dim, rope_theta, max_seq_len, dev, true)
    }

    fn new_with_layout(
        _dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            is_gpt_neox,
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
        let rope = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };
        let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    is_gpt_neox: bool,
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
        Self::new_with_layout(
            dtype,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            max_seq_len,
            dev,
            true,
        )
    }

    fn new_with_layout(
        _dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;

        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
        let _ = vec![0f32; half_dim - rope_angles];

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
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
        let rope = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };
        let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
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
    window_size_left: Option<usize>,
    causal: bool,
) -> Result<Tensor> {
    if window_size_left.is_some() {
        return candle_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            softmax_scale,
            window_size_left,
            Some(0),
        );
    }
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: f32,
    _: Option<usize>,
    _: bool,
) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

impl KvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Normal(cache) => cache.append(k, v),
            Self::Rotating(cache) => cache.append(k, v),
        }
    }

    fn k(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal(cache) => cache.k(),
            Self::Rotating(cache) => cache.k(),
        }
    }

    fn v(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Normal(cache) => cache.v(),
            Self::Rotating(cache) => cache.v(),
        }
    }

    #[allow(unused)]
    fn current_seq_len(&self) -> usize {
        match self {
            Self::Normal(cache) => cache.current_seq_len(),
            Self::Rotating(cache) => cache.current_seq_len(),
        }
    }

    fn positions(&self, seq_len: usize) -> Option<Vec<usize>> {
        match self {
            Self::Normal(_) => None,
            Self::Rotating(cache) => Some(cache.positions(seq_len)),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Normal(cache) => cache.reset(),
            Self::Rotating(cache) => cache.reset(),
        }
    }
}

fn sliding_logit_bias_from_positions(
    query_positions: &[usize],
    key_positions: &[usize],
    dtype: DType,
    device: &Device,
    sliding_window: usize,
) -> Result<Tensor> {
    let mask: Vec<_> = query_positions
        .iter()
        .flat_map(|&q_pos| {
            key_positions.iter().map(move |&k_pos| {
                if k_pos > q_pos || k_pos + sliding_window < q_pos {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (query_positions.len(), key_positions.len()), device)?.to_dtype(dtype)
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    sliding_window: Option<usize>,
    attn_logit_softcapping: Option<f64>,
    _query_pre_attn_scalar: usize,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    kv_shared_layer_index: Option<usize>,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        force_full_kv_cache: bool,
        kv_shared_layer_index: Option<usize>,
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
        let k_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let kv_cache = if is_sliding && !force_full_kv_cache && kv_shared_layer_index.is_none() {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.effective_sliding_window() + 1,
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
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            sliding_window: is_sliding.then_some(cfg.effective_sliding_window()),
            attn_logit_softcapping: if gemma4_disable_attn_softcap() {
                None
            } else {
                cfg.attn_logit_softcapping
            },
            _query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            kv_shared_layer_index,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        donor_kv: Option<&(Tensor, Tensor)>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Q/K norms
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        v = v_norm(&v, self.rms_norm_eps)?;

        // Apply RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        };

        let rotating_sliding_prefill = self.is_sliding
            && !self.use_flash_attn
            && donor_kv.is_none()
            && q_len > 1
            && matches!(self.kv_cache, KvCache::Rotating(_));

        let (k, v, explicit_sliding_mask) = if rotating_sliding_prefill {
            let sliding_window = self
                .sliding_window
                .expect("sliding Gemma4 attention should have a sliding window");
            let query_positions: Vec<_> = (seqlen_offset..seqlen_offset + q_len).collect();
            let mut key_positions = self.kv_cache.positions(0).unwrap_or_default();
            let prefix_k = self.kv_cache.k()?;
            let prefix_v = self.kv_cache.v()?;
            key_positions.extend(query_positions.iter().copied());

            let k_for_attn = if let Some(prefix_k) = prefix_k {
                Tensor::cat(&[&prefix_k, &k], 2)?
            } else {
                k.clone()
            };
            let v_for_attn = if let Some(prefix_v) = prefix_v {
                Tensor::cat(&[&prefix_v, &v], 2)?
            } else {
                v.clone()
            };
            let sliding_mask = sliding_logit_bias_from_positions(
                &query_positions,
                &key_positions,
                q.dtype(),
                xs.device(),
                sliding_window,
            )?;
            let _ = self.kv_cache.append(&k, &v)?;
            (k_for_attn, v_for_attn, Some(sliding_mask))
        } else {
            let (k, v) = if let Some((k, v)) = donor_kv {
                (k.clone(), v.clone())
            } else {
                self.kv_cache.append(&k, &v)?
            };
            (k, v, None)
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mask = if let Some(mask) = explicit_sliding_mask {
            Some(mask)
        } else if self.is_sliding {
            sliding_attention_mask.cloned()
        } else {
            attention_mask.cloned()
        };
        let mask = if let Some(mask) = mask {
            let kv_seq_len = k.dims()[2];
            let mask_dims = mask.dims();
            match mask.rank() {
                2 if mask_dims[1] > kv_seq_len => {
                    Some(mask.narrow(1, mask_dims[1] - kv_seq_len, kv_seq_len)?)
                }
                3 if mask_dims[2] > kv_seq_len => {
                    Some(mask.narrow(2, mask_dims[2] - kv_seq_len, kv_seq_len)?)
                }
                4 if mask_dims[3] > kv_seq_len => {
                    Some(mask.narrow(3, mask_dims[3] - kv_seq_len, kv_seq_len)?)
                }
                _ => Some(mask),
            }
        } else if self.is_sliding && !self.use_flash_attn {
            let kv_seq_len = k.dims()[2];
            let sliding_window = self
                .sliding_window
                .expect("sliding Gemma4 attention should have a sliding window");
            let kv_start = seqlen_offset + q_len - kv_seq_len;
            let sliding_mask: Vec<_> = (0..q_len)
                .flat_map(|i| {
                    let q_pos = seqlen_offset + i;
                    (0..kv_seq_len).map(move |j| {
                        let k_pos = kv_start + j;
                        if k_pos > q_pos || k_pos + sliding_window < q_pos {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(Tensor::from_slice(
                &sliding_mask,
                (q_len, kv_seq_len),
                xs.device(),
            )?)
        } else {
            None
        };

        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let scale = 1.0f32;
            flash_attn(&q, &k, &v, scale, self.sliding_window, mask.is_some())?.transpose(1, 2)?
        } else {
            let attn_input_dtype = q.dtype();
            let (q, k, v) = if gemma4_force_f32_attn() {
                (
                    q.to_dtype(DType::F32)?,
                    k.to_dtype(DType::F32)?,
                    v.to_dtype(DType::F32)?,
                )
            } else {
                (q, k, v)
            };
            let scale = 1.0f64;
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = match self.attn_logit_softcapping {
                None => attn_weights,
                Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
            };

            let attn_weights = match mask.as_ref() {
                None => attn_weights,
                Some(mask) if mask.dtype() == attn_weights.dtype() => {
                    attn_weights.broadcast_add(mask)?
                }
                Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;
            if attn_output.dtype() != attn_input_dtype {
                attn_output.to_dtype(attn_input_dtype)?
            } else {
                attn_output
            }
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset()
    }

    fn cached_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        match (self.kv_cache.k()?, self.kv_cache.v()?) {
            (Some(k), Some(v)) => Ok(Some((k, v))),
            (None, None) => Ok(None),
            _ => candle::bail!("Gemma4 KV cache is internally inconsistent"),
        }
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
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_scalar: Option<Tensor>,
    act: Activation,
    #[allow(dead_code)]
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        force_full_kv_cache: bool,
        kv_shared_layer_index: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            force_full_kv_cache,
            kv_shared_layer_index,
            vb.pp("self_attn"),
        )?;
        let uses_double_wide_mlp = if gemma4_disable_shared_kv() {
            layer_idx >= first_kv_shared_layer_idx(cfg)
        } else {
            kv_shared_layer_index.is_some()
        };
        let intermediate_size = if cfg.use_double_wide_mlp && uses_double_wide_mlp {
            cfg.intermediate_size * 2
        } else {
            cfg.intermediate_size
        };
        let mlp = MLP::new(
            cfg.hidden_size,
            intermediate_size,
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
        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(0);
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if ple_dim > 0
        {
            (
                Some(candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    ple_dim,
                    vb.pp("per_layer_input_gate"),
                )?),
                Some(candle_nn::linear_no_bias(
                    ple_dim,
                    cfg.hidden_size,
                    vb.pp("per_layer_projection"),
                )?),
                Some(RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                )?),
            )
        } else {
            (None, None, None)
        };
        let layer_scalar = vb.get(1, "layer_scalar").ok();
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            act: cfg.hidden_activation,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        donor_kv: Option<&(Tensor, Tensor)>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            sliding_attention_mask,
            donor_kv,
            seqlen_offset,
        )?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let mut xs = (residual + xs)?;

        if let (Some(gate), Some(proj), Some(norm), Some(per_layer_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = xs.clone();
            let gated = gate.forward(&xs)?.apply(&self.act)?;
            let gated = (gated * per_layer_input)?;
            let projected = proj.forward(&gated)?;
            let projected = norm.forward(&projected)?;
            xs = (residual + projected)?;
        }

        if let Some(layer_scalar) = &self.layer_scalar {
            xs = xs.broadcast_mul(layer_scalar)?;
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
    let total_kv_len = tgt_len + seqlen_offset;
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                let q_pos = seqlen_offset + i;
                (0..total_kv_len).map(move |j| {
                    if j > q_pos || j + sliding_window < q_pos {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| {
                let q_pos = seqlen_offset + i;
                (0..total_kv_len).map(move |j| if j > q_pos { f32::NEG_INFINITY } else { 0. })
            })
            .collect()
    };
    Tensor::from_slice(&mask, (tgt_len, total_kv_len), device)?
        .expand((b_size, 1, tgt_len, total_kv_len))?
        .to_dtype(dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    num_hidden_layers: usize,
    per_layer_input_scale: f64,
    per_layer_projection_scalar: f64,
    sliding_window: usize,
}

fn find_text_weights_root<'a>(vb: &VarBuilder<'a>, tensor_name: &str) -> Option<VarBuilder<'a>> {
    let direct = vb.clone();
    if direct.contains_tensor(tensor_name) {
        return Some(direct);
    }

    let model = vb.pp("model");
    if model.contains_tensor(tensor_name) {
        return Some(model);
    }

    let language_model = vb.pp("language_model");
    if language_model.contains_tensor(tensor_name) {
        return Some(language_model);
    }

    let model_language_model = model.pp("language_model");
    if model_language_model.contains_tensor(tensor_name) {
        return Some(model_language_model);
    }

    None
}

fn validate_text_config(cfg: &Gemma4TextConfig) -> Result<()> {
    if gemma4_disable_shared_kv() {
        return Ok(());
    }
    for layer_idx in first_kv_shared_layer_idx(cfg)..cfg.num_hidden_layers {
        kv_shared_layer_index(cfg, layer_idx)?;
    }
    Ok(())
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        validate_text_config(cfg)?;

        let vb_m = find_text_weights_root(&vb, "embed_tokens.weight").ok_or_else(|| {
            candle::Error::msg(
                "cannot find Gemma 4 text weights, expected embed_tokens.weight under one of: ., model, language_model, model.language_model",
            )
        })?;

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let ple_dim = if gemma4_disable_ple() {
            0
        } else {
            cfg.hidden_size_per_layer_input.unwrap_or(0)
        };
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if ple_dim > 0 {
                let ple_vocab = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);
                (
                    Some(candle_nn::embedding(
                        ple_vocab,
                        cfg.num_hidden_layers * ple_dim,
                        vb_m.clone()
                            .set_device(Device::Cpu)
                            .pp("embed_tokens_per_layer"),
                    )?),
                    Some(candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        cfg.num_hidden_layers * ple_dim,
                        vb_m.pp("per_layer_model_projection"),
                    )?),
                    Some(RmsNorm::new(
                        ple_dim,
                        cfg.rms_norm_eps,
                        vb_m.pp("per_layer_projection_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
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
        let donor_flags = if gemma4_disable_shared_kv() {
            vec![false; cfg.num_hidden_layers]
        } else {
            kv_donor_flags(cfg)?
        };
        for (layer_idx, _donor_flag) in donor_flags.iter().enumerate().take(cfg.num_hidden_layers) {
            let kv_shared_layer_index = if gemma4_disable_shared_kv() {
                None
            } else {
                kv_shared_layer_index(cfg, layer_idx)?
            };
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                donor_flags[layer_idx],
                kv_shared_layer_index,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            let vb_lm_head =
                find_text_weights_root(&vb, "lm_head.weight").unwrap_or_else(|| vb_m.clone());
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_lm_head.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            hidden_size_per_layer_input: ple_dim,
            num_hidden_layers: cfg.num_hidden_layers,
            per_layer_input_scale: 2f64.powf(-0.5),
            per_layer_projection_scalar: (cfg.hidden_size as f64).powf(-0.5),
            sliding_window: cfg.effective_sliding_window(),
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
        let scale = (self.hidden_size as f64).sqrt();
        xs * scale
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        self.forward_embeds(input_ids, &xs, seqlen_offset, b_size, seq_len)
    }

    fn compute_ple(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        if self.hidden_size_per_layer_input == 0 {
            return Ok(None);
        }

        let ple_emb = self
            .embed_tokens_per_layer
            .as_ref()
            .expect("PLE embedding missing");
        let ple_proj = self
            .per_layer_model_projection
            .as_ref()
            .expect("PLE projection missing");
        let ple_norm = self
            .per_layer_projection_norm
            .as_ref()
            .expect("PLE norm missing");

        let ple_dim = self.hidden_size_per_layer_input;
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let ple_input_ids = if input_ids.device().is_cuda() {
            input_ids.to_device(&Device::Cpu)?
        } else {
            input_ids.clone()
        };
        let embedded = ple_emb.forward(&ple_input_ids)?;
        let embedded = (embedded * (ple_dim as f64).sqrt())?;
        let embedded = embedded.reshape((b_size, seq_len, self.num_hidden_layers, ple_dim))?;
        let embedded = embedded
            .to_device(inputs_embeds.device())?
            .to_dtype(inputs_embeds.dtype())?;

        let projected = ple_proj.forward(inputs_embeds)?;
        let projected = (projected * self.per_layer_projection_scalar)?;
        let projected = projected.reshape((b_size, seq_len, self.num_hidden_layers, ple_dim))?;
        let projected = ple_norm.forward(&projected)?;

        let combined = ((projected + embedded)? * self.per_layer_input_scale)?;
        let combined = combined.transpose(1, 2)?.contiguous()?;
        let mut per_layer_inputs = Vec::with_capacity(self.num_hidden_layers);
        for layer_idx in 0..self.num_hidden_layers {
            per_layer_inputs.push(combined.narrow(1, layer_idx, 1)?.squeeze(1)?);
        }
        Ok(Some(per_layer_inputs))
    }

    pub fn forward_embeds(
        &mut self,
        input_ids: &Tensor,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let per_layer_inputs = self.compute_ple(input_ids, xs)?;
        let mut xs = xs.clone();
        for layer_idx in 0..self.layers.len() {
            let (before, current_and_after) = self.layers.split_at_mut(layer_idx);
            let layer = &mut current_and_after[0];
            let donor_kv = if let Some(donor_idx) = layer.self_attn.kv_shared_layer_index {
                let donor = before.get(donor_idx).ok_or_else(|| {
                    candle::Error::msg(format!(
                        "Gemma4 shared KV layer {layer_idx} cannot find donor layer {donor_idx}"
                    ))
                })?;
                Some(donor.self_attn.cached_kv()?.ok_or_else(|| {
                    candle::Error::msg(format!(
                        "Gemma4 shared KV layer {layer_idx} expected donor layer {donor_idx} cache to be populated"
                    ))
                })?)
            } else {
                None
            };
            let per_layer_input = per_layer_inputs
                .as_ref()
                .and_then(|inputs| inputs.get(layer_idx));
            xs = layer.forward(
                &xs,
                per_layer_input,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                donor_kv.as_ref(),
                seqlen_offset,
            )?;
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => {
                let logits_dtype = logits.dtype();
                let logits = logits.to_dtype(DType::F32)?;
                Ok(((logits / sc)?.tanh()? * sc)?.to_dtype(logits_dtype)?)
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
