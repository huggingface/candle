//! Condition encoders for ACE-Step 1.5.
//!
//! This module implements the conditioning pipeline that processes text descriptions,
//! lyrics, and timbre (reference audio) into condition tensors consumed by the DiT
//! diffusion model.
//!
//! The main entry point is [`ConditionEncoder`], which orchestrates:
//! - A simple linear projection for text encoder hidden states.
//! - A [`LyricEncoder`] (8-layer bidirectional transformer) for lyric embeddings.
//! - A [`TimbreEncoder`] (4-layer bidirectional transformer with CLS pooling) for
//!   reference audio embeddings.
//!
//! All encoder layers share the same architecture: pre-norm (RmsNorm) bidirectional
//! GQA attention followed by a gated SiLU MLP, matching the Qwen3 block design.

use crate::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use super::AceStepConfig;

// ---------------------------------------------------------------------------
// RotaryEmbedding (standalone for encoder, same algorithm as the DiT variant)
// ---------------------------------------------------------------------------

/// Precomputed sin/cos tables for rotary position embeddings.
///
/// The encoder uses its own instance because it may have a different head dimension
/// or maximum sequence length than the DiT backbone.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Build the sin/cos lookup tables up to `max_seq_len` positions.
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
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
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE to Q and K tensors of shape `(B, H, L, D)`.
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// EncoderMLP
// ---------------------------------------------------------------------------

/// Gated SiLU MLP identical to the Qwen3 feed-forward block.
///
/// ```text
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
/// ```
#[derive(Debug, Clone)]
pub struct EncoderMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl EncoderMLP {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.encoder_hidden_size();
        let intermediate = cfg.encoder_intermediate_size();
        Ok(Self {
            gate_proj: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for EncoderMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = x.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// EncoderAttention
// ---------------------------------------------------------------------------

/// Bidirectional grouped-query attention for encoder layers.
///
/// Uses per-head RmsNorm on Q and K (Qwen3 style) and rotary position
/// embeddings. No KV cache is needed because the encoder runs a single
/// forward pass over the full sequence.
#[derive(Debug, Clone)]
pub struct EncoderAttention {
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
    hidden_size: usize,
}

impl EncoderAttention {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.encoder_num_attention_heads();
        let num_kv_heads = cfg.encoder_num_key_value_heads();
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden = cfg.encoder_hidden_size();

        let q_proj = linear_b(
            hidden,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            hidden,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            hidden,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * num_heads;

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
        })
    }

    /// Run bidirectional self-attention.
    ///
    /// * `x` - input tensor of shape `(B, L, D)`.
    /// * `mask` - optional additive attention mask `(B, 1, L, L)` where `-inf`
    ///   blocks attention. Pass `None` for fully bidirectional attention.
    /// * `rotary` - precomputed rotary embedding tables.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Project
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RmsNorm
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
            b,
            self.num_heads,
            l,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.num_kv_heads,
            l,
            self.head_dim,
        ))?;

        // Rotary position embeddings
        let (q, k) = rotary.apply(&q, &k, 0)?;

        // GQA key/value expansion
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // Reshape back and output projection
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// EncoderLayer
// ---------------------------------------------------------------------------

/// A single bidirectional pre-norm transformer layer.
///
/// ```text
/// x = x + self_attn(input_layernorm(x))
/// x = x + mlp(post_attention_layernorm(x))
/// ```
///
/// Layers alternate between sliding-window and full attention based on
/// `attention_type`, which controls the mask passed to the attention block.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    self_attn: EncoderAttention,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: EncoderMLP,
    attention_type: String,
}

impl EncoderLayer {
    pub fn new(cfg: &AceStepConfig, attention_type: &str, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.encoder_hidden_size();
        Ok(Self {
            self_attn: EncoderAttention::new(cfg, vb.pp("self_attn"))?,
            input_layernorm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: EncoderMLP::new(cfg, vb.pp("mlp"))?,
            attention_type: attention_type.to_string(),
        })
    }

    /// Returns the attention type for this layer (`"sliding_attention"` or
    /// `"full_attention"`), used to select the appropriate mask.
    pub fn attention_type(&self) -> &str {
        &self.attention_type
    }

    /// Forward pass with pre-norm residual connections.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask, rotary)?;
        let x = (residual + h)?;

        // MLP with residual
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// Bidirectional attention mask helpers
// ---------------------------------------------------------------------------

/// Build a bidirectional sliding-window mask.
///
/// Positions outside the window receive `-inf`; positions inside receive `0`.
/// The returned tensor has shape `(1, 1, seq_len, seq_len)`.
fn sliding_window_mask(
    seq_len: usize,
    window: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (i as i64 - j as i64).unsigned_abs() as usize <= window {
                    0.0
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), dev)?.to_dtype(dtype)
}

/// Build an attention mask from a padding mask, optionally combined with a
/// sliding window constraint.
///
/// * `attention_mask` - `(B, L)` with `1` for valid tokens, `0` for padding.
/// * `sliding_window` - if `Some(w)`, restricts attention to a local window.
///
/// Returns `(B, 1, L, L)` additive mask.
fn build_bidirectional_mask(
    attention_mask: &Tensor,
    seq_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let (b, _) = attention_mask.dims2()?;

    // Padding mask: convert (B, L) 1/0 mask → (B, 1, 1, L) with 0.0 for valid, -inf for padding.
    let pad_mask = attention_mask
        .reshape((b, 1, 1, seq_len))?
        .to_dtype(DType::F32)?;
    // Use where_cond to avoid 0 * -inf = NaN
    let zeros = Tensor::zeros_like(&pad_mask)?;
    let neginf = Tensor::full(f32::NEG_INFINITY, pad_mask.shape(), dev)?.to_dtype(DType::F32)?;
    let pad_mask_bool = pad_mask.gt(&zeros)?; // true for valid (mask==1)
    let pad_mask = pad_mask_bool.where_cond(&zeros, &neginf)?; // 0.0 for valid, -inf for padding

    let mask = match sliding_window {
        Some(w) => {
            let sw = sliding_window_mask(seq_len, w, DType::F32, dev)?;
            // Combine: both must allow the position
            pad_mask.broadcast_add(&sw)?
        }
        None => pad_mask
            .broadcast_as((b, 1, seq_len, seq_len))?
            .contiguous()?,
    };
    mask.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// LyricEncoder
// ---------------------------------------------------------------------------

/// Bidirectional transformer encoder for lyric embeddings.
///
/// Takes pre-embedded lyric token features (from a text encoder) and refines
/// them through 8 layers of bidirectional self-attention.
///
/// Weight prefix: `lyric_encoder.*`
#[derive(Debug, Clone)]
pub struct LyricEncoder {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    sliding_window: Option<usize>,
    dtype: DType,
    device: Device,
}

impl LyricEncoder {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hidden = cfg.encoder_hidden_size();

        // Linear projection from text embedding dim to encoder hidden size (with bias).
        let embed_tokens = linear_b(cfg.text_hidden_dim, enc_hidden, true, vb.pp("embed_tokens"))?;

        let num_layers = cfg.num_lyric_encoder_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..num_layers {
            let attn_type = if i % 2 == 0 {
                "sliding_attention"
            } else {
                "full_attention"
            };
            layers.push(EncoderLayer::new(cfg, attn_type, vb_l.pp(i))?);
        }

        let norm = RmsNorm::new(enc_hidden, cfg.rms_norm_eps, vb.pp("norm"))?;

        let rotary_emb = RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;

        let sliding_window = if cfg.use_sliding_window {
            Some(cfg.sliding_window)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            sliding_window,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode lyric embeddings.
    ///
    /// * `inputs_embeds` - `(B, T, text_hidden_dim)` pre-embedded lyric tokens.
    /// * `attention_mask` - `(B, T)` with `1` for valid tokens, `0` for padding.
    ///
    /// Returns `(B, T, encoder_hidden_size)`.
    pub fn forward(&self, inputs_embeds: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = inputs_embeds.dims3()?;

        let mut hidden = self.embed_tokens.forward(inputs_embeds)?;

        for layer in &self.layers {
            let sw = match layer.attention_type() {
                "sliding_attention" => self.sliding_window,
                _ => None,
            };
            let mask =
                build_bidirectional_mask(attention_mask, seq_len, sw, self.dtype, &self.device)?;
            hidden = layer.forward(&hidden, Some(&mask), &self.rotary_emb)?;
        }

        self.norm.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// TimbreEncoder
// ---------------------------------------------------------------------------

/// Bidirectional transformer encoder for timbre (reference audio) embeddings.
///
/// Uses a learned CLS token prepended to the input sequence. After processing,
/// the CLS token representation (position 0) is extracted as the timbre summary.
///
/// Weight prefix: `timbre_encoder.*`
#[derive(Debug, Clone)]
pub struct TimbreEncoder {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    /// CLS token prepended to timbre sequence (XL models only).
    special_token: Tensor,
    /// Whether to prepend the CLS special_token (true for XL, false for 2B).
    use_cls: bool,
    sliding_window: Option<usize>,
    dtype: DType,
    device: Device,
}

impl TimbreEncoder {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hidden = cfg.encoder_hidden_size();

        // Linear projection from timbre embedding dim to encoder hidden size (with bias).
        let embed_tokens = linear_b(
            cfg.timbre_hidden_dim,
            enc_hidden,
            true,
            vb.pp("embed_tokens"),
        )?;

        let num_layers = cfg.num_timbre_encoder_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..num_layers {
            let attn_type = if i % 2 == 0 {
                "sliding_attention"
            } else {
                "full_attention"
            };
            layers.push(EncoderLayer::new(cfg, attn_type, vb_l.pp(i))?);
        }

        let norm = RmsNorm::new(enc_hidden, cfg.rms_norm_eps, vb.pp("norm"))?;

        // XL models prepend a CLS special token to the timbre sequence.
        // Base/2B models have this commented out in Python.
        let special_token = vb.get((1, 1, enc_hidden), "special_token")?;

        let rotary_emb = RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;

        let sliding_window = if cfg.use_sliding_window {
            Some(cfg.sliding_window)
        } else {
            None
        };

        // XL models have encoder_hidden_size != hidden_size and use the CLS token.
        let use_cls = cfg.encoder_hidden_size.is_some();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            special_token,
            use_cls,
            sliding_window,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Unpack packed timbre embeddings into per-batch format.
    ///
    /// Matches the Python `unpack_timbre_embeddings` exactly:
    /// * `timbre_embs_packed` - `(N, D)` embeddings from each reference audio segment.
    /// * `refer_audio_order_mask` - `(N,)` 1D integer tensor mapping each segment
    ///   to its batch index (0..B-1).
    ///
    /// Returns `(timbre_embs, timbre_mask)`:
    /// - `timbre_embs`: `(B, max_count, D)` gathered embeddings.
    /// - `timbre_mask`: `(B, max_count)` integer mask (1=valid, 0=padding).
    fn unpack_timbre_embeddings(
        &self,
        timbre_embs_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (n, d) = timbre_embs_packed.dims2()?;
        let mask_1d = refer_audio_order_mask.flatten_all()?;

        // B = max(order_mask) + 1
        let b = (mask_1d.max(0)?.to_scalar::<i64>()? + 1) as usize;

        // Count elements per batch and find max_count
        let mask_vec = mask_1d.to_vec1::<i64>()?;
        let mut counts = vec![0usize; b];
        for &idx in &mask_vec {
            counts[idx as usize] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&1);

        // Build the output by placing each embedding at the right (batch, position)
        let mut position_in_batch = vec![0usize; b];
        let mut result = vec![0f32; b * max_count * d];
        let mut mask_out = vec![0i64; b * max_count];

        let packed_data = timbre_embs_packed.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for i in 0..n {
            let batch_idx = mask_vec[i] as usize;
            let pos = position_in_batch[batch_idx];
            position_in_batch[batch_idx] += 1;
            let dst_offset = (batch_idx * max_count + pos) * d;
            result[dst_offset..dst_offset + d].copy_from_slice(&packed_data[i]);
            mask_out[batch_idx * max_count + pos] = 1;
        }

        let timbre_embs =
            Tensor::from_vec(result, (b, max_count, d), &self.device)?.to_dtype(self.dtype)?;
        let timbre_mask =
            Tensor::from_vec(mask_out, (b, max_count), &self.device)?.to_dtype(self.dtype)?;

        Ok((timbre_embs, timbre_mask))
    }

    /// Encode timbre reference audio segments.
    ///
    /// * `refer_audio_packed` - `(N, T, timbre_hidden_dim)` packed reference
    ///   audio features for all segments across the batch.
    /// * `refer_audio_order_mask` - `(N,)` 1D integer tensor mapping each
    ///   segment to its batch index (0..B-1).
    ///
    /// Returns `(timbre_embs, timbre_mask)`:
    /// - `timbre_embs`: `(B, max_count, encoder_hidden_size)`.
    /// - `timbre_mask`: `(B, max_count)` integer mask.
    pub fn forward(
        &self,
        refer_audio_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (n, _t, _) = refer_audio_packed.dims3()?;

        // Project input features: (N, T, timbre_dim) -> (N, T, hidden_size)
        let mut x = self.embed_tokens.forward(refer_audio_packed)?;

        // XL models prepend a CLS special_token for timbre aggregation.
        if self.use_cls {
            let cls = self
                .special_token
                .broadcast_as((n, 1, x.dim(2)?))?
                .contiguous()?;
            x = Tensor::cat(&[&cls, &x], 1)?;
        }

        let seq_len = x.dim(1)?;

        // Build attention mask
        let ones = Tensor::ones((n, seq_len), self.dtype, &self.device)?;

        for layer in &self.layers {
            let sw = match layer.attention_type() {
                "sliding_attention" => self.sliding_window,
                _ => None,
            };
            let mask = build_bidirectional_mask(&ones, seq_len, sw, self.dtype, &self.device)?;
            x = layer.forward(&x, Some(&mask), &self.rotary_emb)?;
        }

        x = self.norm.forward(&x)?;

        // Extract position 0 as the timbre summary: (N, T, D) -> (N, D)
        let embeds = x.narrow(1, 0, 1)?.squeeze(1)?;

        self.unpack_timbre_embeddings(&embeds, refer_audio_order_mask)
    }
}

// ---------------------------------------------------------------------------
// pack_sequences
// ---------------------------------------------------------------------------

/// Pack two (hidden, mask) pairs: concatenate, then reorder so valid tokens
/// (mask=1) come before padding tokens (mask=0) in each batch element.
///
/// Uses a CPU-side stable gather (valid tokens first in original order, then
/// padding tokens) to match Python's `argsort(descending=True, stable=True)`.
/// Length computation uses I64 to avoid BF16 precision loss for long sequences.
///
/// * `hidden1` - `(B, L1, D)`
/// * `hidden2` - `(B, L2, D)`
/// * `mask1` - `(B, L1)` float with 1.0=valid, 0.0=padding
/// * `mask2` - `(B, L2)` float
///
/// Returns `(hidden, mask)` with shapes `(B, L1+L2, D)` and `(B, L1+L2)`.
pub fn pack_sequences(
    hidden1: &Tensor,
    hidden2: &Tensor,
    mask1: &Tensor,
    mask2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let hidden_cat = Tensor::cat(&[hidden1, hidden2], 1)?;
    let mask_cat = Tensor::cat(&[mask1, mask2], 1)?;

    let (b, l, d) = hidden_cat.dims3()?;

    // Build stable sort indices on CPU: valid positions first (in order),
    // then padding positions (in order). This is equivalent to Python's
    // stable descending argsort on a binary mask.
    let mask_f32 = mask_cat
        .to_dtype(candle::DType::F32)?
        .to_device(&candle::Device::Cpu)?;
    let mask_data = mask_f32.to_vec2::<f32>()?;

    let mut sort_indices = vec![0u32; b * l];
    let mut valid_counts = vec![0usize; b];
    for bi in 0..b {
        let mut valid = Vec::new();
        let mut invalid = Vec::new();
        for (j, &val) in mask_data[bi].iter().enumerate().take(l) {
            if val > 0.5 {
                valid.push(j as u32);
            } else {
                invalid.push(j as u32);
            }
        }
        valid_counts[bi] = valid.len();
        let offset = bi * l;
        for (k, &idx) in valid.iter().chain(invalid.iter()).enumerate() {
            sort_indices[offset + k] = idx;
        }
    }

    let sort_idx = Tensor::from_vec(sort_indices, (b, l), hidden_cat.device())?;
    let idx_expanded = sort_idx
        .unsqueeze(2)?
        .broadcast_as((b, l, d))?
        .contiguous()?;
    let hidden_sorted = hidden_cat.gather(&idx_expanded, 1)?;

    // Build new mask using integer counts (avoids BF16 precision loss)
    let mut new_mask_data = vec![0f32; b * l];
    for bi in 0..b {
        for j in 0..valid_counts[bi] {
            new_mask_data[bi * l + j] = 1.0;
        }
    }
    let new_mask =
        Tensor::from_vec(new_mask_data, (b, l), hidden_cat.device())?.to_dtype(mask1.dtype())?;

    Ok((hidden_sorted, new_mask))
}

// ---------------------------------------------------------------------------
// ConditionEncoder
// ---------------------------------------------------------------------------

/// Top-level condition encoder that produces the cross-attention context for
/// the DiT diffusion backbone.
///
/// Combines three conditioning signals:
/// 1. **Text** - style/prompt hidden states projected to encoder dimension.
/// 2. **Lyrics** - lyric token embeddings processed by a bidirectional
///    transformer.
/// 3. **Timbre** - reference audio segments encoded with CLS-token pooling.
///
/// The outputs are packed into a single sequence with an accompanying mask.
///
/// Weight prefix: `encoder.*`
#[derive(Debug, Clone)]
pub struct ConditionEncoder {
    text_projector: Linear,
    lyric_encoder: LyricEncoder,
    timbre_encoder: TimbreEncoder,
}

impl ConditionEncoder {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hidden = cfg.encoder_hidden_size();

        // Text projector: no bias
        let text_projector =
            linear_no_bias(cfg.text_hidden_dim, enc_hidden, vb.pp("text_projector"))?;

        let lyric_encoder = LyricEncoder::new(cfg, vb.pp("lyric_encoder"))?;
        let timbre_encoder = TimbreEncoder::new(cfg, vb.pp("timbre_encoder"))?;

        Ok(Self {
            text_projector,
            lyric_encoder,
            timbre_encoder,
        })
    }

    /// Encode all conditioning inputs into a single packed sequence.
    ///
    /// * `text_hidden_states` - `(B, T_text, text_hidden_dim)` from the text encoder.
    /// * `text_attention_mask` - `(B, T_text)` mask for text tokens.
    /// * `lyric_hidden_states` - `(B, T_lyric, text_hidden_dim)` pre-embedded lyrics.
    /// * `lyric_attention_mask` - `(B, T_lyric)` mask for lyric tokens.
    /// * `refer_audio_packed` - `(N, T_audio, timbre_hidden_dim)` packed reference
    ///   audio features.
    /// * `refer_audio_order_mask` - `(N,)` 1D integer tensor mapping each
    ///   segment to its batch index.
    ///
    /// Returns `(enc_hidden, enc_mask)`:
    /// - `enc_hidden`: `(B, T_lyric + max_ref + T_text, encoder_hidden_size)`.
    /// - `enc_mask`: `(B, T_lyric + max_ref + T_text)` float mask.
    pub fn forward(
        &self,
        text_hidden_states: &Tensor,
        text_attention_mask: &Tensor,
        lyric_hidden_states: &Tensor,
        lyric_attention_mask: &Tensor,
        refer_audio_packed: &Tensor,
        refer_audio_order_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Project text hidden states
        let text = self.text_projector.forward(text_hidden_states)?;

        // Encode lyrics
        let lyric = self
            .lyric_encoder
            .forward(lyric_hidden_states, lyric_attention_mask)?;

        // Encode timbre
        let (timbre, timbre_mask) = self
            .timbre_encoder
            .forward(refer_audio_packed, refer_audio_order_mask)?;

        // Pack lyric + timbre
        let (enc_hidden, enc_mask) =
            pack_sequences(&lyric, &timbre, lyric_attention_mask, &timbre_mask)?;

        // Pack (lyric+timbre) + text
        let (enc_hidden, enc_mask) =
            pack_sequences(&enc_hidden, &text, &enc_mask, text_attention_mask)?;

        Ok((enc_hidden, enc_mask))
    }
}
