//! Diffusion Transformer (DiT) for the ACE-Step 1.5 music generation model.
//!
//! This module implements the core denoising transformer that operates on audio
//! latents produced by the Oobleck VAE. It uses AdaLN-Zero modulation for
//! timestep conditioning and cross-attention for text/lyric conditioning.
//!
//! The architecture features:
//! - Patchified input via strided Conv1d / ConvTranspose1d
//! - Grouped-query attention (GQA) with QK-norm and RoPE
//! - Alternating sliding-window and full self-attention layers
//! - Cross-attention to text encoder outputs
//! - SiLU-gated MLP (same structure as Qwen3)
//! - AdaLN-Zero modulation from timestep embeddings
//!
//! Reference: <https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B>

use crate::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder,
};

use super::AceStepConfig;

// ---------------------------------------------------------------------------
// Rotary positional embedding
// ---------------------------------------------------------------------------

/// Precomputed sin/cos tables for rotary positional embeddings.
///
/// Identical to the Qwen3 implementation: positions are mapped through
/// inverse-frequency bands derived from `rope_theta`, producing a pair of
/// (sin, cos) tensors of shape `(max_position_embeddings, head_dim/2)`.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Creates a new rotary embedding table.
    ///
    /// # Arguments
    /// * `dtype` - Target dtype for the sin/cos tables.
    /// * `cfg` - Model configuration (uses `head_dim`, `max_position_embeddings`, `rope_theta`).
    /// * `dev` - Device on which to allocate the tensors.
    pub fn new(dtype: DType, cfg: &AceStepConfig, dev: &Device) -> Result<Self> {
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

    /// Applies rotary embeddings to query and key tensors.
    ///
    /// Both `q` and `k` must have shape `(B, H, L, D)`. The `offset` parameter
    /// allows indexing into the precomputed tables for incremental decoding
    /// (unused during diffusion but kept for API consistency).
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
// Timestep embedding
// ---------------------------------------------------------------------------

/// Converts a scalar diffusion timestep into a high-dimensional embedding.
///
/// The pipeline is: sinusoidal frequency encoding -> Linear+SiLU -> Linear
/// producing a per-sample embedding `temb` of shape `(B, hidden_size)`, plus
/// a 6-way projection `timestep_proj` of shape `(B, 6, hidden_size)` used for
/// AdaLN-Zero modulation in each transformer layer.
#[derive(Debug, Clone)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
    time_proj: Linear,
    in_channels: usize,
    scale: f64,
}

impl TimestepEmbedding {
    /// Creates a new timestep embedding module.
    ///
    /// # Arguments
    /// * `hidden_size` - Output dimension for the embedding and the 6-way projection.
    /// * `vb` - Variable builder scoped to the embedding prefix (e.g. `time_embed`).
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let in_channels = 256;
        let linear_1 = linear_b(in_channels, hidden_size, true, vb.pp("linear_1"))?;
        let linear_2 = linear_b(hidden_size, hidden_size, true, vb.pp("linear_2"))?;
        let time_proj = linear_b(hidden_size, hidden_size * 6, true, vb.pp("time_proj"))?;
        Ok(Self {
            linear_1,
            linear_2,
            time_proj,
            in_channels,
            scale: 1000.0,
        })
    }

    /// Produces sinusoidal frequency features from a 1-D timestep tensor.
    ///
    /// Input `t` has shape `(B,)`. Output has shape `(B, in_channels)`.
    fn timestep_encoding(&self, t: &Tensor, device: &Device) -> Result<Tensor> {
        let t = (t * self.scale)?;
        let half = self.in_channels / 2;
        let max_period: f64 = 10_000.0;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half), device)?.to_dtype(t.dtype())?;
        let t = t.unsqueeze(D::Minus1)?; // (B, 1)
        let args = t.broadcast_mul(&freqs)?; // (B, half)
        let cos = args.cos()?;
        let sin = args.sin()?;
        Tensor::cat(&[&cos, &sin], D::Minus1) // (B, in_channels)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `t` - Diffusion timestep tensor of shape `(B,)`.
    ///
    /// # Returns
    /// * `temb` - Embedding of shape `(B, hidden_size)`.
    /// * `timestep_proj` - Modulation signals of shape `(B, 6, hidden_size)`.
    pub fn forward(&self, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = t.device();
        let t_freq = self.timestep_encoding(t, device)?;
        let temb = t_freq.apply(&self.linear_1)?.silu()?;
        let temb = temb.apply(&self.linear_2)?;
        let hidden_size = temb.dim(D::Minus1)?;
        let timestep_proj = temb.silu()?.apply(&self.time_proj)?;
        let b = timestep_proj.dim(0)?;
        let timestep_proj = timestep_proj.reshape((b, 6, hidden_size))?;
        Ok((temb, timestep_proj))
    }
}

// ---------------------------------------------------------------------------
// MLP (SiLU-gated, same as Qwen3MLP)
// ---------------------------------------------------------------------------

/// SiLU-gated feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, Clone)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Attention (self and cross)
// ---------------------------------------------------------------------------

/// Grouped-query attention with per-head QK-norm.
///
/// Supports both self-attention (with RoPE) and cross-attention (Q from hidden
/// states, K/V from encoder outputs, no RoPE). All projections are bias-free
/// when `attention_bias` is false in the config.
///
/// Cross-attention supports optional KV caching: since encoder hidden states
/// are constant across denoising steps, the projected+normed+GQA-expanded K/V
/// can be computed once and reused.
#[derive(Debug, Clone)]
pub struct CrossAttentionKvCache {
    /// `(B, num_heads, S, head_dim)` — ready for Q @ K^T.
    pub k: Tensor,
    /// `(B, num_heads, S, head_dim)`.
    pub v: Tensor,
}

#[derive(Debug, Clone)]
pub struct AceStepAttention {
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
    is_cross_attention: bool,
}

impl AceStepAttention {
    /// Creates a new attention module.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration.
    /// * `is_cross_attention` - When `true`, K and V projections accept
    ///   `encoder_hidden_size()` as input dimension rather than `hidden_size`.
    /// * `vb` - Variable builder scoped to the attention prefix.
    pub fn new(cfg: &AceStepConfig, is_cross_attention: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden_size = head_dim * num_heads;

        // Cross-attention K/V receive encoder states AFTER condition_embedder
        // projection (hidden_size), not the raw encoder output (encoder_hidden_size).
        let kv_input_dim = cfg.hidden_size;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            kv_input_dim,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            kv_input_dim,
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
            is_cross_attention,
        })
    }

    /// Computes (optionally masked) attention.
    ///
    /// # Arguments
    /// * `x` - Hidden states of shape `(B, L, hidden_size)`.
    /// * `encoder_hidden_states` - For cross-attention: encoder output
    ///   `(B, S, encoder_hidden_size)`. Ignored for self-attention.
    /// * `attn_mask` - Additive mask broadcastable to `(B, H, L, S)`, where
    ///   masked positions contain `f32::NEG_INFINITY`.
    /// * `rotary` - Rotary embedding to apply to Q and K (self-attention only).
    ///
    /// Compute cross-attention K/V cache from encoder hidden states.
    /// Returns fully processed K, V: projected, normed, GQA-expanded,
    /// contiguous, shape `(B, num_heads, S, head_dim)`.
    pub fn compute_kv_cache(
        &self,
        encoder_hidden_states: &Tensor,
    ) -> Result<CrossAttentionKvCache> {
        let (b, kv_len, _) = encoder_hidden_states.dims3()?;
        let k = self.k_proj.forward(encoder_hidden_states)?;
        let v = self.v_proj.forward(encoder_hidden_states)?;
        let k = k
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = {
            let flat = k.contiguous()?.flatten(0, 1)?;
            self.k_norm
                .forward(&flat)?
                .reshape((b, self.num_kv_heads, kv_len, self.head_dim))?
        };
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        Ok(CrossAttentionKvCache { k, v })
    }

    /// Forward pass. For cross-attention, pass `kv_cache` to skip K/V
    /// recomputation (encoder states are constant across denoising steps).
    pub fn forward(
        &self,
        x: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&CrossAttentionKvCache>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Q is always computed fresh from x.
        let q = self.q_proj.forward(x)?;
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = {
            let flat = q.contiguous()?.flatten(0, 1)?;
            self.q_norm
                .forward(&flat)?
                .reshape((b, self.num_heads, l, self.head_dim))?
        };

        // K/V: from cache or computed fresh.
        let (q, k, v) = if let Some(cache) = kv_cache {
            // Cross-attention with cache — K/V already projected+normed+expanded.
            (q, cache.k.clone(), cache.v.clone())
        } else {
            // Self-attention or cross-attention without cache.
            let kv_input = if self.is_cross_attention {
                encoder_hidden_states.expect("cross-attention requires encoder_hidden_states")
            } else {
                x
            };
            let k = self.k_proj.forward(kv_input)?;
            let v = self.v_proj.forward(kv_input)?;
            let kv_len = kv_input.dim(1)?;
            let k = k
                .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = {
                let flat = k.contiguous()?.flatten(0, 1)?;
                self.k_norm.forward(&flat)?.reshape((
                    b,
                    self.num_kv_heads,
                    kv_len,
                    self.head_dim,
                ))?
            };

            // RoPE for self-attention only.
            let (q, k) = match rotary {
                Some(rope) => rope.apply(&q, &k, 0)?,
                None => (q, k),
            };

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
            (q, k, v)
        };

        // Scaled dot-product attention.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// DiT layer (transformer block with AdaLN-Zero)
// ---------------------------------------------------------------------------

/// A single DiT transformer layer with AdaLN-Zero modulation.
///
/// Each layer contains self-attention (with optional sliding window), an
/// optional cross-attention sub-layer, and a SiLU-gated MLP. Timestep
/// information is injected via learned shift/scale/gate modulation applied
/// before self-attention and the MLP.
#[derive(Debug, Clone)]
pub struct AceStepDiTLayer {
    self_attn: AceStepAttention,
    self_attn_norm: RmsNorm,
    cross_attn: AceStepAttention,
    cross_attn_norm: RmsNorm,
    mlp: MLP,
    mlp_norm: RmsNorm,
    scale_shift_table: Tensor,
    pub attention_type: String,
}

impl AceStepDiTLayer {
    /// Creates a new DiT layer.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration.
    /// * `attention_type` - Either `"sliding_attention"` or `"full_attention"`.
    ///   Only full-attention layers include cross-attention.
    /// * `vb` - Variable builder scoped to `layers.{i}`.
    pub fn new(cfg: &AceStepConfig, attention_type: &str, vb: VarBuilder) -> Result<Self> {
        let self_attn = AceStepAttention::new(cfg, false, vb.pp("self_attn"))?;
        let self_attn_norm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("self_attn_norm"))?;

        // All layers have cross-attention (use_cross_attention=True in Python)
        let cross_attn = AceStepAttention::new(cfg, true, vb.pp("cross_attn"))?;
        let cross_attn_norm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("cross_attn_norm"))?;

        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let mlp_norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("mlp_norm"))?;
        let scale_shift_table = vb.get((1, 6, cfg.hidden_size), "scale_shift_table")?;

        Ok(Self {
            self_attn,
            self_attn_norm,
            cross_attn,
            cross_attn_norm,
            mlp,
            mlp_norm,
            scale_shift_table,
            #[allow(unused)]
            attention_type: attention_type.to_string(),
        })
    }

    /// Forward pass through one transformer block.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `(B, L, hidden_size)`.
    /// * `rotary` - Rotary embedding for self-attention.
    /// * `timestep_proj` - AdaLN modulation of shape `(B, 6, hidden_size)`.
    /// * `self_attn_mask` - Additive self-attention mask `(1, 1, L, L)`.
    /// * `encoder_hidden_states` - Projected encoder outputs for cross-attention.
    /// * `cross_attn_mask` - Additive cross-attention mask `(B, 1, L, S)`.
    ///
    /// Compute cross-attention KV cache for this layer.
    pub fn compute_cross_kv_cache(
        &self,
        encoder_hidden_states: &Tensor,
    ) -> Result<CrossAttentionKvCache> {
        self.cross_attn.compute_kv_cache(encoder_hidden_states)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rotary: &RotaryEmbedding,
        timestep_proj: &Tensor,
        self_attn_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        cross_kv_cache: Option<&CrossAttentionKvCache>,
    ) -> Result<Tensor> {
        // Compute AdaLN modulation parameters: 6 vectors from scale_shift_table + temb.
        let modulation = self
            .scale_shift_table
            .broadcast_as(timestep_proj.shape())?
            .broadcast_add(timestep_proj)?;
        let chunks = modulation.chunk(6, 1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let c_shift_msa = &chunks[3];
        let c_scale_msa = &chunks[4];
        let c_gate_msa = &chunks[5];

        // Self-attention with AdaLN (never cached — input changes each step).
        let norm_x = self.self_attn_norm.forward(hidden_states)?;
        let norm_x = norm_x
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(shift_msa)?;
        let attn_out = self
            .self_attn
            .forward(&norm_x, None, self_attn_mask, Some(rotary), None)?;
        let mut hidden_states = (hidden_states + attn_out.broadcast_mul(gate_msa)?)?;

        // Cross-attention (standard residual, no AdaLN modulation).
        // When kv_cache is provided, K/V are not recomputed from encoder states.
        let cross_out = {
            let norm_x = self.cross_attn_norm.forward(&hidden_states)?;
            self.cross_attn.forward(
                &norm_x,
                encoder_hidden_states,
                cross_attn_mask,
                None,
                cross_kv_cache,
            )?
        };
        hidden_states = (&hidden_states + cross_out)?;

        // MLP with AdaLN.
        let norm_x = self.mlp_norm.forward(&hidden_states)?;
        let norm_x = norm_x
            .broadcast_mul(&(c_scale_msa + 1.0)?)?
            .broadcast_add(c_shift_msa)?;
        let mlp_out = norm_x.apply(&self.mlp)?;
        let hidden_states = (hidden_states + mlp_out.broadcast_mul(c_gate_msa)?)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Mask construction helpers
// ---------------------------------------------------------------------------

/// Creates a bidirectional sliding-window attention mask.
///
/// Returns a tensor of shape `(1, 1, seq_len, seq_len)` with `0.0` for
/// positions within the window and `f32::NEG_INFINITY` outside it. This
/// produces non-causal (bidirectional) attention restricted to a local
/// neighbourhood of `window` positions in each direction.
pub fn bidirectional_sliding_window_mask(
    seq_len: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (i as i64 - j as i64).unsigned_abs() as usize <= window {
                    0f32
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

/// Converts a 1-D attention mask `(B, S)` of 0/1 values into an additive mask
/// of shape `(B, 1, 1, S)` suitable for cross-attention, where masked positions
/// are filled with `f32::NEG_INFINITY`.
fn expand_cross_attention_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    // mask: (B, S) with 1 = attend, 0 = ignore
    // Convert to additive mask: 0.0 for valid, -inf for padding.
    // Use where_cond to avoid 0 * -inf = NaN.
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let zeros = Tensor::zeros_like(&mask_f32)?;
    let neginf = Tensor::full(f32::NEG_INFINITY, mask_f32.shape(), mask_f32.device())?;
    let valid = mask_f32.gt(&zeros)?; // true for valid (mask > 0)
    let additive = valid.where_cond(&zeros, &neginf)?;
    additive.unsqueeze(1)?.unsqueeze(1)?.to_dtype(dtype) // (B, 1, 1, S)
}

// ---------------------------------------------------------------------------
// Top-level DiT model
// ---------------------------------------------------------------------------

/// The complete Diffusion Transformer model for ACE-Step 1.5.
///
/// This model takes noisy audio latents, diffusion timesteps, and text encoder
/// outputs, and predicts the denoised latent. The input is patchified via a
/// strided Conv1d and the output is reconstructed via ConvTranspose1d.
#[derive(Debug, Clone)]
pub struct AceStepDiTModel {
    proj_in_conv: Conv1d,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    condition_embedder: Linear,
    layers: Vec<AceStepDiTLayer>,
    rotary_emb: RotaryEmbedding,
    norm_out: RmsNorm,
    proj_out_conv: ConvTranspose1d,
    scale_shift_table: Tensor,
    patch_size: usize,
    use_sliding_window: bool,
    sliding_window: usize,
    /// Per-layer cross-attention KV cache (populated on first forward call).
    cross_kv_caches: Vec<Option<CrossAttentionKvCache>>,
}

impl AceStepDiTModel {
    /// Loads the DiT model from pretrained weights.
    ///
    /// The `vb` should be scoped to the `decoder` prefix in the safetensors
    /// checkpoint (e.g., `vb.pp("decoder")`).
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let hidden_size = cfg.hidden_size;

        // Patchify convolution: (B, C, T) -> (B, hidden, T/patch)
        let proj_in_conv = candle_nn::conv1d(
            cfg.in_channels,
            hidden_size,
            patch_size,
            Conv1dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj_in").pp("1"),
        )?;

        let time_embed = TimestepEmbedding::new(hidden_size, vb.pp("time_embed"))?;
        let time_embed_r = TimestepEmbedding::new(hidden_size, vb.pp("time_embed_r"))?;

        let condition_embedder = linear_b(
            cfg.encoder_hidden_size(),
            hidden_size,
            true,
            vb.pp("condition_embedder"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let attn_type = cfg
                .layer_types
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("full_attention");
            layers.push(AceStepDiTLayer::new(cfg, attn_type, vb_l.pp(i))?);
        }

        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;
        let norm_out = RmsNorm::new(hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?;

        // De-patchify transposed convolution: (B, hidden, T/patch) -> (B, out_ch, T)
        let proj_out_conv = candle_nn::conv_transpose1d(
            hidden_size,
            cfg.audio_acoustic_hidden_dim,
            patch_size,
            ConvTranspose1dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj_out").pp("1"),
        )?;

        let scale_shift_table = vb.get((1, 2, hidden_size), "scale_shift_table")?;

        Ok(Self {
            proj_in_conv,
            time_embed,
            time_embed_r,
            condition_embedder,
            layers,
            rotary_emb,
            norm_out,
            proj_out_conv,
            scale_shift_table,
            patch_size,
            use_sliding_window: cfg.use_sliding_window,
            sliding_window: cfg.sliding_window,
            cross_kv_caches: (0..cfg.num_hidden_layers).map(|_| None).collect(),
        })
    }

    /// Denoises audio latents conditioned on timestep and text embeddings.
    ///
    /// # Arguments
    /// * `hidden_states` - Noisy latents of shape `(B, T, audio_acoustic_hidden_dim)`.
    /// * `timestep` - Diffusion timestep tensor of shape `(B,)`.
    /// * `timestep_r` - Reference timestep tensor of shape `(B,)`.
    /// * `_attention_mask` - Unused padding mask for the latent sequence.
    /// * `encoder_hidden_states` - Text encoder output `(B, S, encoder_hidden_size)`.
    /// * `encoder_attention_mask` - Boolean mask for encoder tokens `(B, S)`.
    /// * `context_latents` - Context latent tensor `(B, T, timbre_hidden_dim)` concatenated
    ///   along the channel dimension with `hidden_states`.
    ///
    /// # Returns
    /// Predicted denoised latent of shape `(B, T, audio_acoustic_hidden_dim)`.
    #[allow(clippy::too_many_arguments)]
    /// Clear the cross-attention KV cache. Call this when encoder hidden states
    /// change (e.g. new prompt, or between separate generations).
    pub fn clear_kv_cache(&mut self) {
        for c in &mut self.cross_kv_caches {
            *c = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        _attention_mask: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        context_latents: &Tensor,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let device = hidden_states.device();

        // Timestep embeddings: combine main and reference timestep signals.
        let (temb_t, proj_t) = self.time_embed.forward(timestep)?;
        let delta = (timestep - timestep_r)?;
        let (temb_r, proj_r) = self.time_embed_r.forward(&delta)?;
        let temb = (&temb_t + &temb_r)?;
        let timestep_proj = (&proj_t + &proj_r)?;
        // Concatenate context latents with noisy latents along channel dim.
        let hidden_states = Tensor::cat(&[context_latents, hidden_states], D::Minus1)?;

        // Pad sequence length to a multiple of patch_size.
        let (b, orig_len, c) = hidden_states.dims3()?;
        let hidden_states = if orig_len % self.patch_size != 0 {
            let pad_len = self.patch_size - (orig_len % self.patch_size);
            let padding = Tensor::zeros((b, pad_len, c), dtype, device)?;
            Tensor::cat(&[&hidden_states, &padding], 1)?
        } else {
            hidden_states
        };

        // Patchify: transpose to (B, C, T) for Conv1d, then back to (B, T/patch, hidden).
        let hidden_states = hidden_states
            .transpose(1, 2)?
            .apply(&self.proj_in_conv)?
            .transpose(1, 2)?;
        // Project encoder hidden states to model dimension.
        let encoder_hidden_states = encoder_hidden_states.apply(&self.condition_embedder)?;

        let seq_len = hidden_states.dim(1)?;

        // Build self-attention masks (bidirectional).
        let sliding_mask = if self.use_sliding_window && seq_len > self.sliding_window {
            Some(bidirectional_sliding_window_mask(
                seq_len,
                self.sliding_window,
                device,
                dtype,
            )?)
        } else {
            None
        };

        // Build cross-attention mask from encoder attention mask.
        let cross_attn_mask = match encoder_attention_mask {
            Some(mask) => Some(expand_cross_attention_mask(mask, dtype)?),
            None => None,
        };

        // Build cross-attention KV cache on first call (encoder states are constant).
        for (i, layer) in self.layers.iter().enumerate() {
            if self.cross_kv_caches[i].is_none() {
                self.cross_kv_caches[i] =
                    Some(layer.compute_cross_kv_cache(&encoder_hidden_states)?);
            }
        }

        // Process through transformer layers.
        let mut hidden_states = hidden_states;
        for (i, layer) in self.layers.iter().enumerate() {
            let self_mask = if layer.attention_type == "sliding_attention" {
                sliding_mask.as_ref()
            } else {
                None
            };
            hidden_states = layer.forward(
                &hidden_states,
                &self.rotary_emb,
                &timestep_proj,
                self_mask,
                Some(&encoder_hidden_states),
                cross_attn_mask.as_ref(),
                self.cross_kv_caches[i].as_ref(),
            )?;
        }

        // Output AdaLN: modulate with temb before final projection.
        // scale_shift_table: (1, 2, hidden), temb: (B, hidden) -> unsqueeze -> (B, 1, hidden)
        // Broadcast both to (B, 2, hidden) via broadcast_add
        let temb_unsqueezed = temb.unsqueeze(1)?; // (B, 1, hidden_size)
        let out_modulation = self.scale_shift_table.broadcast_add(&temb_unsqueezed)?; // (B, 2, hidden)
        let chunks = out_modulation.chunk(2, 1)?;
        let shift = &chunks[0]; // (B, 1, hidden_size)
        let scale = &chunks[1]; // (B, 1, hidden_size)
        let hidden_states = self.norm_out.forward(&hidden_states)?;
        let hidden_states = hidden_states
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(shift)?;

        // De-patchify: transpose to (B, hidden, T/patch), conv_transpose, transpose back.
        let hidden_states = hidden_states
            .transpose(1, 2)? // (B, hidden, T/patch)
            .apply(&self.proj_out_conv)? // (B, out_ch, T_reconstructed)
            .transpose(1, 2)?; // (B, T_reconstructed, out_ch)

        // Crop to original sequence length.
        hidden_states.narrow(1, 0, orig_len)
    }
}
