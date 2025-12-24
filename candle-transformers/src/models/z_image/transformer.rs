//! Z-Image Transformer (ZImageTransformer2DModel)
//!
//! Core transformer implementation for Z-Image text-to-image generation.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, VarBuilder};

use crate::models::with_tracing::RmsNorm;

// ==================== Constants ====================

/// AdaLN embedding dimension (256)
pub const ADALN_EMBED_DIM: usize = 256;
/// Sequence padding alignment (32)
pub const SEQ_MULTI_OF: usize = 32;
/// Frequency embedding size for timestep encoding
pub const FREQUENCY_EMBEDDING_SIZE: usize = 256;
/// Max period for sinusoidal encoding
pub const MAX_PERIOD: f64 = 10000.0;

// ==================== Config ====================

/// Z-Image Transformer configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_patch_size")]
    pub all_patch_size: Vec<usize>,
    #[serde(default = "default_f_patch_size")]
    pub all_f_patch_size: Vec<usize>,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_dim")]
    pub dim: usize,
    #[serde(default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(default = "default_n_refiner_layers")]
    pub n_refiner_layers: usize,
    #[serde(default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(default = "default_n_kv_heads")]
    pub n_kv_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_qk_norm")]
    pub qk_norm: bool,
    #[serde(default = "default_cap_feat_dim")]
    pub cap_feat_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_t_scale")]
    pub t_scale: f64,
    #[serde(default = "default_axes_dims")]
    pub axes_dims: Vec<usize>,
    #[serde(default = "default_axes_lens")]
    pub axes_lens: Vec<usize>,
}

fn default_patch_size() -> Vec<usize> {
    vec![2]
}
fn default_f_patch_size() -> Vec<usize> {
    vec![1]
}
fn default_in_channels() -> usize {
    16
}
fn default_dim() -> usize {
    3840
}
fn default_n_layers() -> usize {
    30
}
fn default_n_refiner_layers() -> usize {
    2
}
fn default_n_heads() -> usize {
    30
}
fn default_n_kv_heads() -> usize {
    30
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_qk_norm() -> bool {
    true
}
fn default_cap_feat_dim() -> usize {
    2560
}
fn default_rope_theta() -> f64 {
    256.0
}
fn default_t_scale() -> f64 {
    1000.0
}
fn default_axes_dims() -> Vec<usize> {
    vec![32, 48, 48]
}
fn default_axes_lens() -> Vec<usize> {
    vec![1536, 512, 512]
}

impl Config {
    /// Create configuration for Z-Image Turbo model
    pub fn z_image_turbo() -> Self {
        Self {
            all_patch_size: vec![2],
            all_f_patch_size: vec![1],
            in_channels: 16,
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            n_kv_heads: 30,
            norm_eps: 1e-5,
            qk_norm: true,
            cap_feat_dim: 2560,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![1536, 512, 512],
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Get hidden dimension for FFN
    /// Matches Python: int(dim / 3 * 8) = 10240 for dim=3840
    pub fn hidden_dim(&self) -> usize {
        (self.dim / 3) * 8
    }
}

// ==================== TimestepEmbedder ====================

/// Timestep embedding using sinusoidal encoding + MLP
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(out_size: usize, mid_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(FREQUENCY_EMBEDDING_SIZE, mid_size, vb.pp("mlp").pp("0"))?;
        let linear2 = linear(mid_size, out_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn timestep_embedding(&self, t: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        let half = self.frequency_embedding_size / 2;
        let freqs = Tensor::arange(0u32, half as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (freqs * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
        let args = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        embedding.to_dtype(dtype)
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let dtype = self.linear1.weight().dtype();
        let t_freq = self.timestep_embedding(t, device, dtype)?;
        t_freq.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

// ==================== FeedForward (SwiGLU) ====================

/// SwiGLU feedforward network
#[derive(Debug, Clone)]
pub struct FeedForward {
    w1: candle_nn::Linear,
    w2: candle_nn::Linear,
    w3: candle_nn::Linear,
}

impl FeedForward {
    pub fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = x.apply(&self.w1)?.silu()?;
        let x3 = x.apply(&self.w3)?;
        (x1 * x3)?.apply(&self.w2)
    }
}

// ==================== QkNorm ====================

/// QK normalization using RMSNorm
#[derive(Debug, Clone)]
pub struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    pub fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm_q = RmsNorm::new(head_dim, eps, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(head_dim, eps, vb.pp("norm_k"))?;
        Ok(Self { norm_q, norm_k })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        // q, k shape: (B, seq_len, n_heads, head_dim)
        let q = self.norm_q.forward(q)?;
        let k = self.norm_k.forward(k)?;
        Ok((q, k))
    }
}

// ==================== RopeEmbedder (3D) ====================

/// 3D Rotary Position Embedding for video/image generation
#[derive(Debug, Clone)]
pub struct RopeEmbedder {
    #[allow(dead_code)]
    theta: f64,
    axes_dims: Vec<usize>,
    #[allow(dead_code)]
    axes_lens: Vec<usize>,
    /// Pre-computed cos cache per axis
    cos_cached: Vec<Tensor>,
    /// Pre-computed sin cache per axis
    sin_cached: Vec<Tensor>,
}

impl RopeEmbedder {
    pub fn new(
        theta: f64,
        axes_dims: Vec<usize>,
        axes_lens: Vec<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        assert_eq!(axes_dims.len(), axes_lens.len());
        let mut cos_cached = Vec::with_capacity(axes_dims.len());
        let mut sin_cached = Vec::with_capacity(axes_dims.len());

        for (d, e) in axes_dims.iter().zip(axes_lens.iter()) {
            let half_d = d / 2;
            let inv_freq: Vec<f32> = (0..half_d)
                .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / *d as f32))
                .collect();
            let inv_freq = Tensor::from_vec(inv_freq, half_d, device)?;

            let positions = Tensor::arange(0u32, *e as u32, device)?.to_dtype(DType::F32)?;
            let freqs = positions
                .unsqueeze(1)?
                .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

            cos_cached.push(freqs.cos()?.to_dtype(dtype)?);
            sin_cached.push(freqs.sin()?.to_dtype(dtype)?);
        }

        Ok(Self {
            theta,
            axes_dims,
            axes_lens,
            cos_cached,
            sin_cached,
        })
    }

    /// Get RoPE cos/sin from position IDs
    /// ids: (seq_len, 3) - [frame_id, height_id, width_id]
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::with_capacity(self.axes_dims.len());
        let mut sin_parts = Vec::with_capacity(self.axes_dims.len());

        for (i, _) in self.axes_dims.iter().enumerate() {
            let axis_ids = ids.i((.., i))?.contiguous()?; // (seq_len,) - must be contiguous for Metal
            let cos_i = self.cos_cached[i].index_select(&axis_ids, 0)?;
            let sin_i = self.sin_cached[i].index_select(&axis_ids, 0)?;
            cos_parts.push(cos_i);
            sin_parts.push(sin_i);
        }

        let cos = Tensor::cat(&cos_parts, D::Minus1)?; // (seq_len, head_dim/2)
        let sin = Tensor::cat(&sin_parts, D::Minus1)?;
        Ok((cos, sin))
    }
}

/// Apply RoPE (real-number form, equivalent to PyTorch complex multiplication)
///
/// x: (B, seq_len, n_heads, head_dim)
/// cos, sin: (seq_len, head_dim/2)
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, n_heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Reshape x to interleaved real/imag form: (B, seq_len, n_heads, half_dim, 2)
    let x = x.reshape((b, seq_len, n_heads, half_dim, 2))?;

    // Extract real and imag parts
    let x_real = x.i((.., .., .., .., 0))?; // (B, seq_len, n_heads, half_dim)
    let x_imag = x.i((.., .., .., .., 1))?;

    // Expand cos/sin for broadcasting: (seq_len, half_dim) -> (1, seq_len, 1, half_dim)
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let y_real = (x_real.broadcast_mul(&cos)? - x_imag.broadcast_mul(&sin)?)?;
    let y_imag = (x_real.broadcast_mul(&sin)? + x_imag.broadcast_mul(&cos)?)?;

    // Interleave back
    Tensor::stack(&[y_real, y_imag], D::Minus1)?.reshape((b, seq_len, n_heads, head_dim))
}

// ==================== ZImageAttention ====================

/// Z-Image attention with QK normalization and 3D RoPE
#[derive(Debug, Clone)]
pub struct ZImageAttention {
    to_q: candle_nn::Linear,
    to_k: candle_nn::Linear,
    to_v: candle_nn::Linear,
    to_out: candle_nn::Linear,
    qk_norm: Option<QkNorm>,
    n_heads: usize,
    head_dim: usize,
}

impl ZImageAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim();

        let to_q = linear_no_bias(dim, n_heads * head_dim, vb.pp("to_q"))?;
        let to_k = linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_k"))?;
        let to_v = linear_no_bias(dim, cfg.n_kv_heads * head_dim, vb.pp("to_v"))?;
        let to_out = linear_no_bias(n_heads * head_dim, dim, vb.pp("to_out").pp("0"))?;

        let qk_norm = if cfg.qk_norm {
            Some(QkNorm::new(head_dim, 1e-5, vb.clone())?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            qk_norm,
            n_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let q = hidden_states.apply(&self.to_q)?;
        let k = hidden_states.apply(&self.to_k)?;
        let v = hidden_states.apply(&self.to_v)?;

        // Reshape: (B, seq_len, n_heads * head_dim) -> (B, seq_len, n_heads, head_dim)
        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_heads, self.head_dim))?;

        // Apply QK norm
        let (q, k) = if let Some(ref norm) = self.qk_norm {
            norm.forward(&q, &k)?
        } else {
            (q, k)
        };

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // Transpose for attention: (B, n_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = attention_mask {
            // Convert mask from (B, seq_len) to attention bias
            // 1 = valid, 0 = padding -> 0 = valid, -inf = padding
            let mask = mask.unsqueeze(1)?.unsqueeze(2)?;
            let mask = mask.to_dtype(attn_weights.dtype())?;
            let mask = ((mask - 1.0)? * 1e9)?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let context = attn_probs.matmul(&v)?;

        // Reshape back: (B, n_heads, seq_len, head_dim) -> (B, seq_len, dim)
        let context = context.transpose(1, 2)?.reshape((b, seq_len, ()))?;

        context.apply(&self.to_out)
    }
}

// ==================== ZImageTransformerBlock ====================

/// Z-Image transformer block with optional AdaLN modulation
#[derive(Debug, Clone)]
pub struct ZImageTransformerBlock {
    attention: ZImageAttention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm1: RmsNorm,
    ffn_norm2: RmsNorm,
    adaln_modulation: Option<candle_nn::Linear>,
}

impl ZImageTransformerBlock {
    pub fn new(cfg: &Config, modulation: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim();

        let attention = ZImageAttention::new(cfg, vb.pp("attention"))?;
        let feed_forward = FeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?;

        let attention_norm1 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm1"))?;
        let attention_norm2 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("attention_norm2"))?;
        let ffn_norm1 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm1"))?;
        let ffn_norm2 = RmsNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm2"))?;

        let adaln_modulation = if modulation {
            let adaln_dim = dim.min(ADALN_EMBED_DIM);
            Some(linear(adaln_dim, 4 * dim, vb.pp("adaLN_modulation").pp("0"))?)
        } else {
            None
        };

        Ok(Self {
            attention,
            feed_forward,
            attention_norm1,
            attention_norm2,
            ffn_norm1,
            ffn_norm2,
            adaln_modulation,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        adaln_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(ref adaln) = self.adaln_modulation {
            let adaln_input = adaln_input.expect("adaln_input required when modulation=true");
            // (B, 256) -> (B, 4*dim) -> (B, 1, 4*dim) -> chunk into 4
            let modulation = adaln_input.apply(adaln)?.unsqueeze(1)?;
            let chunks = modulation.chunk(4, D::Minus1)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
                (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

            // Apply tanh gate
            let gate_msa = gate_msa.tanh()?;
            let gate_mlp = gate_mlp.tanh()?;
            let scale_msa = (scale_msa + 1.0)?;
            let scale_mlp = (scale_mlp + 1.0)?;

            // Attention block
            let normed = self.attention_norm1.forward(x)?;
            let scaled = normed.broadcast_mul(&scale_msa)?;
            let attn_out = self.attention.forward(&scaled, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + gate_msa.broadcast_mul(&attn_out)?)?;

            // FFN block
            let normed = self.ffn_norm1.forward(&x)?;
            let scaled = normed.broadcast_mul(&scale_mlp)?;
            let ffn_out = self.feed_forward.forward(&scaled)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + gate_mlp.broadcast_mul(&ffn_out)?
        } else {
            // Without modulation
            let normed = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&normed, attn_mask, cos, sin)?;
            let attn_out = self.attention_norm2.forward(&attn_out)?;
            let x = (x + attn_out)?;

            let normed = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let ffn_out = self.ffn_norm2.forward(&ffn_out)?;
            x + ffn_out
        }
    }
}

// ==================== FinalLayer ====================

/// LayerNorm without learnable parameters (elementwise_affine=False)
#[derive(Debug, Clone)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        // Subtract mean
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        // Divide by std
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)
    }
}

/// Final layer for output projection
#[derive(Debug, Clone)]
pub struct FinalLayer {
    norm_final: LayerNormNoParams,
    linear: candle_nn::Linear,
    adaln_silu: candle_nn::Linear,
}

impl FinalLayer {
    pub fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNormNoParams::new(1e-6);
        let linear = candle_nn::linear(hidden_size, out_channels, vb.pp("linear"))?;
        let adaln_dim = hidden_size.min(ADALN_EMBED_DIM);
        let adaln_silu =
            candle_nn::linear(adaln_dim, hidden_size, vb.pp("adaLN_modulation").pp("1"))?;

        Ok(Self {
            norm_final,
            linear,
            adaln_silu,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let scale = c.silu()?.apply(&self.adaln_silu)?;
        let scale = (scale + 1.0)?.unsqueeze(1)?;
        let x = self.norm_final.forward(x)?.broadcast_mul(&scale)?;
        x.apply(&self.linear)
    }
}

// ==================== Patchify / Unpatchify ====================

/// Convert image to patch sequence
/// input: (B, C, F, H, W) where F=1 for images
/// output: (B, num_patches, patch_dim), (F, H, W) original size
///
/// For F=1, patch_size=2:
/// - Input: (B, 16, 1, 64, 64)
/// - After reshape to patches: (B, 32, 32, 2, 2, 16) -> (B, 1024, 64)
pub fn patchify(
    x: &Tensor,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<(Tensor, (usize, usize, usize))> {
    let (b, c, f, h, w) = x.dims5()?;
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;

    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let num_patches = f_tokens * h_tokens * w_tokens;
    let patch_dim = pf * ph * pw * c;

    // Simplified approach: use unfold-like operations
    // For Z-Image with f_patch_size=1, F dimension stays as-is
    
    // Step 1: (B, C, F, H, W) -> (B, F, H, W, C)
    let x = x.permute((0, 2, 3, 4, 1))?;
    
    // Step 2: Reshape H: (B, F, H, W, C) -> (B, F, h_tokens, ph, W, C)
    let x = x.reshape((b, f, h_tokens, ph, w, c))?;
    
    // Step 3: Permute to (B, F, h_tokens, W, ph, C)
    let x = x.permute((0, 1, 2, 4, 3, 5))?;
    
    // Step 4: Reshape W and merge ph*C: (B, F, h_tokens, w_tokens, pw, ph*C)
    let x = x.reshape((b, f, h_tokens, w_tokens, pw, ph * c))?;
    
    // Step 5: Permute to (B, F, h_tokens, w_tokens, ph*C, pw) - not needed
    // Step 6: Merge to patches: (B, F*h_tokens*w_tokens, pw*ph*C)
    let x = x.reshape((b, f * h_tokens * w_tokens, pw * ph * c))?;
    
    // For f_patch_size=1, patch_dim = pf * ph * pw * c = ph * pw * c
    // Ensure final shape
    let x = x.reshape((b, num_patches, patch_dim))?;

    Ok((x, (f, h, w)))
}

/// Convert patch sequence back to image
/// input: (B, seq_len, patch_dim)
/// output: (B, C, F, H, W)
pub fn unpatchify(
    x: &Tensor,
    size: (usize, usize, usize),
    patch_size: usize,
    f_patch_size: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let (f, h, w) = size;
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;

    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let ori_len = f_tokens * h_tokens * w_tokens;

    let (b, _, _) = x.dims3()?;
    let x = x.narrow(1, 0, ori_len)?; // Remove padding

    // Reverse of patchify using <=6D operations
    // (B, num_patches, patch_dim) -> (B, C, F, H, W)
    
    // Step 1: (B, num_patches, patch_dim) -> (B, f*h_tokens*w_tokens, pw, ph*C)
    // patch_dim = pf * ph * pw * C, for pf=1: patch_dim = ph * pw * C
    let x = x.reshape((b, f_tokens * h_tokens * w_tokens, pw, ph * out_channels))?;
    
    // Step 2: Reshape to (B, f_tokens, h_tokens, w_tokens, pw, ph*C)
    let x = x.reshape((b, f_tokens, h_tokens, w_tokens, pw, ph * out_channels))?;
    
    // Step 3: Permute to (B, f_tokens, h_tokens, w_tokens, ph*C, pw) -> not needed
    // Step 4: Reshape to split ph from C: (B, f_tokens, h_tokens, w_tokens*pw, ph, C)
    let x = x.reshape((b, f_tokens, h_tokens, w_tokens * pw, ph, out_channels))?;
    
    // Step 5: Permute to (B, f_tokens, h_tokens, ph, w_tokens*pw, C)
    let x = x.permute((0, 1, 2, 4, 3, 5))?;
    
    // Step 6: Reshape to (B, f_tokens, h_tokens*ph, w_tokens*pw, C) = (B, F, H, W, C)
    let x = x.reshape((b, f_tokens, h, w, out_channels))?;
    
    // Step 7: Permute to (B, C, F, H, W)
    let x = x.permute((0, 4, 1, 2, 3))?;
    
    Ok(x)
}

/// Create 3D coordinate grid for RoPE position IDs
/// size: (F, H, W)
/// start: (f0, h0, w0)
/// output: (F*H*W, 3)
pub fn create_coordinate_grid(
    size: (usize, usize, usize),
    start: (usize, usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let (f, h, w) = size;
    let (f0, h0, w0) = start;

    let mut coords = Vec::with_capacity(f * h * w * 3);
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                coords.push((f0 + fi) as u32);
                coords.push((h0 + hi) as u32);
                coords.push((w0 + wi) as u32);
            }
        }
    }

    Tensor::from_vec(coords, (f * h * w, 3), device)
}

// ==================== ZImageTransformer2DModel ====================

/// Z-Image Transformer 2D Model
#[derive(Debug, Clone)]
pub struct ZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: candle_nn::Linear,
    x_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    #[allow(dead_code)]
    x_pad_token: Tensor,
    #[allow(dead_code)]
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl ZImageTransformer2DModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        // TimestepEmbedder
        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        // Caption embedder
        let cap_embedder_norm =
            RmsNorm::new(cfg.cap_feat_dim, cfg.norm_eps, vb.pp("cap_embedder").pp("0"))?;
        let cap_embedder_linear =
            linear(cfg.cap_feat_dim, cfg.dim, vb.pp("cap_embedder").pp("1"))?;

        // Patch embedder (assuming patch_size=2, f_patch_size=1)
        let patch_dim =
            cfg.all_f_patch_size[0] * cfg.all_patch_size[0] * cfg.all_patch_size[0] * cfg.in_channels;
        let x_embedder = linear(patch_dim, cfg.dim, vb.pp("all_x_embedder").pp("2-1"))?;

        // Final layer
        let out_channels =
            cfg.all_patch_size[0] * cfg.all_patch_size[0] * cfg.all_f_patch_size[0] * cfg.in_channels;
        let final_layer =
            FinalLayer::new(cfg.dim, out_channels, vb.pp("all_final_layer").pp("2-1"))?;

        // Pad tokens
        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;

        // Noise refiner (with modulation)
        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        // Context refiner (without modulation)
        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        // Main layers (with modulation)
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ZImageTransformerBlock::new(cfg, true, vb.pp("layers").pp(i))?);
        }

        // RoPE embedder
        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Latent tensor (B, C, F, H, W)
    /// * `t` - Timesteps [0, 1] (B,)
    /// * `cap_feats` - Caption features (B, text_len, cap_feat_dim)
    /// * `cap_mask` - Caption attention mask (B, text_len), 1=valid, 0=padding
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        let device = x.device();
        let (b, _c, f, h, w) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let f_patch_size = self.cfg.all_f_patch_size[0];

        // 1. Timestep embedding
        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?; // (B, 256)

        // 2. Patchify and embed image
        let (x_patches, orig_size) = patchify(x, patch_size, f_patch_size)?;
        let mut x = x_patches.apply(&self.x_embedder)?; // (B, img_seq, dim)
        let img_seq_len = x.dim(1)?;

        // 3. Create image position IDs
        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w / patch_size;
        let text_len = cap_feats.dim(1)?;

        let x_pos_ids = create_coordinate_grid(
            (f_tokens, h_tokens, w_tokens),
            (text_len + 1, 0, 0), // offset for text
            device,
        )?;
        let (x_cos, x_sin) = self.rope_embedder.forward(&x_pos_ids)?;

        // 4. Caption embedding
        let cap_normed = self.cap_embedder_norm.forward(cap_feats)?;
        let mut cap = cap_normed.apply(&self.cap_embedder_linear)?; // (B, text_len, dim)

        // 5. Create caption position IDs
        let cap_pos_ids = create_coordinate_grid((text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        // 6. Create attention masks
        let x_attn_mask = Tensor::ones((b, img_seq_len), DType::U8, device)?;
        let cap_attn_mask = cap_mask.to_dtype(DType::U8)?;

        // 7. Noise refiner (process image with modulation)
        for layer in &self.noise_refiner {
            x = layer.forward(&x, Some(&x_attn_mask), &x_cos, &x_sin, Some(&adaln_input))?;
        }

        // 8. Context refiner (process text without modulation)
        for layer in &self.context_refiner {
            cap = layer.forward(&cap, Some(&cap_attn_mask), &cap_cos, &cap_sin, None)?;
        }

        // 9. Concatenate image and text: [image_tokens, text_tokens]
        let unified = Tensor::cat(&[&x, &cap], 1)?; // (B, img_seq + text_len, dim)

        // 10. Create unified position IDs and attention mask
        let unified_pos_ids = Tensor::cat(&[&x_pos_ids, &cap_pos_ids], 0)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;
        let unified_attn_mask = Tensor::cat(&[&x_attn_mask, &cap_attn_mask], 1)?;

        // 11. Main transformer layers
        let mut unified = unified;
        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                Some(&unified_attn_mask),
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        // 12. Final layer (only on image portion)
        let x_out = unified.narrow(1, 0, img_seq_len)?;
        let x_out = self.final_layer.forward(&x_out, &adaln_input)?;

        // 13. Unpatchify
        unpatchify(
            &x_out,
            orig_size,
            patch_size,
            f_patch_size,
            self.cfg.in_channels,
        )
    }

    /// Get model configuration
    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
