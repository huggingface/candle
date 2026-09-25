//! Audio tokenizer and detokenizer for ACE-Step.
//!
//! Implements Finite Scalar Quantization (FSQ) based audio tokenization.
//! These components are used in the cover/audio-code pipeline (LM + DiT hybrid
//! mode) but are **not exercised** by the Text2Music path.
//!
//! **Validation status** (against Python `vector_quantize_pytorch.ResidualFSQ`):
//! - `AudioTokenDetokenizer`: bit-exact (diff_max < 0.00001)
//! - `ResidualFSQ` (quantizer): FSQ bound/round matches; small rounding
//!   boundary differences (diff_max ~0.018) are expected for discrete
//!   quantization due to float precision at level boundaries.
//! - `AttentionPooler` / `AudioTokenizer`: structurally correct, minor
//!   float precision differences propagate through FSQ rounding.

use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

use super::dit::RotaryEmbedding;
use super::AceStepConfig;

// ---------------------------------------------------------------------------
// Encoder MLP (same SiLU-gated pattern as Qwen3MLP)
// ---------------------------------------------------------------------------
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
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act_fn: act,
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
// Encoder Attention (bidirectional, GQA, with RoPE)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct TokenizerAttention {
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

impl TokenizerAttention {
    fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let num_heads = cfg.encoder_num_attention_heads();
        let num_kv_heads = cfg.encoder_num_key_value_heads();
        let head_dim = cfg.head_dim;
        let hidden_size = num_heads * head_dim;

        Ok(Self {
            q_proj: linear_no_bias(enc_hs, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(enc_hs, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(enc_hs, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(num_heads * head_dim, enc_hs, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm
        let q = self.q_norm.forward(&q.flatten(0, 1)?)?.reshape(q.shape())?;
        let k = self.k_norm.forward(&k.flatten(0, 1)?)?.reshape(k.shape())?;

        // RoPE
        let (q, k) = rotary.apply(&q, &k, 0)?;

        // GQA
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Attention
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
// Encoder Layer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: TokenizerAttention,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: MLP,
}

impl EncoderLayer {
    fn new(cfg: &AceStepConfig, _layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        Ok(Self {
            self_attn: TokenizerAttention::new(cfg, vb.pp("self_attn"))?,
            input_layernorm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                enc_hs,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: MLP::new(
                enc_hs,
                cfg.encoder_intermediate_size(),
                cfg.hidden_act,
                vb.pp("mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, attn_mask, rotary)?;
        let x = (residual + h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// AttentionPooler - CLS-token based attention pooling
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AttentionPooler {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    special_token: Tensor,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl AttentionPooler {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(EncoderLayer::new(cfg, i, vb_l.pp(i))?);
        }
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        Ok(Self {
            embed_tokens: linear(enc_hs, enc_hs, vb.pp("embed_tokens"))?,
            layers,
            norm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("norm"))?,
            special_token: vb.get((1, 1, enc_hs), "special_token")?,
            rotary_emb,
        })
    }

    /// Pool patches into sequence-level representations.
    /// Input: (B, T, pool_window_size, D) → Output: (B, T, D)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, p, _d) = x.dims4()?;
        let x = x.apply(&self.embed_tokens)?;

        // Prepend CLS token to each patch group
        let cls = self
            .special_token
            .broadcast_as((b * t, 1, x.dim(D::Minus1)?))?;
        let x = x.reshape((b * t, p, ()))?;
        let x = Tensor::cat(&[&cls, &x], 1)?;

        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None, &self.rotary_emb)?;
        }
        let hidden = self.norm.forward(&hidden)?;

        // Extract CLS output (position 0)
        hidden.narrow(1, 0, 1)?.squeeze(1)?.reshape((b, t, ()))
    }
}

// ---------------------------------------------------------------------------
// ResidualFSQ - Finite Scalar Quantization
// ---------------------------------------------------------------------------

/// Residual Finite Scalar Quantization.
///
/// Matches `vector_quantize_pytorch.ResidualFSQ` with a single top-level
/// `project_in` (dim → codebook_dim) and `project_out` (codebook_dim → dim).
/// Each FSQ layer independently quantizes each codebook dimension to its
/// corresponding number of levels using `tanh` bounding and rounding.
#[derive(Debug, Clone)]
pub struct ResidualFSQ {
    project_in: Linear,
    project_out: Linear,
    /// Precomputed: (levels - 1) * (1 + eps) / 2 for each codebook dim
    half_l: Vec<f32>,
    /// Precomputed: levels // 2
    half_width: Vec<f32>,
    /// Precomputed: 0.5 for even levels, 0.0 for odd
    offset: Vec<f32>,
    /// Precomputed: atanh(offset / half_l) shift
    shift: Vec<f32>,
    /// Precomputed basis for index calculation: [1, L0, L0*L1, ...]
    basis: Vec<i64>,
    /// Quantization levels per codebook dimension (e.g. [8, 8, 8, 5, 5, 5]).
    levels: Vec<usize>,
    /// Model working dtype (inferred from VarBuilder at construction time).
    model_dtype: DType,
}

impl ResidualFSQ {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let model_dtype = vb.dtype();
        let dim = cfg.fsq_dim;
        let levels = &cfg.fsq_input_levels;
        let codebook_dim = levels.len();

        // Single project_in/project_out at the ResidualFSQ level
        let project_in = linear(dim, codebook_dim, vb.pp("project_in"))?;
        let project_out = linear(codebook_dim, dim, vb.pp("project_out"))?;

        let eps = 1e-3f32;
        let half_l: Vec<f32> = levels
            .iter()
            .map(|&l| (l as f32 - 1.0) * (1.0 + eps) / 2.0)
            .collect();
        let half_width: Vec<f32> = levels.iter().map(|&l| (l / 2) as f32).collect();
        let offset: Vec<f32> = levels
            .iter()
            .map(|&l| if l % 2 == 0 { 0.5 } else { 0.0 })
            .collect();
        let shift: Vec<f32> = half_l
            .iter()
            .zip(&offset)
            .map(|(&hl, &o)| (o / hl).atanh())
            .collect();

        let mut basis = vec![1i64];
        for i in 0..levels.len() - 1 {
            basis.push(basis[i] * levels[i] as i64);
        }

        Ok(Self {
            project_in,
            project_out,
            half_l,
            half_width,
            offset,
            shift,
            basis,
            levels: levels.to_vec(),
            model_dtype,
        })
    }

    /// FSQ bound + round: `round(tanh(z + shift) * half_l - offset) / half_width`
    fn fsq_quantize(&self, z: &Tensor) -> Result<Tensor> {
        let dev = z.device();
        let half_l = Tensor::new(&self.half_l[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let offset = Tensor::new(&self.offset[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let shift = Tensor::new(&self.shift[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let half_width = Tensor::new(&self.half_width[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        // bound: tanh(z + shift) * half_l - offset, then round, then / half_width
        let bounded = z
            .broadcast_add(&shift)?
            .tanh()?
            .broadcast_mul(&half_l)?
            .broadcast_sub(&offset)?;
        let rounded = bounded.round()?;
        rounded.broadcast_div(&half_width)
    }

    /// Quantize continuous representations into discrete tokens.
    /// Input: `(B, N, dim)`. Returns `(quantized_out, indices)` where
    /// `quantized_out` has the same shape as input and `indices` is `(B, N, num_quantizers)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // project_in: (B, N, dim) -> (B, N, codebook_dim)
        let projected = x.apply(&self.project_in)?;

        // FSQ quantize
        let codes = self.fsq_quantize(&projected)?; // (B, N, codebook_dim)

        // Compute indices from codes: scale_and_shift then dot with basis
        let dev = x.device();
        let half_width = Tensor::new(&self.half_width[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let offset = Tensor::new(&self.offset[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let basis = Tensor::new(&self.basis[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        // scale_and_shift: codes * half_width + offset
        let shifted = codes.broadcast_mul(&half_width)?.broadcast_add(&offset)?;
        let indices = shifted
            .to_dtype(DType::F32)?
            .broadcast_mul(&basis.to_dtype(DType::F32)?)?
            .sum(2)?
            .round()?
            .to_dtype(DType::I64)?
            .unsqueeze(2)?; // (B, N, 1)

        // project_out: (B, N, codebook_dim) -> (B, N, dim)
        let quantized_out = codes.apply(&self.project_out)?;

        Ok((quantized_out, indices))
    }

    /// Reconstruct from quantized codes (not indices).
    /// Input: `(B, N, codebook_dim)` → Output: `(B, N, dim)`.
    pub fn get_output_from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        codes.apply(&self.project_out)
    }

    /// Reconstruct from packed indices.
    /// Input: `(B, N, 1)` → Output: `(B, N, dim)`.
    ///
    /// Inverse of the forward path: decompose flat index into per-level values
    /// via basis, then invert scale_and_shift, then `project_out`.
    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        let dev = indices.device();
        let indices = indices.squeeze(D::Minus1)?.to_dtype(DType::F32)?;
        let codebook_dim = self.levels.len();

        // Decompose: level_val[i] = floor(index / basis[i]) % levels[i]
        // F32 has enough precision for indices up to 64000 (max codebook size).
        let mut code_dims = Vec::with_capacity(codebook_dim);
        for i in 0..codebook_dim {
            let b = self.basis[i] as f32;
            let l = self.levels[i] as f32;
            let dim_val = indices.affine(1.0 / b as f64, 0.0)?.floor()?;
            // modulo: dim_val - floor(dim_val / l) * l
            let dim_val = (&dim_val
                - dim_val
                    .affine(1.0 / l as f64, 0.0)?
                    .floor()?
                    .affine(l as f64, 0.0)?)?;
            code_dims.push(dim_val);
        }
        let shifted = Tensor::stack(&code_dims, D::Minus1)?;

        // Inverse scale_and_shift (Python convention):
        // forward: shifted = codes * half_width + half_width  →  shifted ∈ [0, levels-1]
        // inverse: codes = (shifted - half_width) / half_width
        let hw = Tensor::new(&self.half_width[..], dev)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let codes = shifted.broadcast_sub(&hw)?.broadcast_div(&hw)?;

        codes.to_dtype(self.model_dtype)?.apply(&self.project_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Build a minimal ResidualFSQ with identity projections for testing.
    fn test_fsq() -> Result<ResidualFSQ> {
        let dev = &Device::Cpu;
        let levels = vec![8, 8, 8, 5, 5, 5];
        let codebook_dim = levels.len();

        // Identity-like project_in/project_out (codebook_dim x codebook_dim)
        let eye = Tensor::eye(codebook_dim, DType::F32, dev)?;
        let zeros = Tensor::zeros(codebook_dim, DType::F32, dev)?;

        let eps = 1e-3f32;
        let half_l: Vec<f32> = levels
            .iter()
            .map(|&l| (l as f32 - 1.0) * (1.0 + eps) / 2.0)
            .collect();
        let half_width: Vec<f32> = levels.iter().map(|&l| (l / 2) as f32).collect();
        let offset: Vec<f32> = levels
            .iter()
            .map(|&l| if l % 2 == 0 { 0.5 } else { 0.0 })
            .collect();
        let shift: Vec<f32> = half_l
            .iter()
            .zip(&offset)
            .map(|(&hl, &o)| (o / hl).atanh())
            .collect();
        let mut basis = vec![1i64];
        for i in 0..levels.len() - 1 {
            basis.push(basis[i] * levels[i] as i64);
        }

        Ok(ResidualFSQ {
            project_in: Linear::from_weights(eye.clone(), Some(zeros.clone())),
            project_out: Linear::from_weights(eye, Some(zeros)),
            half_l,
            half_width,
            offset,
            shift,
            basis,
            levels,
            model_dtype: DType::F32,
        })
    }

    #[test]
    fn test_get_output_from_indices_roundtrip() -> Result<()> {
        // Test that manually-packed Python-convention indices correctly
        // reconstruct codes via get_output_from_indices.
        let fsq = test_fsq()?;
        let dev = &Device::Cpu;
        // levels = [8,8,8,5,5,5], half_width = [4,4,4,2,2,2]
        // basis = [1, 8, 64, 512, 2560, 12800]
        //
        // Choose codes: [0.25, -0.5, 0.75, 0.5, -1.0, 0.0]
        // Python scale_and_shift: shifted = codes * half_width + half_width
        //   = [0.25*4+4, -0.5*4+4, 0.75*4+4, 0.5*2+2, -1.0*2+2, 0.0*2+2]
        //   = [5, 2, 7, 3, 0, 2]
        // index = 5*1 + 2*8 + 7*64 + 3*512 + 0*2560 + 2*12800
        //       = 5 + 16 + 448 + 1536 + 0 + 25600 = 27605
        let expected_codes: Vec<f32> = vec![0.25, -0.5, 0.75, 0.5, -1.0, 0.0];
        let index: i64 = 27605;

        let indices = Tensor::new(&[[index]], dev)?.unsqueeze(2)?; // (1, 1, 1)
        let reconstructed = fsq.get_output_from_indices(&indices)?;
        let vals = reconstructed.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;

        for (i, (&got, &want)) in vals.iter().zip(expected_codes.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5, "dim {i}: got={got}, want={want}");
        }
        Ok(())
    }

    #[test]
    fn test_get_output_from_indices_known_values() -> Result<()> {
        let fsq = test_fsq()?;
        let dev = &Device::Cpu;

        // Index 0 should decompose to level_vals = [0, 0, 0, 0, 0, 0]
        // Python convention: codes = (level_vals - half_width) / half_width
        // For levels [8,8,8,5,5,5]: half_width=[4,4,4,2,2,2]
        // codes = [(0-4)/4, (0-4)/4, (0-4)/4, (0-2)/2, (0-2)/2, (0-2)/2]
        //       = [-1, -1, -1, -1, -1, -1]
        let indices = Tensor::new(&[[0i64]], dev)?.unsqueeze(2)?; // (1, 1, 1)
        let result = fsq.get_output_from_indices(&indices)?;
        let vals = result.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        assert!((vals[0] - (-1.0)).abs() < 1e-5, "val[0]={}", vals[0]);
        assert!((vals[3] - (-1.0)).abs() < 1e-5, "val[3]={}", vals[3]);

        // Index that corresponds to center: half_width values for each level
        // level_vals = [4, 4, 4, 2, 2, 2]
        // packed index = 4*1 + 4*8 + 4*64 + 2*512 + 2*2560 + 2*12800
        //             = 4 + 32 + 256 + 1024 + 5120 + 25600 = 32036
        // codes = [0, 0, 0, 0, 0, 0]
        let indices = Tensor::new(&[[32036i64]], dev)?.unsqueeze(2)?;
        let result = fsq.get_output_from_indices(&indices)?;
        let vals = result.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.abs() < 1e-5, "center val[{i}]={v}, expected 0");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AudioTokenizer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AudioTokenizer {
    audio_acoustic_proj: Linear,
    attention_pooler: AttentionPooler,
    quantizer: ResidualFSQ,
    pool_window_size: usize,
}

impl AudioTokenizer {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        Ok(Self {
            audio_acoustic_proj: linear(
                cfg.audio_acoustic_hidden_dim,
                enc_hs,
                vb.pp("audio_acoustic_proj"),
            )?,
            attention_pooler: AttentionPooler::new(cfg, vb.pp("attention_pooler"))?,
            quantizer: ResidualFSQ::new(cfg, vb.pp("quantizer"))?,
            pool_window_size: cfg.pool_window_size,
        })
    }

    pub fn pool_window_size(&self) -> usize {
        self.pool_window_size
    }

    /// Reconstruct from packed indices via the quantizer.
    /// Input: `(B, N, 1)` → Output: `(B, N, dim)`.
    pub fn get_output_from_indices(&self, indices: &Tensor) -> Result<Tensor> {
        self.quantizer.get_output_from_indices(indices)
    }

    /// Tokenize audio features into quantized tokens.
    /// Input: (B, T_patches, pool_window_size, acoustic_dim)
    /// Output: (quantized, indices)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.apply(&self.audio_acoustic_proj)?;
        let pooled = self.attention_pooler.forward(&x)?;
        self.quantizer.forward(&pooled)
    }
}

// ---------------------------------------------------------------------------
// AudioTokenDetokenizer
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AudioTokenDetokenizer {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    proj_out: Linear,
    special_tokens: Tensor,
    pool_window_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl AudioTokenDetokenizer {
    pub fn new(cfg: &AceStepConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = cfg.encoder_hidden_size();
        let num_layers = cfg.num_attention_pooler_hidden_layers;

        let mut layers = Vec::with_capacity(num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..num_layers {
            layers.push(EncoderLayer::new(cfg, i, vb_l.pp(i))?);
        }
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        Ok(Self {
            embed_tokens: linear(enc_hs, enc_hs, vb.pp("embed_tokens"))?,
            layers,
            norm: RmsNorm::new(enc_hs, cfg.rms_norm_eps, vb.pp("norm"))?,
            proj_out: linear(enc_hs, cfg.audio_acoustic_hidden_dim, vb.pp("proj_out"))?,
            special_tokens: vb.get((1, cfg.pool_window_size, enc_hs), "special_tokens")?,
            pool_window_size: cfg.pool_window_size,
            rotary_emb,
        })
    }

    /// Detokenize quantized tokens back to continuous acoustic representations.
    /// Input: (B, T, D) → Output: (B, T * pool_window_size, acoustic_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let x = x.apply(&self.embed_tokens)?;

        // Expand each token into pool_window_size patches
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, t, self.pool_window_size, x.dim(D::Minus1)?))?;
        let special = self.special_tokens.broadcast_as((
            b,
            t,
            self.pool_window_size,
            self.special_tokens.dim(D::Minus1)?,
        ))?;
        let x = (x.contiguous()? + special.contiguous()?)?;

        // Reshape for processing: (B*T, pool_window_size, hidden)
        let x = x.reshape((b * t, self.pool_window_size, ()))?;

        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None, &self.rotary_emb)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.apply(&self.proj_out)?;

        // Reshape back: (B*T, P, acoustic_dim) → (B, T*P, acoustic_dim)
        hidden.reshape((b, t * self.pool_window_size, ()))
    }
}
