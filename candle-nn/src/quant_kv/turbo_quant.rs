//! TurboQuant: Combined 4-bit KV Cache Quantization using PolarQuant (MSE) and QJL (Residual).
//!
//! ## Paper Reference
//!
//! "TurboQuant: 4-Bit KV Cache Quantization with Polar MSE and QJL Residuals"
//! (2024) — arxiv:2407.13246

use candle::{Result, Tensor};
use super::polar_quant::{
    polar_attention_scores_vectorized, polar_dequantize_batch, polar_quantize_tensor,
    PolarQuantConfig, PolarQuantTensors,
};
use super::qjl::{qjl_attention_scores_vectorized, qjl_quantize_tensor, QjlConfig, QjlTensors};

/// Configuration for TurboQuant quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboQuantConfig {
    pub polar_config: PolarQuantConfig,
    pub qjl_config: QjlConfig,
}

impl TurboQuantConfig {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self::new_3p5bit(dim, seed)
    }

    pub fn new_2bit(dim: usize, seed: u64) -> Self {
        let polar_config = PolarQuantConfig::new(dim, seed, 2, 2);
        let qjl_config = QjlConfig::new(dim / 16, seed + 1);
        Self { polar_config, qjl_config }
    }

    pub fn new_3p5bit(dim: usize, seed: u64) -> Self {
        let polar_config = PolarQuantConfig::new(dim, seed, 4, 1);
        let qjl_config = QjlConfig::new(dim / 8, seed + 1);
        Self { polar_config, qjl_config }
    }

    pub fn new_4bit(dim: usize, seed: u64) -> Self {
        let polar_config = PolarQuantConfig::new(dim, seed, 4, 2);
        let qjl_config = QjlConfig::new(dim / 4, seed + 1);
        Self { polar_config, qjl_config }
    }
}

/// GPU-native storage for a batch of TurboQuant-quantized keys.
#[derive(Debug, Clone)]
pub struct TurboQuantTensors {
    pub mse_tensors: PolarQuantTensors,
    pub qjl_tensors: QjlTensors,
    pub cur_len: usize,
}

impl TurboQuantTensors {
    pub fn new(num_heads: usize, max_seq_len: usize, dim: usize, device: &candle::Device) -> Result<Self> {
        let mse_tensors = PolarQuantTensors::new(num_heads, max_seq_len, dim, device)?;
        let qjl_tensors = QjlTensors::new(num_heads, max_seq_len, dim / 8, device)?;
        Ok(Self { mse_tensors, qjl_tensors, cur_len: 0 })
    }
}

/// Quantize a key tensor using TurboQuant.
pub fn turbo_quantize_tensor(
    k: &Tensor,
    config: &TurboQuantConfig,
) -> Result<((Tensor, Tensor, Tensor), (Tensor, Tensor))> {
    // Stage 1: MSE-optimized PolarQuant encoding
    let (l1, ln, r) = polar_quantize_tensor(k, &config.polar_config)?;

    // Dequantize MSE codes to get k̃, then compute residual r = k - k̃
    let dims = k.dims();
    let seq_len = dims[dims.len() - 2];
    let temp_polar = PolarQuantTensors {
        level1_codes: vec![l1.clone()],
        leveln_codes: vec![ln.clone()],
        radii: vec![r.clone()],
        cur_len: seq_len,
    };
    let k_reconstructed = polar_dequantize_batch(&temp_polar, &config.polar_config)?;
    // k_reconstructed: [num_heads, seq_len, d] in F32

    // Align ranks: k may be 4D [1, num_heads, seq_len, d], k_reconstructed is 3D
    let k_f32 = k.to_dtype(candle::DType::F32)?;
    let k_reconstructed_matched = if dims.len() == 4 {
        k_reconstructed.unsqueeze(0)?
    } else {
        k_reconstructed
    };
    // Residual in original dtype so QJL's random projection dtype matches
    let residual = k_f32
        .sub(&k_reconstructed_matched)?
        .to_dtype(k.dtype())?;

    // Stage 2: QJL encoding of residual (inner-product-optimized correction)
    let qjl = qjl_quantize_tensor(&residual, &config.qjl_config)?;

    Ok(((l1, ln, r), qjl))
}

/// Compute attention scores using vectorized TurboQuant estimation on GPU.
pub fn turbo_attention_scores_vectorized(
    q: &Tensor,
    k_tensors: &TurboQuantTensors,
    config: &TurboQuantConfig,
) -> Result<Tensor> {
    let mse_scores = polar_attention_scores_vectorized(q, &k_tensors.mse_tensors, &config.polar_config)?;
    let qjl_scores = qjl_attention_scores_vectorized(q, &k_tensors.qjl_tensors, &config.qjl_config)?;
    
    mse_scores.add(&qjl_scores)
}
