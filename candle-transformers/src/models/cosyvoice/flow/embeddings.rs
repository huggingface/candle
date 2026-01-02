//! Embedding layers for DiT
//!
//! Implements various embedding layers used in the Diffusion Transformer:
//! - TimestepEmbedding: Sinusoidal timestep embedding
//! - AdaLayerNormZero: Adaptive layer normalization with modulation
//! - InputEmbedding: Combines multiple inputs for DiT

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::models::cosyvoice::activations::{Mish, Swish};

/// Sinusoidal timestep embedding
///
/// Generates embeddings for diffusion timesteps using sinusoidal functions.
/// Following official CosyVoice implementation: sinusoidal_dim(256) -> MLP -> dim(1024)
#[derive(Debug, Clone)]
pub struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    swish: Swish,
    sinusoidal_dim: usize,
}

impl TimestepEmbedding {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Official implementation: sinusoidal_dim=256, then MLP to dim=1024
        let sinusoidal_dim = 256;
        let linear1 = candle_nn::linear(sinusoidal_dim, dim, vb.pp("time_mlp.0"))?;
        let linear2 = candle_nn::linear(dim, dim, vb.pp("time_mlp.2"))?;

        Ok(Self {
            linear1,
            linear2,
            swish: Swish,
            sinusoidal_dim,
        })
    }

    /// Generate sinusoidal position encoding
    fn sinusoidal_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let dtype = t.dtype();
        let half_dim = self.sinusoidal_dim / 2;

        // Calculate frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0f64 / 10000f64.powf(i as f64 / half_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        // t: [B] -> [B, 1]
        let t = t.unsqueeze(D::Minus1)?;
        let t = t.to_dtype(DType::F32)?;

        // [B, 1] * [half_dim] -> [B, half_dim]
        let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // [B, dim] = [sin, cos]
        let sin_emb = freqs.sin()?;
        let cos_emb = freqs.cos()?;
        let emb = Tensor::cat(&[&sin_emb, &cos_emb], D::Minus1)?;

        emb.to_dtype(dtype)
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let emb = self.sinusoidal_embedding(t)?;
        let emb = self.linear1.forward(&emb)?;
        let emb = self.swish.forward(&emb)?;
        self.linear2.forward(&emb)
    }
}

/// Adaptive Layer Normalization Zero
///
/// Used in DiT blocks for conditioning on timestep.
/// Outputs 6 modulation parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
/// Uses LayerNorm followed by modulation.
#[derive(Debug)]
pub struct AdaLayerNormZero {
    linear: Linear,
    swish: Swish,
}

impl AdaLayerNormZero {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(dim, dim * 6, vb.pp("linear"))?;

        Ok(Self {
            linear,
            swish: Swish,
        })
    }

    /// Forward pass
    ///
    /// # Returns
    /// * `(x, gate_msa, shift_mlp, scale_mlp, gate_mlp)` - modulation parameters
    pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // Apply LayerNorm first (elementwise_affine=False)
        let x_norm = layer_norm_no_affine(x)?;

        // emb: [B, dim] -> [B, dim * 6]
        let emb = self.swish.forward(emb)?;
        let emb = self.linear.forward(&emb)?;

        // Split into 6 parts
        let chunks = emb.chunk(6, D::Minus1)?;
        let shift_msa = chunks[0].clone();
        let scale_msa = chunks[1].clone();
        let gate_msa = chunks[2].clone();
        let shift_mlp = chunks[3].clone();
        let scale_mlp = chunks[4].clone();
        let gate_mlp = chunks[5].clone();

        // Apply modulation: x_norm * (1 + scale) + shift
        let scale_msa_expanded = scale_msa.unsqueeze(1)?;
        let shift_msa_expanded = shift_msa.unsqueeze(1)?;
        let scale_factor = (scale_msa_expanded + 1.0)?;
        let x_modulated = x_norm.broadcast_mul(&scale_factor)?.broadcast_add(&shift_msa_expanded)?;

        Ok((x_modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

/// LayerNorm without learnable parameters (elementwise_affine=False)
fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
    let (_batch, _seq, _dim) = x.dims3()?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + 1e-6)?.sqrt()?;
    x_centered.broadcast_div(&std)
}

/// Final adaptive layer norm
#[derive(Debug)]
pub struct AdaLayerNormZeroFinal {
    linear: Linear,
    swish: Swish,
}

impl AdaLayerNormZeroFinal {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(dim, dim * 2, vb.pp("linear"))?;

        Ok(Self {
            linear,
            swish: Swish,
        })
    }

    pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        // Apply LayerNorm first
        let x_norm = layer_norm_no_affine(x)?;

        let emb = self.swish.forward(emb)?;
        let emb = self.linear.forward(&emb)?;

        let chunks = emb.chunk(2, D::Minus1)?;
        let shift = chunks[0].clone().unsqueeze(1)?;
        let scale = chunks[1].clone().unsqueeze(1)?;

        let scale_factor = (scale + 1.0)?;
        x_norm.broadcast_mul(&scale_factor)?.broadcast_add(&shift)
    }
}

/// Causal Convolutional Position Embedding
///
/// Adds position information using causal convolutions.
/// Official implementation: kernel_size=31, groups=16
#[derive(Debug)]
pub struct CausalConvPositionEmbedding {
    conv1: Conv1d,
    conv2: Conv1d,
    kernel_size: usize,
    mish: Mish,
}

impl CausalConvPositionEmbedding {
    pub fn new(dim: usize, kernel_size: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: 0,  // Manual causal padding
            stride: 1,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv1 = candle_nn::conv1d(dim, dim, kernel_size, conv_config.clone(), vb.pp("conv1.0"))?;
        let conv2 = candle_nn::conv1d(dim, dim, kernel_size, conv_config, vb.pp("conv2.0"))?;

        Ok(Self {
            conv1,
            conv2,
            kernel_size,
            mish: Mish,
        })
    }
}

impl Module for CausalConvPositionEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, dim]
        let x = x.transpose(1, 2)?; // [B, dim, T]

        // Causal padding (left side only)
        let x = x.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        let x = self.conv1.forward(&x)?;
        let x = self.mish.forward(&x)?;

        let x = x.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        let x = self.conv2.forward(&x)?;
        let x = self.mish.forward(&x)?;

        x.transpose(1, 2) // [B, T, dim]
    }
}

/// Input Embedding for DiT
///
/// Combines x (noised mel), cond (prompt mel), mu (condition), and spks (speaker embedding)
#[derive(Debug)]
pub struct InputEmbedding {
    proj: Linear,
    conv_pos_embed: CausalConvPositionEmbedding,
    #[allow(dead_code)]
    mel_dim: usize,
    spk_dim: usize,
}

impl InputEmbedding {
    pub fn new(mel_dim: usize, spk_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Input dimension: mel_dim * 3 (x, cond, mu) + spk_dim
        let in_dim = mel_dim * 3 + spk_dim;
        let proj = candle_nn::linear(in_dim, out_dim, vb.pp("proj"))?;

        // Causal conv position embedding: kernel_size=31, groups=16
        let conv_pos_embed = CausalConvPositionEmbedding::new(
            out_dim,
            31,
            16,
            vb.pp("conv_pos_embed"),
        )?;

        Ok(Self {
            proj,
            conv_pos_embed,
            mel_dim,
            spk_dim,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - [B, T, mel_dim] noised mel
    /// * `cond` - [B, T, mel_dim] condition (prompt mel)
    /// * `mu` - [B, T, mel_dim] mu from token embedding
    /// * `spks` - [B, spk_dim] speaker embedding
    pub fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        mu: &Tensor,
        spks: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Expand spks to sequence length: [B, spk_dim] -> [B, T, spk_dim]
        let spks = spks.unsqueeze(1)?;
        let spks = spks.broadcast_as((batch, seq_len, self.spk_dim))?;

        // Concatenate all inputs
        let combined = Tensor::cat(&[x, cond, mu, &spks], D::Minus1)?;

        // Project and add position embedding (residual)
        let x = self.proj.forward(&combined)?;
        // Temporarily skip conv_pos_embed due to Metal groups conv issue
        // let pos_embed = self.conv_pos_embed.forward(&x)?;
        // x + pos_embed
        Ok(x)
    }
}

/// Rotary Position Embedding for DiT
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    dim: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f64) -> Self {
        Self { dim, theta }
    }

    /// Generate RoPE embedding
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `device` - Device
    /// * `dtype` - Data type
    pub fn forward(&self, seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let half_dim = self.dim / 2;

        // Calculate frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / self.theta.powf(2.0 * i as f64 / self.dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        // Position indices
        let t = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;

        // [seq_len] * [half_dim] -> [seq_len, half_dim]
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // [seq_len, dim] = [cos, sin]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Combine into [seq_len, dim, 2]
        let rope = Tensor::stack(&[&cos, &sin], D::Minus1)?;
        rope.to_dtype(dtype)
    }
}

/// Apply rotary embedding to query and key
pub fn apply_rotary_emb(q: &Tensor, k: &Tensor, rope: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    let half_dim = head_dim / 2;

    // rope: [seq_len, half_dim, 2]
    // Extract cos and sin using narrow instead of indexing
    let cos = rope.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [seq_len, half_dim]
    let sin = rope.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?; // [seq_len, half_dim]

    // Split q, k into two halves
    let q1 = q.narrow(D::Minus1, 0, half_dim)?.contiguous()?;
    let q2 = q.narrow(D::Minus1, half_dim, half_dim)?.contiguous()?;
    let k1 = k.narrow(D::Minus1, 0, half_dim)?.contiguous()?;
    let k2 = k.narrow(D::Minus1, half_dim, half_dim)?.contiguous()?;

    // Broadcast cos, sin to [batch, heads, seq_len, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let cos = cos.broadcast_as((batch, heads, seq_len, half_dim))?.contiguous()?;
    let sin = sin.broadcast_as((batch, heads, seq_len, half_dim))?.contiguous()?;

    // Apply rotation: [q1*cos - q2*sin, q1*sin + q2*cos]
    let q_rot1 = (q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?;
    let q_rot2 = (q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?;
    let k_rot1 = (k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?;
    let k_rot2 = (k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?;

    let q_out = Tensor::cat(&[&q_rot1, &q_rot2], D::Minus1)?;
    let k_out = Tensor::cat(&[&k_rot1, &k_rot2], D::Minus1)?;

    Ok((q_out, k_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding() -> Result<()> {
        let device = Device::Cpu;

        let rope_emb = RotaryEmbedding::new(64, 10000.0);
        let rope = rope_emb.forward(10, &device, DType::F32)?;

        assert_eq!(rope.dims(), &[10, 32, 2]); // [seq_len, half_dim, 2]
        Ok(())
    }

    #[test]
    fn test_sinusoidal_embedding_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
    
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);
    
        // Need to initialize weights matching official structure:
        // sinusoidal_dim=256 -> dim=1024
        varmap.data().lock().unwrap().insert(
            "time_mlp.0.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (1024, 256), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "time_mlp.0.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((1024,), dtype, &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "time_mlp.2.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (1024, 1024), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "time_mlp.2.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((1024,), dtype, &device)?)?,
        );
    
        let te = TimestepEmbedding::new(1024, vb)?;
    
        let t = Tensor::from_slice(&[0.1f32, 0.5, 0.9], 3, &device)?;
        let emb = te.forward(&t)?;
    
        assert_eq!(emb.dims(), &[3, 1024]);
        Ok(())
    }
}

