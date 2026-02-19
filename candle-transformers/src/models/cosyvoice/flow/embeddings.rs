//! Embedding layers for DiT
//!
//! Implements various embedding layers used in the Diffusion Transformer:
//! - TimestepEmbedding: Sinusoidal timestep embedding
//! - AdaLayerNormZero: Adaptive layer normalization with modulation
//! - InputEmbedding: Combines multiple inputs for DiT

use candle::{DType, Device, Module, Result, Tensor, D};
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
    /// Matches Python's SinusPositionEmbedding with scale=1000
    fn sinusoidal_embedding(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let dtype = t.dtype();
        let half_dim = self.sinusoidal_dim / 2;

        // Python: emb = math.log(10000) / (half_dim - 1)
        //         emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        let log_10000 = 10000f64.ln();
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f64) * log_10000 / (half_dim as f64 - 1.0)).exp() as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        // t: [B] -> [B, 1]
        let t = t.unsqueeze(D::Minus1)?;
        let t = t.to_dtype(DType::F32)?;

        // Python: emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        // scale = 1000
        let scale = 1000.0;
        let freqs = (t * scale)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

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
    pub fn forward(
        &self,
        x: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
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
        let x_modulated = x_norm
            .broadcast_mul(&scale_factor)?
            .broadcast_add(&shift_msa_expanded)?;

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

        // Python: scale, shift = torch.chunk(emb, 2, dim=1)
        let chunks = emb.chunk(2, D::Minus1)?;
        let scale = chunks[0].clone().unsqueeze(1)?;
        let shift = chunks[1].clone().unsqueeze(1)?;

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
            padding: 0, // Manual causal padding
            stride: 1,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        };
        // Weight path: conv1.0 and conv2.0 (the .0 is from nn.Sequential)
        let conv1 = candle_nn::conv1d(dim, dim, kernel_size, conv_config, vb.pp("conv1.0"))?;
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

        // Causal padding (left side only): F.pad(x, (kernel_size - 1, 0, 0, 0))
        // In PyTorch, F.pad for 1D is (left, right) for the last dimension
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
        let conv_pos_embed =
            CausalConvPositionEmbedding::new(out_dim, 31, 16, vb.pp("conv_pos_embed"))?;

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
    pub fn forward(&self, x: &Tensor, cond: &Tensor, mu: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Expand spks to sequence length: [B, spk_dim] -> [B, T, spk_dim]
        let spks = spks.unsqueeze(1)?;
        let spks = spks.broadcast_as((batch, seq_len, self.spk_dim))?;

        // Concatenate all inputs
        let combined = Tensor::cat(&[x, cond, mu, &spks], D::Minus1)?;

        // Project and add position embedding (residual)
        let x = self.proj.forward(&combined)?;
        let pos_embed = self.conv_pos_embed.forward(&x)?;
        x + pos_embed
    }
}

/// Rotary Position Embedding for DiT
///
/// Matches x_transformers.RotaryEmbedding implementation.
/// Returns freqs tensor that will be used with cos/sin in attention.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    dim: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f64) -> Self {
        Self { dim, theta }
    }

    /// Generate RoPE frequencies
    ///
    /// Returns freqs tensor of shape [1, seq_len, dim]
    /// This matches x_transformers.RotaryEmbedding.forward_from_seq_len
    pub fn forward(&self, seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let half_dim = self.dim / 2;

        // Calculate inverse frequencies: 1 / (theta^(2i/dim))
        // This matches: inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / self.theta.powf(2.0 * i as f64 / self.dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        // Position indices: [0, 1, 2, ..., seq_len-1]
        let t = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;

        // freqs = einsum('i, j -> i j', t, inv_freq) = outer product
        // [seq_len] * [half_dim] -> [seq_len, half_dim]
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Stack to get [seq_len, dim]: freqs = stack((freqs, freqs), dim=-1).reshape(...)
        // This duplicates each frequency: [f0, f0, f1, f1, ...]
        // Python: freqs = stack((freqs, freqs), dim = -1)
        //         freqs = rearrange(freqs, '... d r -> ... (d r)')
        // This means: [seq_len, half_dim] -> [seq_len, half_dim, 2] -> [seq_len, dim]
        // Result pattern: [f0, f0, f1, f1, f2, f2, ...]
        let freqs = freqs.unsqueeze(D::Minus1)?; // [seq_len, half_dim, 1]
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [seq_len, half_dim, 2]
        let freqs = freqs.reshape((seq_len, self.dim))?; // [seq_len, dim]

        // Add batch dimension: [1, seq_len, dim]
        let freqs = freqs.unsqueeze(0)?;

        freqs.to_dtype(dtype)
    }
}

/// Apply rotary position embedding to a 3D tensor (before reshape to heads)
///
/// Matches Python's apply_rotary_pos_emb function.
/// t: [batch, seq_len, dim] - query or key tensor
/// freqs: [1, seq_len, rot_dim] - frequency tensor
///
/// Returns: [batch, seq_len, dim] with rotary embedding applied to first rot_dim dimensions
pub fn apply_rotary_pos_emb_3d(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let rot_dim = freqs.dim(D::Minus1)?;
    let seq_len = t.dim(1)?;
    let orig_dtype = t.dtype();

    // Get freqs for this sequence length
    let freqs = if freqs.dim(1)? > seq_len {
        freqs.narrow(1, freqs.dim(1)? - seq_len, seq_len)?
    } else {
        freqs.clone()
    };

    // Partial rotary: only rotate first rot_dim dimensions
    let t_rot = t.narrow(D::Minus1, 0, rot_dim)?;
    let t_pass = if t.dim(D::Minus1)? > rot_dim {
        Some(t.narrow(D::Minus1, rot_dim, t.dim(D::Minus1)? - rot_dim)?)
    } else {
        None
    };

    // Compute cos and sin
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // rotate_half
    let t_rot_half = rotate_half(&t_rot)?;

    // Apply rotation: t * cos + rotate_half(t) * sin
    let t_out = (t_rot.broadcast_mul(&cos)? + t_rot_half.broadcast_mul(&sin)?)?;

    // Concatenate with unrotated part if exists
    let out = match t_pass {
        Some(pass) => Tensor::cat(&[&t_out, &pass], D::Minus1)?,
        None => t_out,
    };

    out.to_dtype(orig_dtype)
}

/// Apply rotary position embedding to query and key (4D tensors after reshape)
///
/// Matches x_transformers apply_rotary_pos_emb function.
/// freqs: [1, seq_len, dim] or [batch, seq_len, dim]
/// q, k: [batch, heads, seq_len, head_dim]
pub fn apply_rotary_emb(q: &Tensor, k: &Tensor, freqs: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch, _heads, seq_len, head_dim) = q.dims4()?;
    let rot_dim = freqs.dim(D::Minus1)?;

    // Get freqs for this sequence length
    let freqs = if freqs.dim(1)? > seq_len {
        freqs.narrow(1, freqs.dim(1)? - seq_len, seq_len)?
    } else {
        freqs.clone()
    };

    // Expand freqs to [batch, 1, seq_len, rot_dim] for broadcasting
    let freqs = if freqs.dim(0)? == 1 && batch > 1 {
        freqs.broadcast_as((batch, 1, seq_len, rot_dim))?
    } else {
        freqs.unsqueeze(1)? // [batch, 1, seq_len, dim]
    };

    // Partial rotary: only rotate first rot_dim dimensions
    let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
    let q_pass = if head_dim > rot_dim {
        Some(q.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?)
    } else {
        None
    };

    let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
    let k_pass = if head_dim > rot_dim {
        Some(k.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?)
    } else {
        None
    };

    // Compute cos and sin
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // rotate_half: [-x2, x1, -x4, x3, ...]
    // For [f0, f0, f1, f1, ...] pattern, we need to rotate pairs
    let q_rot_half = rotate_half(&q_rot)?;
    let k_rot_half = rotate_half(&k_rot)?;

    // Apply rotation: t * cos + rotate_half(t) * sin
    let q_out = (q_rot.broadcast_mul(&cos)? + q_rot_half.broadcast_mul(&sin)?)?;
    let k_out = (k_rot.broadcast_mul(&cos)? + k_rot_half.broadcast_mul(&sin)?)?;

    // Concatenate with unrotated part if exists
    let q_out = match q_pass {
        Some(pass) => Tensor::cat(&[&q_out, &pass], D::Minus1)?,
        None => q_out,
    };
    let k_out = match k_pass {
        Some(pass) => Tensor::cat(&[&k_out, &pass], D::Minus1)?,
        None => k_out,
    };

    Ok((q_out, k_out))
}

/// Rotate half of the tensor
///
/// Python implementation:
/// x = rearrange(x, '... (d r) -> ... d r', r = 2)
/// x1, x2 = x.unbind(dim = -1)
/// x = stack((-x2, x1), dim = -1)
/// return rearrange(x, '... d r -> ... (d r)')
///
/// This means: [a0, a1, b0, b1, ...] -> [-a1, a0, -b1, b0, ...]
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let shape = x.dims();
    let last_dim = shape[shape.len() - 1];
    let half = last_dim / 2;

    // Reshape to [..., half, 2]
    let mut new_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    new_shape.push(half);
    new_shape.push(2);
    let x = x.reshape(new_shape)?;

    // Split into x1, x2 (unbind on last dim)
    let x1 = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // [..., half]
    let x2 = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?; // [..., half]

    // Stack (-x2, x1) on last dim
    let neg_x2 = x2.neg()?;
    let rotated = Tensor::stack(&[&neg_x2, &x1], D::Minus1)?; // [..., half, 2]

    // Reshape back to [..., dim]
    let mut final_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    final_shape.push(last_dim);
    rotated.reshape(final_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding() -> Result<()> {
        let device = Device::Cpu;

        let rope_emb = RotaryEmbedding::new(64, 10000.0);
        let rope = rope_emb.forward(10, &device, DType::F32)?;

        // Should be [1, seq_len, dim] to match x_transformers
        assert_eq!(rope.dims(), &[1, 10, 64]);
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
