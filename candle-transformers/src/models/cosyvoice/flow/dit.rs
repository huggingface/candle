//! Diffusion Transformer (DiT) for CosyVoice3
//!
//! Implements the DiT architecture for flow-based speech synthesis.
//! Based on "Scalable Diffusion Models with Transformers" paper.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::embeddings::{
    apply_rotary_pos_emb_3d, AdaLayerNormZero, AdaLayerNormZeroFinal, InputEmbedding,
    RotaryEmbedding, TimestepEmbedding,
};
use crate::models::cosyvoice::config::DiTConfig;

/// Feed-Forward Network
#[derive(Debug)]
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, ff_mult: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = dim * ff_mult;
        // Weight paths match official implementation: ff.ff.0.0 and ff.ff.2
        let linear1 = candle_nn::linear(dim, hidden_dim, vb.pp("ff.0.0"))?;
        let linear2 = candle_nn::linear(hidden_dim, dim, vb.pp("ff.2"))?;

        Ok(Self { linear1, linear2 })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu_erf()?; // GELU activation
        self.linear2.forward(&x)
    }
}

/// Multi-Head Attention for DiT
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(dim: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let to_q = candle_nn::linear(dim, num_heads * head_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(dim, num_heads * head_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(dim, num_heads * head_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(num_heads * head_dim, dim, vb.pp("to_out.0"))?;

        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, rope: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // QKV projections
        let q = self.to_q.forward(x)?; // [B, T, H*D]
        let k = self.to_k.forward(x)?; // [B, T, H*D]
        let v = self.to_v.forward(x)?; // [B, T, H*D]

        // Apply RoPE BEFORE reshape (matching Python implementation)
        // Python: query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
        // where query is [B, T, H*D] and freqs is [1, T, rot_dim]
        let q = apply_rotary_pos_emb_3d(&q, rope)?;
        let k = apply_rotary_pos_emb_3d(&k, rope)?;

        // Reshape to [B, H, T, D]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        // Apply mask if provided
        let attn_weights = match mask {
            Some(m) => {
                let mask = m.unsqueeze(1)?; // [B, 1, T, T]
                let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?;
                attn_weights.where_cond(&mask.broadcast_as(attn_weights.shape())?, &neg_inf)?
            }
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.to_out.forward(&attn_output)
    }
}

/// DiT Block
#[derive(Debug)]
pub struct DiTBlock {
    attn_norm: AdaLayerNormZero,
    attn: Attention,
    ff_norm: LayerNormNoAffine,
    ff: FeedForward,
}

/// LayerNorm without learnable parameters (elementwise_affine=False)
#[derive(Debug, Clone)]
struct LayerNormNoAffine {
    eps: f64,
}

impl LayerNormNoAffine {
    fn new() -> Self {
        Self { eps: 1e-6 }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        x_centered.broadcast_div(&std)
    }
}

impl DiTBlock {
    pub fn new(config: &DiTConfig, vb: VarBuilder) -> Result<Self> {
        let attn_norm = AdaLayerNormZero::new(config.dim, vb.pp("attn_norm"))?;
        let attn = Attention::new(config.dim, config.heads, config.dim_head, vb.pp("attn"))?;
        let ff_norm = LayerNormNoAffine::new();
        let ff = FeedForward::new(config.dim, config.ff_mult, vb.pp("ff"))?;

        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        mask: Option<&Tensor>,
        rope: &Tensor,
    ) -> Result<Tensor> {
        // AdaLN modulation
        let (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.attn_norm.forward(x, t_emb)?;

        // Self-attention with RoPE
        let attn_out = self.attn.forward(&norm_x, mask, rope)?;
        let x = (x + gate_msa.unsqueeze(1)?.broadcast_mul(&attn_out)?)?;

        // FeedForward with modulation
        // Python: ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        let ff_norm = self.ff_norm.forward(&x)?;
        let scale_mlp = scale_mlp.unsqueeze(1)?;
        let shift_mlp = shift_mlp.unsqueeze(1)?;
        let scale_factor = (scale_mlp + 1.0)?;
        let ff_input = ff_norm
            .broadcast_mul(&scale_factor)?
            .broadcast_add(&shift_mlp)?;
        let ff_out = self.ff.forward(&ff_input)?;

        x + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff_out)?
    }
}

/// Diffusion Transformer
#[derive(Debug)]
pub struct DiT {
    config: DiTConfig,
    time_embed: TimestepEmbedding,
    input_embed: InputEmbedding,
    rotary_embed: RotaryEmbedding,
    transformer_blocks: Vec<DiTBlock>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
}

impl DiT {
    pub fn new(config: DiTConfig, vb: VarBuilder) -> Result<Self> {
        let time_embed = TimestepEmbedding::new(config.dim, vb.pp("time_embed"))?;
        let input_embed = InputEmbedding::new(
            config.mel_dim,
            config.spk_dim,
            config.dim,
            vb.pp("input_embed"),
        )?;
        let rotary_embed = RotaryEmbedding::new(config.dim_head, 10000.0);

        let mut transformer_blocks = Vec::new();
        for i in 0..config.depth {
            let block = DiTBlock::new(&config, vb.pp(format!("transformer_blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        let norm_out = AdaLayerNormZeroFinal::new(config.dim, vb.pp("norm_out"))?;
        let proj_out = candle_nn::linear(config.dim, config.mel_dim, vb.pp("proj_out"))?;

        Ok(Self {
            config,
            time_embed,
            input_embed,
            rotary_embed,
            transformer_blocks,
            norm_out,
            proj_out,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - [B, 80, T] noised mel (transposed to [B, T, 80] internally)
    /// * `mask` - [B, 1, T] mask
    /// * `mu` - [B, 80, T] condition from token embedding
    /// * `t` - [B] timestep
    /// * `spks` - [B, 80] speaker embedding
    /// * `cond` - [B, 80, T] prompt mel condition
    /// * `streaming` - Whether in streaming mode
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        _mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let (_batch, _, seq_len) = x.dims3()?;

        // Transpose to [B, T, C]
        let x = x.transpose(1, 2)?;
        let mu = mu.transpose(1, 2)?;
        let cond = cond.transpose(1, 2)?;

        // Time embedding
        let t_emb = self.time_embed.forward(t)?; // [B, dim]

        // Input embedding (concat x, cond, mu, spks and project)
        let x = self.input_embed.forward(&x, &cond, &mu, spks)?;

        // RoPE
        let rope = self.rotary_embed.forward(seq_len, x.device(), x.dtype())?;

        // Attention mask (causal for streaming)
        let attn_mask = if streaming {
            Some(self.create_chunk_mask(seq_len, x.device())?)
        } else {
            None
        };

        // Transformer blocks
        let mut x = x;
        for block in &self.transformer_blocks {
            x = block.forward(&x, &t_emb, attn_mask.as_ref(), &rope)?;
        }

        // Final norm and projection
        let x = self.norm_out.forward(&x, &t_emb)?;
        let output = self.proj_out.forward(&x)?;

        output.transpose(1, 2) // [B, 80, T]
    }

    /// Create chunk-based causal mask for streaming
    fn create_chunk_mask(&self, size: usize, device: &Device) -> Result<Tensor> {
        let chunk_size = self.config.static_chunk_size;

        // Create position indices
        let pos = Tensor::arange(0u32, size as u32, device)?;

        // block_value[i] = ((i / chunk_size) + 1) * chunk_size
        let chunk_idx = (pos.to_dtype(DType::F32)? / chunk_size as f64)?;
        let chunk_idx = chunk_idx.floor()?;
        let block_value = ((chunk_idx + 1.0)? * chunk_size as f64)?;

        // mask[i, j] = j < block_value[i]
        let pos_j = Tensor::arange(0f32, size as f32, device)?;
        let pos_j = pos_j.unsqueeze(0)?; // [1, size]
        let block_value = block_value.unsqueeze(1)?; // [size, 1]

        pos_j.broadcast_lt(&block_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

        // Initialize weights
        varmap.data().lock().unwrap().insert(
            "ff.0.0.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (2048, 1024), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "ff.0.0.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((2048,), dtype, &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "ff.2.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (1024, 2048), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "ff.2.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((1024,), dtype, &device)?)?,
        );

        let ff = FeedForward::new(1024, 2, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 10, 1024), &device)?;
        let y = ff.forward(&x)?;

        assert_eq!(y.dims(), &[2, 10, 1024]);
        Ok(())
    }
}
