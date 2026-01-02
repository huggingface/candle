//! PreLookahead Layer
//!
//! Processes speech token embeddings with lookahead for streaming inference.
//! Includes 2x upsampling to convert from token rate to mel rate.

use candle::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use crate::models::cosyvoice::activations::LeakyReLU;

/// PreLookahead Layer for CosyVoice3 Flow Decoder
///
/// Processes token embeddings with causal convolutions and performs
/// 2x upsampling to match mel spectrogram frame rate.
///
/// # Streaming Inference
///
/// In streaming mode, the layer uses `context` parameter:
/// - `finalize=true`: No context, uses zero padding
/// - `finalize=false`: Uses context (last `pre_lookahead_len` tokens)
#[derive(Debug)]
pub struct PreLookaheadLayer {
    conv1: Conv1d,
    conv2: Conv1d,
    lrelu: LeakyReLU,
    pre_lookahead_len: usize,
    token_mel_ratio: usize,
}

impl PreLookaheadLayer {
    /// Create a new PreLookahead layer
    ///
    /// # Arguments
    /// * `in_dim` - Input dimension (mel_dim = 80)
    /// * `hidden_dim` - Hidden dimension (1024)
    /// * `out_dim` - Output dimension (mel_dim = 80)
    /// * `pre_lookahead_len` - Lookahead length (3)
    /// * `token_mel_ratio` - Upsampling ratio (2)
    /// * `vb` - VarBuilder
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        pre_lookahead_len: usize,
        token_mel_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Conv1: (in_dim, hidden_dim, kernel=4, causal)
        // For causal, kernel=4 needs left padding=3
        let conv1_config = Conv1dConfig {
            padding: 0, // Manually handle causal padding
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = candle_nn::conv1d(in_dim, hidden_dim, 4, conv1_config, vb.pp("conv1"))?;

        // Conv2: (hidden_dim, out_dim, kernel=3, causal)
        let conv2_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv2 = candle_nn::conv1d(hidden_dim, out_dim, 3, conv2_config, vb.pp("conv2"))?;

        Ok(Self {
            conv1,
            conv2,
            lrelu: LeakyReLU::new(0.1),
            pre_lookahead_len,
            token_mel_ratio,
        })
    }

    /// Forward pass with optional context for streaming
    ///
    /// # Arguments
    /// * `x` - [B, T, mel_dim] token embeddings
    /// * `context` - Optional context tokens for streaming (last pre_lookahead_len tokens)
    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        // x: [B, T, 80]
        let input = x.clone(); // Save for residual connection
        let x = x.transpose(1, 2)?; // [B, 80, T]

        // Causal padding with optional context
        let x = match context {
            Some(ctx) => {
                // Streaming mode: use actual context tokens
                let ctx = ctx.transpose(1, 2)?; // [B, 80, pre_lookahead_len]
                Tensor::cat(&[&x, &ctx], 2)?
            }
            None => {
                // Non-streaming/finalize mode: zero padding
                let (batch, channels, _) = x.dims3()?;
                let pad = Tensor::zeros(
                    (batch, channels, self.pre_lookahead_len),
                    x.dtype(),
                    x.device(),
                )?;
                Tensor::cat(&[&x, &pad], 2)?
            }
        };

        // Conv1 with LeakyReLU
        let x = self.conv1.forward(&x)?;
        let x = self.lrelu.forward(&x)?;

        // Conv2 with causal padding (kernel_size=3, needs left padding 2)
        let (batch, channels, _) = x.dims3()?;
        let pad = Tensor::zeros((batch, channels, 2), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&pad, &x], 2)?;
        let x = self.conv2.forward(&x)?;

        // Residual + transpose back
        let x = x.transpose(1, 2)?; // [B, T, 80]
        let output = (&x + &input)?;

        // 2x upsampling (using repeat_interleave)
        self.repeat_interleave(&output, self.token_mel_ratio)
    }

    /// repeat_interleave implementation
    ///
    /// Equivalent to PyTorch: tensor.repeat_interleave(repeats, dim=1)
    fn repeat_interleave(&self, x: &Tensor, repeats: usize) -> Result<Tensor> {
        // x: [B, T, C]
        let (batch, time, channels) = x.dims3()?;
        let new_time = time * repeats;

        // [B, T, C] -> [B, T, 1, C] -> [B, T, repeats, C] -> [B, T*repeats, C]
        let x = x.unsqueeze(2)?; // [B, T, 1, C]

        // Expand
        let x = x.broadcast_as((batch, time, repeats, channels))?;

        // Reshape
        x.reshape((batch, new_time, channels))
    }

    /// Get pre_lookahead_len
    pub fn pre_lookahead_len(&self) -> usize {
        self.pre_lookahead_len
    }

    /// Get token_mel_ratio
    pub fn token_mel_ratio(&self) -> usize {
        self.token_mel_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_repeat_interleave() -> Result<()> {
        let device = Device::Cpu;

        // Create test layer (actual weights not needed to test repeat_interleave)
        let varmap = candle_nn::VarMap::new();
        let dtype = DType::F32;
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

        // Initialize required weights
        varmap.data().lock().unwrap().insert(
            "conv1.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (1024, 80, 4), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "conv1.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((1024,), dtype, &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "conv2.weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (80, 1024, 3), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "conv2.bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((80,), dtype, &device)?)?,
        );

        let layer = PreLookaheadLayer::new(80, 1024, 80, 3, 2, vb)?;

        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (1, 3, 2), // [batch=1, time=3, channels=2]
            &device,
        )?;

        let y = layer.repeat_interleave(&x, 2)?;

        assert_eq!(y.dims(), &[1, 6, 2]); // time * 2 = 6

        let y_vec: Vec<f32> = y.flatten_all()?.to_vec1()?;
        assert_eq!(
            y_vec,
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );

        Ok(())
    }
}

