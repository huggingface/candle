//! Causal Transposed 1D Convolution — used for upsampling in the decoder.

use candle::{Module, Result, Tensor};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Causal transposed 1D convolution.
///
/// Applies `ConvTranspose1d` then trims `kernel_size - stride` samples from
/// the right to maintain causal length (matching the official Qwen3-TTS impl).
pub struct CausalTransConv1d {
    conv: ConvTranspose1d,
    right_trim: usize,
}

impl CausalTransConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            ConvTranspose1dConfig {
                padding: 0,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
            vb,
        )?;
        Ok(Self {
            conv,
            right_trim: kernel_size.saturating_sub(stride),
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, stride: usize) -> Result<Self> {
        let kernel_size = weight.dim(2)?;
        let conv = ConvTranspose1d::new(
            weight,
            bias,
            ConvTranspose1dConfig {
                padding: 0,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
        );
        Ok(Self {
            conv,
            right_trim: kernel_size.saturating_sub(stride),
        })
    }
}

impl Module for CausalTransConv1d {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let out = self.conv.forward(x)?;
        if self.right_trim == 0 {
            return Ok(out);
        }
        let len = out.dim(candle::D::Minus1)?;
        if len <= self.right_trim {
            return Ok(out);
        }
        out.narrow(candle::D::Minus1, 0, len - self.right_trim)
    }
}
