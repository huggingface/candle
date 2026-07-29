//! Causal 1D Convolution — only looks at past context.

use candle::{Module, Result, Tensor};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, VarBuilder};

/// Causal 1D convolution with left-side zero-padding.
///
/// Output at position *t* depends only on inputs at positions ≤ *t*.
pub struct CausalConv1d {
    conv: Conv1d,
    causal_padding: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel_size,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            },
            vb,
        )?;
        Ok(Self {
            conv,
            causal_padding: dilation * (kernel_size - 1),
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, dilation: usize) -> Result<Self> {
        Self::from_weights_grouped(weight, bias, dilation, 1)
    }

    pub fn from_weights_grouped(
        weight: Tensor,
        bias: Option<Tensor>,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let kernel_size = weight.dim(2)?;
        let causal_padding = dilation * (kernel_size - 1);
        let conv = Conv1d::new(
            weight,
            bias,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation,
                groups,
                ..Default::default()
            },
        );
        Ok(Self { conv, causal_padding })
    }
}

impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        // Pad left side only
        let padded = x.pad_with_zeros(candle::D::Minus1, self.causal_padding, 0)?;
        self.conv.forward(&padded)
    }
}
