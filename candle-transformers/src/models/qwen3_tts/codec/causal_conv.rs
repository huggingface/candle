//! Causal 1D Convolution — only looks at past context.

use candle::{Module, Result, Tensor, D};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, VarBuilder};

/// Rolling left-context buffer for one [`CausalConv1d`] during streaming decode.
///
/// Initialise with [`CausalConv1dState::default`]; the buffer is created lazily
/// on the first [`CausalConv1d::forward_with_state`] call (zero-initialised,
/// identical to the zero-padding used in batch mode).
#[derive(Default)]
pub struct CausalConv1dState {
    /// `[B, C_in, causal_padding]`, or `None` before the first frame.
    pub buf: Option<Tensor>,
}

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

    /// Number of left-context samples this conv requires.
    #[inline]
    pub fn causal_padding(&self) -> usize {
        self.causal_padding
    }

    /// Streaming forward: process `x` (any `T ≥ 1`) and update `state`.
    ///
    /// On the first call `state.buf` is `None` and left-context is zero-padded
    /// (identical to [`Module::forward`] on the first batch-mode call).
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        state: &mut CausalConv1dState,
    ) -> Result<Tensor> {
        // Build padded input: [left_ctx | x]  shape [B, C, causal_padding + T]
        let padded = match &state.buf {
            Some(buf) => Tensor::cat(&[buf, x], D::Minus1)?,
            None => x.pad_with_zeros(D::Minus1, self.causal_padding, 0)?,
        };
        let out = self.conv.forward(&padded)?;
        // New buffer = last causal_padding frames of the padded input
        state.buf = if self.causal_padding > 0 {
            let total = padded.dim(D::Minus1)?;
            Some(padded.narrow(D::Minus1, total - self.causal_padding, self.causal_padding)?)
        } else {
            None
        };
        Ok(out)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        // Pad left side only
        let padded = x.pad_with_zeros(candle::D::Minus1, self.causal_padding, 0)?;
        self.conv.forward(&padded)
    }
}
