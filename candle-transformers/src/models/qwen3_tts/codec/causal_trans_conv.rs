//! Causal Transposed 1D Convolution — used for upsampling in the decoder.

use candle::{Module, Result, Tensor, D};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Rolling overlap-add carry buffer for one [`CausalTransConv1d`] during
/// streaming decode.
///
/// Each upsampled single-frame output is `stride` samples long in batch mode,
/// but the transposed convolution actually produces `kernel_size` samples and
/// the batch path trims the last `kernel_size - stride` from the right.  In
/// streaming those trimmed samples must be carried over and added to the
/// leading samples of the next frame.
#[derive(Default)]
pub struct CausalTransConv1dState {
    /// `[B, C_out, right_trim]` overlap-add carry, or `None` before first frame.
    pub carry: Option<Tensor>,
}

/// Causal transposed 1D convolution.
///
/// Applies `ConvTranspose1d` then trims `kernel_size - stride` samples from
/// the right to maintain causal length (matching the official Qwen3-TTS impl).
pub struct CausalTransConv1d {
    conv: ConvTranspose1d,
    right_trim: usize,
    stride: usize,
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
            stride,
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
            stride,
        })
    }

    /// Streaming forward: upsample one input frame and update `state`.
    ///
    /// For an input of `T` frames the output is `T * stride` samples.  The
    /// `right_trim` overlap-add carry from the previous call is added to the
    /// front of the output, and the new trailing carry is saved in `state`.
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        state: &mut CausalTransConv1dState,
    ) -> Result<Tensor> {
        // Full transposed-conv output: [B, C_out, T*kernel_size]
        let raw = self.conv.forward(x)?;
        let raw_len = raw.dim(D::Minus1)?;
        // Samples we emit this call
        let emit_len = raw_len - self.right_trim;
        let emit = raw.narrow(D::Minus1, 0, emit_len)?;
        // Overlap-add: add carry from previous call to the leading right_trim
        // samples of `emit`.
        let out = if let Some(carry) = &state.carry {
            let head = (emit.narrow(D::Minus1, 0, self.right_trim)? + carry)?;
            let tail = emit.narrow(D::Minus1, self.right_trim, emit_len - self.right_trim)?;
            Tensor::cat(&[&head, &tail], D::Minus1)?
        } else {
            emit
        };
        // Save new carry
        state.carry = if self.right_trim > 0 {
            Some(raw.narrow(D::Minus1, emit_len, self.right_trim)?)
        } else {
            None
        };
        Ok(out)
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
