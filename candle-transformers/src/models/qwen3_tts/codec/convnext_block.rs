//! ConvNeXt block used in pre-upsampling stages of the 12 Hz decoder.

use candle::{Module, Result, Tensor};
use candle_nn::{
    layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

use super::CausalConv1d;

/// ConvNeXt block: depthwise-conv → LayerNorm → expand → GELU → project → γ-scale → residual.
pub struct ConvNeXtBlock {
    pub(super) dwconv: CausalConv1d,
    pub(super) norm: LayerNorm,
    pub(super) pwconv1: Linear,
    pub(super) pwconv2: Linear,
    pub(super) gamma: Tensor,
}

impl ConvNeXtBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, vb.pp("dwconv.conv"))?;
        let norm = layer_norm(
            dim,
            LayerNormConfig { eps: 1e-6, ..Default::default() },
            vb.pp("norm"),
        )?;
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get((dim,), "gamma")?;
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma })
    }

    pub fn from_weights(
        dw_w: Tensor,
        dw_b: Option<Tensor>,
        norm_w: Tensor,
        norm_b: Tensor,
        pw1_w: Tensor,
        pw1_b: Tensor,
        pw2_w: Tensor,
        pw2_b: Tensor,
        gamma: Tensor,
    ) -> Result<Self> {
        let dwconv = CausalConv1d::from_weights_grouped(dw_w, dw_b, 1, gamma.dim(0)?)?;
        let norm = LayerNorm::new(norm_w, norm_b, 1e-6);
        let pwconv1 = Linear::new(pw1_w, Some(pw1_b));
        let pwconv2 = Linear::new(pw2_w, Some(pw2_b));
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.dwconv.forward(x)?; // [B, C, T]
        let h = h.transpose(1, 2)?; // [B, T, C]
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        // Apply gamma scale: [C] → [1, 1, C]
        let h = h.broadcast_mul(&self.gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        let h = h.transpose(1, 2)?; // [B, C, T]
        Ok((h + residual)?)
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        ConvNeXtBlock::forward(self, x)
    }
}
