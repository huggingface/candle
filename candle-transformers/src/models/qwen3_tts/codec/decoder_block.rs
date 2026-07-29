//! Decoder block: SnakeBeta → CausalTransConv upsample → 3 × ResidualUnit.

use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::{CausalConv1d, CausalTransConv1d, SnakeBeta};

/// Residual unit: SnakeBeta → dilated causal conv → SnakeBeta → 1×1 conv → residual.
pub struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::new(dim, dim, 7, dilation, vb.pp("conv1.conv"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::new(dim, dim, 1, 1, vb.pp("conv2.conv"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        a1a: Tensor, a1b: Tensor,
        c1w: Tensor, c1b: Tensor,
        a2a: Tensor, a2b: Tensor,
        c2w: Tensor, c2b: Tensor,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::from_weights(a1a, a1b)?,
            conv1: CausalConv1d::from_weights(c1w, Some(c1b), dilation)?,
            act2: SnakeBeta::from_weights(a2a, a2b)?,
            conv2: CausalConv1d::from_weights(c2w, Some(c2b), 1)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let r = x;
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        Ok((h + r)?)
    }
}

impl Module for ResidualUnit {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        ResidualUnit::forward(self, x)
    }
}

/// BigVGAN-style decoder block: upsample + 3 residual units with dilation 1, 3, 9.
pub struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalTransConv1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(in_ch: usize, out_ch: usize, rate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            snake: SnakeBeta::new(in_ch, vb.pp("block.0"))?,
            upsample: CausalTransConv1d::new(in_ch, out_ch, rate * 2, rate, vb.pp("block.1.conv"))?,
            res1: ResidualUnit::new(out_ch, 1, vb.pp("block.2"))?,
            res2: ResidualUnit::new(out_ch, 3, vb.pp("block.3"))?,
            res3: ResidualUnit::new(out_ch, 9, vb.pp("block.4"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        sa: Tensor, sb: Tensor,
        uw: Tensor, ub: Tensor,
        r1a1a: Tensor, r1a1b: Tensor, r1c1w: Tensor, r1c1b: Tensor,
        r1a2a: Tensor, r1a2b: Tensor, r1c2w: Tensor, r1c2b: Tensor,
        r2a1a: Tensor, r2a1b: Tensor, r2c1w: Tensor, r2c1b: Tensor,
        r2a2a: Tensor, r2a2b: Tensor, r2c2w: Tensor, r2c2b: Tensor,
        r3a1a: Tensor, r3a1b: Tensor, r3c1w: Tensor, r3c1b: Tensor,
        r3a2a: Tensor, r3a2b: Tensor, r3c2w: Tensor, r3c2b: Tensor,
        rate: usize,
    ) -> Result<Self> {
        Ok(Self {
            snake: SnakeBeta::from_weights(sa, sb)?,
            upsample: CausalTransConv1d::from_weights(uw, Some(ub), rate)?,
            res1: ResidualUnit::from_weights(r1a1a, r1a1b, r1c1w, r1c1b, r1a2a, r1a2b, r1c2w, r1c2b, 1)?,
            res2: ResidualUnit::from_weights(r2a1a, r2a1b, r2c1w, r2c1b, r2a2a, r2a2b, r2c2w, r2c2b, 3)?,
            res3: ResidualUnit::from_weights(r3a1a, r3a1b, r3c1w, r3c1b, r3a2a, r3a2b, r3c2w, r3c2b, 9)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let h = self.upsample.forward(&h)?;
        let h = self.res1.forward(&h)?;
        let h = self.res2.forward(&h)?;
        self.res3.forward(&h)
    }
}

impl Module for DecoderBlock {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        DecoderBlock::forward(self, x)
    }
}
