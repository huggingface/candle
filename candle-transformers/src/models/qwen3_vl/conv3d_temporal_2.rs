//! Conv3dConfig assuming a temporal patch size of 2

use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_sizes[0],
                kernel_sizes[1],
                kernel_sizes[2],
            ),
            "weight",
        )?;

        // Split on temporal dimension
        // https://github.com/pytorch/pytorch/issues/139066

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
            cudnn_fwd_algo: None,
        };

        Ok(Self {
            conv2d_1: Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;

        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}
