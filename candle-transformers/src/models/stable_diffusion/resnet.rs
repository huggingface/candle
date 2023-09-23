//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! https://arxiv.org/abs/1512.03385
use crate::models::with_tracing::{conv2d, Conv2d};
use candle::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

/// Configuration for a ResNet block.
#[derive(Debug, Clone, Copy)]
pub struct ResnetBlock2DConfig {
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    pub output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: Conv2d,
    norm2: nn::GroupNorm,
    conv2: Conv2d,
    time_emb_proj: Option<nn::Linear>,
    conv_shortcut: Option<Conv2d>,
    span: tracing::Span,
    config: ResnetBlock2DConfig,
}

impl ResnetBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        config: ResnetBlock2DConfig,
    ) -> Result<Self> {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cfg = nn::Conv2dConfig {
            stride: 1,
            padding: 1,
            groups: 1,
            dilation: 1,
        };
        let norm1 = nn::group_norm(config.groups, in_channels, config.eps, vs.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg, vs.pp("conv1"))?;
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = nn::group_norm(groups_out, out_channels, config.eps, vs.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg, vs.pp("conv2"))?;
        let use_in_shortcut = config
            .use_in_shortcut
            .unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = nn::Conv2dConfig {
                stride: 1,
                padding: 0,
                groups: 1,
                dilation: 1,
            };
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                conv_cfg,
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        let time_emb_proj = match config.temb_channels {
            None => None,
            Some(temb_channels) => Some(nn::linear(
                temb_channels,
                out_channels,
                vs.pp("time_emb_proj"),
            )?),
        };
        let span = tracing::span!(tracing::Level::TRACE, "resnet2d");
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            span,
            config,
            conv_shortcut,
        })
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => conv_shortcut.forward(xs)?,
            None => xs.clone(),
        };
        let xs = self.norm1.forward(xs)?;
        let xs = self.conv1.forward(&nn::ops::silu(&xs)?)?;
        let xs = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(time_emb_proj)) => time_emb_proj
                .forward(&nn::ops::silu(temb)?)?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
                .broadcast_add(&xs)?,
            _ => xs,
        };
        let xs = self
            .conv2
            .forward(&nn::ops::silu(&self.norm2.forward(&xs)?)?)?;
        (shortcut_xs + xs)? / self.config.output_scale_factor
    }
}
