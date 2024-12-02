//! # MobileNet-v4
//!
//! MobileNet-v4 inference implementation based on timm.
//!
//! ## Paper
//!
//! ["MobileNetV4 - Universal Models for the Mobile Ecosystem"](https://arxiv.org/abs/2404.10518)
//!
//! ## References
//!
//! - [PyTorch Implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py)

use candle::{Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d_no_bias, linear, ops::softmax, Activation, Conv2dConfig, Func, VarBuilder,
};

#[derive(Clone, Debug)]
enum BlockType {
    Convolutional {
        out_channels: usize,
        kernel: usize,
        stride: usize,
    },
    UniversalBottleneck {
        out_channels: usize,
        start_kernel: usize,
        mid_kernel: usize,
        stride: usize,
        expand: usize,
    },
    EdgeResidual {
        out_channels: usize,
        kernel: usize,
        stride: usize,
        expand: usize,
    },
    Attention {
        out_channels: usize,
        heads: usize,
        kernel: usize,
        stride: usize,
        kv_dim: usize,
        kv_stride: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Config {
    stem_dim: usize,
    activation: Activation,
    stages: [Vec<BlockType>; 5],
}

#[rustfmt::skip]
impl Config {
    pub fn small() -> Self {
        Self {
            stem_dim: 32,
            activation: Activation::Relu,
            stages: [
                vec![
                    BlockType::Convolutional { out_channels: 32, kernel: 3, stride: 2},
                    BlockType::Convolutional { out_channels: 32, kernel: 1, stride: 1},
                ],
                vec![
                    BlockType::Convolutional { out_channels: 96, kernel: 3, stride: 2},
                    BlockType::Convolutional { out_channels: 64, kernel: 1, stride: 1},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 5, mid_kernel: 5, stride: 2, expand: 3},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 3, mid_kernel: 3, stride: 2, expand: 6},
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 5, stride: 1, expand: 3},
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 128, start_kernel: 0, mid_kernel: 3, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::Convolutional { out_channels: 960, kernel: 1, stride: 1},
                ],
            ],
        }
    }

    pub fn medium() -> Self {
        Self {
            stem_dim: 32,
            activation: Activation::Relu,
            stages: [
                 vec![
                    BlockType::EdgeResidual { out_channels: 48, kernel: 3, stride: 2, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 80, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 80, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 2},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 6},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 2, expand: 6},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 2},

               ],
                vec![
                    BlockType::Convolutional { out_channels: 960, kernel: 1, stride: 1},
                ],
            ],
        }
    }

    pub fn hybrid_medium() -> Self {
        Self {
            stem_dim: 32,
            activation: Activation::Relu,
            stages: [
                 vec![
                    BlockType::EdgeResidual { out_channels: 48, kernel: 3, stride: 2, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 80, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 80, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 2},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 6},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 160, heads: 4, kernel: 3, stride: 1, kv_stride:2, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 160, heads: 4, kernel: 3, stride: 1, kv_stride:2, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 160, heads: 4, kernel: 3, stride: 1, kv_stride:2, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 160, heads: 4, kernel: 3, stride: 1, kv_stride:2, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 160, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                ],

               vec![
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 2, expand: 6},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 2},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 0, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 256, heads: 4, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 256, heads: 4, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 256, heads: 4, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 256, heads: 4, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 256, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
               ],
                vec![
                    BlockType::Convolutional { out_channels: 960, kernel: 1, stride: 1},
                ],
            ],
        }
    }

    pub fn large() -> Self {
        Self {
            stem_dim: 24,
            activation: Activation::Relu,
            stages: [
                vec![
                    BlockType::EdgeResidual { out_channels: 48, kernel: 3, stride: 2, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::Convolutional { out_channels: 960, kernel: 1, stride: 1},
                ],
            ],
        }
    }

    pub fn hybrid_large() -> Self {
        Self {
            stem_dim: 24,
            activation: Activation::Gelu,
            stages: [
                vec![
                    BlockType::EdgeResidual { out_channels: 48, kernel: 3, stride: 2, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 96, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 192, heads: 8, kernel: 3, stride: 1, kv_stride:2, kv_dim: 48},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 192, heads: 8, kernel: 3, stride: 1, kv_stride:2, kv_dim: 48},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 192, heads: 8, kernel: 3, stride: 1, kv_stride:2, kv_dim: 48},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 192, heads: 8, kernel: 3, stride: 1, kv_stride:2, kv_dim: 48},
                    BlockType::UniversalBottleneck { out_channels: 192, start_kernel: 3, mid_kernel: 0, stride: 1, expand: 4},
                ],

                vec![
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 2, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 3, stride: 1, expand: 4},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 5, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 512, heads: 8, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 512, heads: 8, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 512, heads: 8, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                    BlockType::Attention { out_channels: 512, heads: 8, kernel: 3, stride: 1, kv_stride:1, kv_dim: 64},
                    BlockType::UniversalBottleneck { out_channels: 512, start_kernel: 5, mid_kernel: 0, stride: 1, expand: 4},
                ],
                vec![
                    BlockType::Convolutional { out_channels: 960, kernel: 1, stride: 1},
                ],
            ],
          }
    }
}

fn depthwise_conv(
    channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride,
        padding,
        groups: channels,
        ..Default::default()
    };

    let bn = batch_norm(channels, 1e-5, vb.pp("bn"))?;
    let conv = conv2d_no_bias(channels, channels, kernel, conv2d_cfg, vb.pp("conv"))?;

    Ok(Func::new(move |xs| xs.apply(&conv)?.apply_t(&bn, false)))
}

fn pointwise_conv(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        ..Default::default()
    };

    let bn = batch_norm(out_channels, 1e-5, vb.pp("bn"))?;
    let conv = conv2d_no_bias(in_channels, out_channels, 1, conv2d_cfg, vb.pp("conv"))?;

    Ok(Func::new(move |xs| xs.apply(&conv)?.apply_t(&bn, false)))
}

//Universal block that uses two pointwise convolutions and all combinations of two depthwise convolutions.
#[allow(clippy::too_many_arguments)]
fn universal_inverted_bottleneck_block(
    cfg: &Config,
    in_channels: usize,
    out_channels: usize,
    expand: usize,
    start_kernel: usize,
    mid_kernel: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let act = cfg.activation;
    let skip_connection = (in_channels == out_channels) && (stride == 1);

    let dw_start_stride = if mid_kernel > 0 { 1 } else { stride };
    let dw_start = depthwise_conv(
        in_channels,
        start_kernel,
        dw_start_stride,
        start_kernel / 2,
        vb.pp("dw_start"),
    );
    let pw_exp = pointwise_conv(in_channels, in_channels * expand, vb.pp("pw_exp"))?;
    let dw_mid = depthwise_conv(
        in_channels * expand,
        mid_kernel,
        stride,
        mid_kernel / 2,
        vb.pp("dw_mid"),
    );
    let pw_proj = pointwise_conv(in_channels * expand, out_channels, vb.pp("pw_proj"))?;

    let gamma = vb.get(out_channels, "layer_scale.gamma");

    Ok(Func::new(move |xs| {
        let residual = xs.clone();

        let mut xs = xs.clone();

        if let Ok(f) = &dw_start {
            xs = xs.apply(f)?;
        }

        xs = xs.apply(&pw_exp)?.apply(&act)?;

        if let Ok(f) = &dw_mid {
            xs = xs.apply(f)?.apply(&act)?;
        }

        xs = xs.apply(&pw_proj)?;

        if let Ok(g) = &gamma {
            xs = xs.broadcast_mul(&g.reshape((1, (), 1, 1))?)?;
        };

        if skip_connection {
            xs = (xs + residual)?;
        }

        Ok(xs)
    }))
}

// Convolutional block including norm and activation.
fn conv_block(
    cfg: &Config,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride,
        padding: kernel / 2,
        ..Default::default()
    };

    let act = cfg.activation;
    let bn = batch_norm(out_channels, 1e-5, vb.pp("bn1"))?;
    let conv = conv2d_no_bias(in_channels, out_channels, kernel, conv2d_cfg, vb.pp("conv"))?;

    Ok(Func::new(move |xs| {
        xs.apply(&conv)?.apply_t(&bn, false)?.apply(&act)
    }))
}

fn edge_residual_block(
    cfg: &Config,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    expand: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv_exp_cfg = Conv2dConfig {
        stride,
        padding: kernel / 2,
        ..Default::default()
    };

    let conv_pwl_cfg = Conv2dConfig {
        ..Default::default()
    };

    let act = cfg.activation;
    let mid_channels = in_channels * expand;
    let conv_exp = conv2d_no_bias(
        in_channels,
        mid_channels,
        kernel,
        conv_exp_cfg,
        vb.pp("conv_exp"),
    )?;
    let bn1 = batch_norm(mid_channels, 1e-5, vb.pp("bn1"))?;

    let conv_pwl = conv2d_no_bias(
        mid_channels,
        out_channels,
        1,
        conv_pwl_cfg,
        vb.pp("conv_pwl"),
    )?;
    let bn2 = batch_norm(out_channels, 1e-5, vb.pp("bn2"))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv_exp)?
            .apply_t(&bn1, false)?
            .apply(&act)?
            .apply(&conv_pwl)?
            .apply_t(&bn2, false)?;

        Ok(xs)
    }))
}

fn reshape_kv(t: &Tensor) -> Result<Tensor> {
    let d = t.dims4()?;
    let t = t
        .reshape((d.0, d.1, ()))?
        .transpose(1, 2)?
        .unsqueeze(1)?
        .contiguous()?;
    Ok(t)
}

fn reshape_query(t: &Tensor, heads: usize, kv_dim: usize) -> Result<Tensor> {
    let d = t.dims4()?;

    let t = t
        .reshape((d.0, heads, kv_dim, ()))?
        .transpose(D::Minus1, D::Minus2)?
        .contiguous()?;
    Ok(t)
}

fn reshape_output(t: &Tensor, heads: usize, h: usize, w: usize) -> Result<Tensor> {
    let d = t.dims4()?;
    let t = t.transpose(1, 2)?;
    let t = t
        .reshape((d.0, h, w, d.3 * heads))?
        .permute((0, 3, 1, 2))?
        .contiguous()?;
    Ok(t)
}

// Mobile multi-query attention
#[allow(clippy::too_many_arguments)]
fn mqa_block(
    in_channels: usize,
    out_channels: usize,
    heads: usize,
    kernel: usize,
    stride: usize,
    kv_dim: usize,
    kv_stride: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let down_conv2d_cfg = Conv2dConfig {
        stride: kv_stride,
        padding: kernel / 2,
        groups: in_channels,
        ..Default::default()
    };

    let proj_conv2d_cfg = Conv2dConfig {
        stride,
        ..Default::default()
    };

    let skip_connection = (in_channels == out_channels) && (stride == 1);
    let gamma = vb.get(out_channels, "layer_scale.gamma");
    let norm = batch_norm(out_channels, 1e-5, vb.pp("norm"))?;
    let scale = (kv_dim as f64).powf(-0.5);

    let vb = vb.pp("attn");

    let query_proj = conv2d_no_bias(
        out_channels,
        kv_dim * heads,
        1,
        proj_conv2d_cfg,
        vb.pp("query.proj"),
    )?;

    let key_down_conv = conv2d_no_bias(
        in_channels,
        out_channels,
        kernel,
        down_conv2d_cfg,
        vb.pp("key.down_conv"),
    );
    let key_norm = batch_norm(out_channels, 1e-5, vb.pp("key.norm"));

    let key_proj = conv2d_no_bias(out_channels, kv_dim, 1, proj_conv2d_cfg, vb.pp("key.proj"))?;

    let value_down_conv = conv2d_no_bias(
        in_channels,
        out_channels,
        kernel,
        down_conv2d_cfg,
        vb.pp("value.down_conv"),
    );

    let value_norm = batch_norm(out_channels, 1e-5, vb.pp("value.norm"));
    let value_proj = conv2d_no_bias(
        out_channels,
        kv_dim,
        1,
        proj_conv2d_cfg,
        vb.pp("value.proj"),
    )?;

    let output_proj = conv2d_no_bias(
        kv_dim * heads,
        out_channels,
        1,
        proj_conv2d_cfg,
        vb.pp("output.proj"),
    )?;

    Ok(Func::new(move |xs| {
        let (_, _, h, w) = xs.dims4()?;

        let residual = xs.clone();

        let xs = xs.apply_t(&norm, false)?;

        // Query
        let q = xs.apply(&query_proj)?;

        let q = reshape_query(&q, heads, kv_dim)?;
        let q = (q * scale)?;

        // Keys
        let mut k = xs.clone();

        if let (Ok(kd), Ok(n)) = (&key_down_conv, &key_norm) {
            k = k.apply(kd)?.apply_t(n, false)?;
        }

        let k = k.apply(&key_proj)?;

        let k = reshape_kv(&k)?;

        // Value
        let mut v = xs.clone();

        if let (Ok(vd), Ok(n)) = (&value_down_conv, &value_norm) {
            v = v.apply(vd)?;
            v = v.apply_t(n, false)?;
        }

        let v = v.apply(&value_proj)?;
        let v = reshape_kv(&v)?;

        let attn = q.broadcast_matmul(&(k.transpose(D::Minus2, D::Minus1)?))?;
        let attn = softmax(&attn, D::Minus1)?;
        let o = attn.broadcast_matmul(&v)?;

        let o = reshape_output(&o, heads, h, w)?;

        let mut xs = o.apply(&output_proj)?;

        // Layer scale

        if let Ok(g) = &gamma {
            xs = xs.broadcast_mul(&g.reshape((1, (), 1, 1))?)?;
        };

        if skip_connection {
            xs = (xs + residual)?;
        }
        Ok(xs)
    }))
}

// Stem.
fn mobilenetv4_stem(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 2,
        padding: 1,
        ..Default::default()
    };

    let act = cfg.activation;
    let out_channels = cfg.stem_dim;
    let bn = batch_norm(out_channels, 1e-5, vb.pp("bn1"))?;
    let conv = conv2d_no_bias(3, out_channels, 3, conv2d_cfg, vb.pp("conv_stem"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&conv)?.apply_t(&bn, false)?.apply(&act)?;
        Ok(xs)
    }))
}

// The blocks in all the 5 stages of the model.
fn mobilenetv4_blocks(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    let mut in_channels = cfg.stem_dim;
    let mut blocks = Vec::new();

    for stage in 0..5 {
        let nblocks = cfg.stages[stage].len();

        for block in 0..nblocks {
            match cfg.stages[stage][block] {
                BlockType::Convolutional {
                    out_channels,
                    kernel,
                    stride,
                } => {
                    blocks.push(conv_block(
                        cfg,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        vb.pp(format!("{stage}.{block}")),
                    )?);
                    in_channels = out_channels;
                }

                BlockType::EdgeResidual {
                    out_channels,
                    kernel,
                    stride,
                    expand,
                } => {
                    blocks.push(edge_residual_block(
                        cfg,
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        expand,
                        vb.pp(format!("{stage}.{block}")),
                    )?);
                    in_channels = out_channels;
                }

                BlockType::UniversalBottleneck {
                    out_channels,
                    start_kernel,
                    mid_kernel,
                    stride,
                    expand,
                } => {
                    blocks.push(universal_inverted_bottleneck_block(
                        cfg,
                        in_channels,
                        out_channels,
                        expand,
                        start_kernel,
                        mid_kernel,
                        stride,
                        vb.pp(format!("{stage}.{block}")),
                    )?);
                    in_channels = out_channels;
                }

                BlockType::Attention {
                    out_channels,
                    heads,
                    kernel,
                    stride,
                    kv_dim,
                    kv_stride,
                } => {
                    blocks.push(mqa_block(
                        in_channels,
                        out_channels,
                        heads,
                        kernel,
                        stride,
                        kv_dim,
                        kv_stride,
                        vb.pp(format!("{stage}.{block}")),
                    )?);
                    in_channels = out_channels;
                }
            }
        }
    }

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for block in blocks.iter() {
            xs = xs.apply(block)?
        }
        Ok(xs)
    }))
}

// Classification head.
fn mobilenetv4_head(
    cfg: &Config,
    outputs: usize,
    nclasses: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        ..Default::default()
    };

    let act = cfg.activation;
    let conv = conv2d_no_bias(960, outputs, 1, conv2d_cfg, vb.pp("conv_head"))?;
    let norm = batch_norm(outputs, 1e-5, vb.pp("norm_head"))?;
    let cls = linear(outputs, nclasses, vb.pp("classifier"))?;

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        xs = xs.apply(&conv)?;
        xs = xs.apply_t(&norm, false)?.apply(&act)?;
        xs = xs.flatten_from(1)?;
        xs = xs.apply(&cls)?;
        Ok(xs)
    }))
}

// Build a mobilenetv4 model for a given configuration.
fn mobilenetv4_model(
    cfg: &Config,
    nclasses: Option<usize>,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let outputs = 1280;
            let head = mobilenetv4_head(cfg, outputs, nclasses, vb.clone())?;
            Some(head)
        }
    };

    let stem = mobilenetv4_stem(cfg, vb.clone())?;

    let blocks = mobilenetv4_blocks(cfg, vb.pp("blocks"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&stem)?.apply(&blocks)?;
        let xs = xs.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.apply(cls),
        }
    }))
}

pub fn mobilenetv4(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    mobilenetv4_model(cfg, Some(nclasses), vb)
}

pub fn mobilenetv4_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    mobilenetv4_model(cfg, None, vb)
}
