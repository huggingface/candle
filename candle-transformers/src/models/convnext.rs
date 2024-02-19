//! ConvNeXt implementation.
//!
//! See "A ConvNet for the 2020s" Liu et al. 2022
//! <https://arxiv.org/abs/2201.03545>
//! and
//! "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" Woo et al. 2023
//! <https://arxiv.org/abs/2301.00808>

//! Original code:
//! https://github.com/facebookresearch/ConvNeXt/
//! https://github.com/facebookresearch/ConvNeXt-V2/
//! timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

use candle::shape::ShapeWithOneHole;
use candle::{Result, D};
use candle_nn::{conv2d, layer_norm, linear, Conv2dConfig, Func, VarBuilder};

#[derive(Clone)]
pub struct Config {
    blocks: [usize; 4],
    channels: [usize; 4],
    use_conv_mlp: bool,
}

impl Config {
    pub fn atto() -> Self {
        Self {
            blocks: [2, 2, 6, 2],
            channels: [40, 80, 160, 320],
            use_conv_mlp: true,
        }
    }

    pub fn femto() -> Self {
        Self {
            blocks: [2, 2, 6, 2],
            channels: [48, 96, 192, 384],
            use_conv_mlp: true,
        }
    }

    pub fn pico() -> Self {
        Self {
            blocks: [2, 2, 6, 2],
            channels: [64, 128, 256, 512],
            use_conv_mlp: true,
        }
    }

    pub fn nano() -> Self {
        Self {
            blocks: [2, 2, 8, 2],
            channels: [80, 160, 320, 640],
            use_conv_mlp: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            blocks: [3, 3, 9, 3],
            channels: [96, 192, 384, 768],
            use_conv_mlp: false,
        }
    }

    pub fn small() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [96, 192, 384, 768],
            use_conv_mlp: false,
        }
    }

    pub fn base() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [128, 256, 512, 1024],
            use_conv_mlp: false,
        }
    }

    pub fn large() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [192, 384, 768, 1536],
            use_conv_mlp: false,
        }
    }

    pub fn xlarge() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [256, 512, 1024, 2048],
            use_conv_mlp: false,
        }
    }

    pub fn huge() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [352, 704, 1408, 2816],
            use_conv_mlp: false,
        }
    }
}

// Layer norm for data in channels-last format.
fn layer_norm_cl(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = layer_norm(dim, 1e-6, vb)?;

    Ok(Func::new(move |xs| xs.apply(&norm)))
}

// Layer norm for data in channels-first format.
fn layer_norm_cf(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = layer_norm(dim, 1e-6, vb)?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .permute((0, 2, 3, 1))?
            .apply(&norm)?
            .permute((0, 3, 1, 2))?;
        Ok(xs)
    }))
}

// Global response normalization layer
// Based on https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/grn.py
fn convnext2_grn(dim: usize, channels_last: bool, vb: VarBuilder) -> Result<Func<'static>> {
    let (shape, spatial_dim, channel_dim) = if channels_last {
        ((1, 1, 1, ()).into_shape(dim)?, [1, 2], 3)
    } else {
        ((1, (), 1, 1).into_shape(dim)?, [2, 3], 1)
    };

    let gamma = vb.get(dim, "weight")?.reshape(&shape)?;
    let beta = vb.get(dim, "bias")?.reshape(&shape)?;

    Ok(Func::new(move |xs| {
        let residual = xs;
        let gx = xs
            .sqr()?
            .sum_keepdim(spatial_dim)?
            .mean_keepdim(spatial_dim)?
            .sqrt()?;

        let gxmean = gx.mean_keepdim(channel_dim)?;
        let nx = gx.broadcast_div(&(gxmean + 1e-6)?)?;
        let xs = xs
            .broadcast_mul(&nx)?
            .broadcast_mul(&gamma)?
            .broadcast_add(&beta)?;

        xs + residual
    }))
}

// Initial downsampling via a patchify layer.
fn convnext_stem(out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 4,
        ..Default::default()
    };
    let patchify = conv2d(3, out_channels, 4, conv2d_cfg, vb.pp(0))?;
    let norm = layer_norm_cf(out_channels, vb.pp(1))?;

    Ok(Func::new(move |xs| xs.apply(&patchify)?.apply(&norm)))
}

// Downsampling applied after the stages.
fn convnext_downsample(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 2,
        ..Default::default()
    };
    let norm = layer_norm_cf(dim / 2, vb.pp(0))?;
    let conv = conv2d(dim / 2, dim, 2, conv2d_cfg, vb.pp(1))?;

    Ok(Func::new(move |xs| xs.apply(&norm)?.apply(&conv)))
}

// MLP block from the original paper with optional GRN layer (v2 models).
fn convnext_mlp(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let fc1 = linear(dim, 4 * dim, vb.pp("fc1"))?;
    let fc2 = linear(4 * dim, dim, vb.pp("fc2"))?;
    let grn = convnext2_grn(4 * dim, true, vb.pp("grn"));

    Ok(Func::new(move |xs| {
        let mut xs = xs.apply(&fc1)?.gelu_erf()?;
        if let Ok(g) = &grn {
            xs = xs.apply(g)?;
        }
        xs = xs.apply(&fc2)?;
        Ok(xs)
    }))
}

// MLP block using pointwise convolutions, with optional GRN layer (v2 models).
fn convnext_conv_mlp(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        ..Default::default()
    };
    let fc1 = conv2d(dim, 4 * dim, 1, conv2d_cfg, vb.pp("fc1"))?;
    let fc2 = conv2d(4 * dim, dim, 1, conv2d_cfg, vb.pp("fc2"))?;

    let grn = convnext2_grn(4 * dim, false, vb.pp("grn"));
    Ok(Func::new(move |xs| {
        let mut xs = xs.apply(&fc1)?.gelu_erf()?;
        if let Ok(g) = &grn {
            xs = xs.apply(g)?;
        }
        xs = xs.apply(&fc2)?;
        Ok(xs)
    }))
}

// A block consisting of a depthwise convolution, a MLP and layer scaling (v1 models only).
fn convnext_block(dim: usize, use_conv_mlp: bool, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        groups: dim,
        padding: 3,
        ..Default::default()
    };

    let conv_dw = conv2d(dim, dim, 7, conv2d_cfg, vb.pp("conv_dw"))?;
    let gamma = vb.get(dim, "gamma");

    let (mlp, norm) = if use_conv_mlp {
        (
            convnext_conv_mlp(dim, vb.pp("mlp"))?,
            layer_norm_cf(dim, vb.pp("norm"))?,
        )
    } else {
        (
            convnext_mlp(dim, vb.pp("mlp"))?,
            layer_norm_cl(dim, vb.pp("norm"))?,
        )
    };

    Ok(Func::new(move |xs| {
        let residual = xs;
        let mut xs = xs.apply(&conv_dw)?;

        xs = if use_conv_mlp {
            xs.apply(&norm)?.apply(&mlp)?
        } else {
            xs.permute((0, 2, 3, 1))?
                .apply(&norm)?
                .apply(&mlp)?
                .permute((0, 3, 1, 2))?
        };

        if let Ok(g) = &gamma {
            xs = xs.broadcast_mul(&g.reshape((1, (), 1, 1))?)?;
        };

        xs + residual
    }))
}

// Each stage contains blocks and a downsampling layer for the previous stage.
fn convnext_stage(cfg: &Config, stage_idx: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let nblocks = cfg.blocks[stage_idx];
    let mut blocks = Vec::with_capacity(nblocks);

    let dim = cfg.channels[stage_idx];

    if stage_idx > 0 {
        blocks.push(convnext_downsample(dim, vb.pp("downsample"))?);
    }

    for block_idx in 0..nblocks {
        blocks.push(convnext_block(
            dim,
            cfg.use_conv_mlp,
            vb.pp(format!("blocks.{block_idx}")),
        )?);
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
fn convnext_head(outputs: usize, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = layer_norm_cl(outputs, vb.pp("norm"))?;
    let linear = linear(outputs, nclasses, vb.pp("fc"))?;
    Ok(Func::new(move |xs| xs.apply(&norm)?.apply(&linear)))
}

// Build a convnext model for a given configuration.
fn convnext_model(
    config: &Config,
    nclasses: Option<usize>,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let head = match nclasses {
        None => None,
        Some(nclasses) => {
            let head = convnext_head(config.channels[3], nclasses, vb.pp("head"))?;
            Some(head)
        }
    };

    let stem = convnext_stem(config.channels[0], vb.pp("stem"))?;
    let vb = vb.pp("stages");
    let stage1 = convnext_stage(config, 0, vb.pp(0))?;
    let stage2 = convnext_stage(config, 1, vb.pp(1))?;
    let stage3 = convnext_stage(config, 2, vb.pp(2))?;
    let stage4 = convnext_stage(config, 3, vb.pp(3))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&stem)?
            .apply(&stage1)?
            .apply(&stage2)?
            .apply(&stage3)?
            .apply(&stage4)?
            .mean(D::Minus2)?
            .mean(D::Minus1)?;
        match &head {
            None => Ok(xs),
            Some(head) => xs.apply(head),
        }
    }))
}

pub fn convnext(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    convnext_model(cfg, Some(nclasses), vb)
}

pub fn convnext_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    convnext_model(cfg, None, vb)
}
