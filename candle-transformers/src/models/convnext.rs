//! ConvNeXt implementation.
//!
//! See "A ConvNet for the 2020s" Liu et al. 2022
//! <https://arxiv.org/abs/2201.03545>

//! Original code: https://github.com/facebookresearch/ConvNeXt/
//! timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

use candle::{Result, D};
use candle_nn::{conv2d, layer_norm, linear, Conv2dConfig, Func, VarBuilder};

#[derive(Clone)]
pub struct Config {
    blocks: [usize; 4],
    channels: [usize; 4],
}

impl Config {
    pub fn tiny() -> Self {
        Self {
            blocks: [3, 3, 9, 3],
            channels: [96, 192, 384, 768],
        }
    }
    pub fn small() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [96, 192, 384, 768],
        }
    }
    pub fn base() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [128, 256, 512, 1024],
        }
    }
    pub fn large() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [192, 384, 768, 1536],
        }
    }

    pub fn xlarge() -> Self {
        Self {
            blocks: [3, 3, 27, 3],
            channels: [256, 512, 1024, 2048],
        }
    }
}

// Initial downsampling via a patchify layer.
fn convnext_stem(out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 4,
        ..Default::default()
    };
    let patchify = conv2d(3, out_channels, 4, conv2d_cfg, vb.pp(0))?;
    let norm = layer_norm(out_channels, 1e-6, vb.pp(1))?;
    Ok(Func::new(move |xs| {
        // The layer norm works with channels-last format.
        let xs = xs
            .apply(&patchify)?
            .permute((0, 2, 3, 1))?
            .apply(&norm)?
            .permute((0, 3, 1, 2))?;
        Ok(xs)
    }))
}

// Downsampling applied after the stages.
fn convnext_downsample(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 2,
        ..Default::default()
    };
    let norm = layer_norm(dim / 2, 1e-5, vb.pp(0))?;
    let conv = conv2d(dim / 2, dim, 2, conv2d_cfg, vb.pp(1))?;
    Ok(Func::new(move |xs| {
        let xs = xs
            .permute((0, 2, 3, 1))?
            .apply(&norm)?
            .permute((0, 3, 1, 2))?
            .apply(&conv)?;
        Ok(xs)
    }))
}

// MLP equivalent of pointwise convolutions.
fn convnext_mlp(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let fc1 = linear(dim, 4 * dim, vb.pp("fc1"))?;
    let fc2 = linear(4 * dim, dim, vb.pp("fc2"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&fc1)?.gelu_erf()?.apply(&fc2)?;
        Ok(xs)
    }))
}

// A block consisting of a depthwise convolution, a MLP and layer scaling.
fn convnext_block(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        groups: dim,
        padding: 3,
        ..Default::default()
    };

    let conv_dw = conv2d(dim, dim, 7, conv2d_cfg, vb.pp("conv_dw"))?;

    let gamma = vb.get(dim, "gamma")?;
    let mlp = convnext_mlp(dim, vb.pp("mlp"))?;
    let norm = layer_norm(dim, 1e-6, vb.pp("norm"))?;

    Ok(Func::new(move |xs| {
        let residual = xs;
        let xs = xs
            .apply(&conv_dw)?
            .permute((0, 2, 3, 1))?
            .apply(&norm)?
            .apply(&mlp)?
            .broadcast_mul(&gamma)?
            .permute((0, 3, 1, 2))?;

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
        blocks.push(convnext_block(dim, vb.pp(format!("blocks.{block_idx}")))?);
    }

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for block in blocks.iter() {
            xs = xs.apply(block)?
        }
        Ok(xs)
    }))
}

fn convnext_head(outputs: usize, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = layer_norm(outputs, 1e-6, vb.pp("norm"))?;
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
