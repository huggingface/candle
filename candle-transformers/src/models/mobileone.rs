//! MobileOne inference implementation based on timm and candle-repvgg
//!
//! See "MobileOne: An Improved One millisecond Mobile Backbone"
//! https://arxiv.org/abs/2206.04040

use candle::{DType, Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, ops::sigmoid, BatchNorm, Conv2d, Conv2dConfig,
    Func, VarBuilder,
};

struct StageConfig {
    blocks: usize,
    channels: usize,
}

// The architecture in the paper has 6 stages. The timm implementation uses an equivalent form
// by concatenating the 5th stage (starts with stride 1) to the previous one.
const STAGES: [StageConfig; 5] = [
    StageConfig {
        blocks: 1,
        channels: 64,
    },
    StageConfig {
        blocks: 2,
        channels: 64,
    },
    StageConfig {
        blocks: 8,
        channels: 128,
    },
    StageConfig {
        blocks: 10,
        channels: 256,
    },
    StageConfig {
        blocks: 1,
        channels: 512,
    },
];

#[derive(Clone)]
pub struct Config {
    /// overparameterization factor
    k: usize,
    /// per-stage channel number multipliers
    alphas: [f32; 5],
}

impl Config {
    pub fn s0() -> Self {
        Self {
            k: 4,
            alphas: [0.75, 0.75, 1.0, 1.0, 2.0],
        }
    }
    pub fn s1() -> Self {
        Self {
            k: 1,
            alphas: [1.5, 1.5, 1.5, 2.0, 2.5],
        }
    }
    pub fn s2() -> Self {
        Self {
            k: 1,
            alphas: [1.5, 1.5, 2.0, 2.5, 4.0],
        }
    }
    pub fn s3() -> Self {
        Self {
            k: 1,
            alphas: [2.0, 2.0, 2.5, 3.0, 4.0],
        }
    }
    pub fn s4() -> Self {
        Self {
            k: 1,
            alphas: [3.0, 3.0, 3.5, 3.5, 4.0],
        }
    }
}

// SE blocks are used in the last stages of the s4 variant.
fn squeeze_and_excitation(
    in_channels: usize,
    squeeze_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        ..Default::default()
    };
    let fc1 = conv2d(in_channels, squeeze_channels, 1, conv2d_cfg, vb.pp("fc1"))?;
    let fc2 = conv2d(squeeze_channels, in_channels, 1, conv2d_cfg, vb.pp("fc2"))?;

    Ok(Func::new(move |xs| {
        let residual = xs;
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = sigmoid(&xs.apply(&fc1)?.relu()?.apply(&fc2)?)?;

        residual.broadcast_mul(&xs)
    }))
}

// fuses a convolutional kernel and a batchnorm layer into a convolutional layer
// based on the _fuse_bn_tensor method in timm
// see https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/byobnet.py#L602
fn fuse_conv_bn(weights: &Tensor, bn: BatchNorm) -> Result<(Tensor, Tensor)> {
    let (gamma, beta) = bn.weight_and_bias().unwrap();
    let mu = bn.running_mean();
    let sigma = (bn.running_var() + bn.eps())?.sqrt();
    let gps = (gamma / sigma)?;
    let bias = (beta - mu * &gps)?;
    let weights = weights.broadcast_mul(&gps.reshape(((), 1, 1, 1))?)?;

    Ok((weights, bias))
}

// A mobileone block has a different training time and inference time architecture.
// The latter is a simple and efficient equivalent transformation of the former
// realized by a structural reparameterization technique, where convolutions
// along with identity branches and batchnorm layers are fused into a single convolution.
#[allow(clippy::too_many_arguments)]
fn mobileone_block(
    has_identity: bool,
    k: usize,
    dim: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride,
        padding,
        groups,
        ..Default::default()
    };

    let mut w = Tensor::zeros(
        (out_channels, in_channels / groups, kernel, kernel),
        DType::F32,
        vb.device(),
    )?;
    let mut b = Tensor::zeros(dim, DType::F32, vb.device())?;

    // k is the training-time overparameterization factor, larger than 1 only in the s0 variant
    for i in 0..k {
        let conv_kxk_bn = batch_norm(dim, 1e-5, vb.pp(format!("conv_kxk.{i}.bn")))?;
        let conv_kxk = conv2d_no_bias(
            in_channels,
            out_channels,
            kernel,
            conv2d_cfg,
            vb.pp(format!("conv_kxk.{i}.conv")),
        )?;
        let (wk, bk) = fuse_conv_bn(conv_kxk.weight(), conv_kxk_bn)?;
        w = (w + wk)?;
        b = (b + bk)?;
    }

    if kernel > 1 {
        let conv_scale_bn = batch_norm(dim, 1e-5, vb.pp("conv_scale.bn"))?;
        let conv_scale = conv2d_no_bias(
            in_channels,
            out_channels,
            1,
            conv2d_cfg,
            vb.pp("conv_scale.conv"),
        )?;

        let (mut ws, bs) = fuse_conv_bn(conv_scale.weight(), conv_scale_bn)?;
        // resize to 3x3
        ws = ws.pad_with_zeros(D::Minus1, 1, 1)?;
        ws = ws.pad_with_zeros(D::Minus2, 1, 1)?;

        w = (w + ws)?;
        b = (b + bs)?;
    }

    // Use SE blocks if present (last layers of the s4 variant)
    let se = squeeze_and_excitation(out_channels, out_channels / 16, vb.pp("attn"));

    // read and reparameterize the identity bn into wi and bi
    if has_identity {
        let identity_bn = batch_norm(dim, 1e-5, vb.pp("identity"))?;

        let mut weights: Vec<f32> = vec![0.0; w.elem_count()];

        let id = in_channels / groups;
        // See https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/byobnet.py#L809
        for i in 0..in_channels {
            if kernel > 1 {
                weights[i * kernel * kernel + 4] = 1.0;
            } else {
                weights[i * (id + 1)] = 1.0;
            }
        }

        let weights = &Tensor::from_vec(weights, w.shape(), w.device())?;
        let (wi, bi) = fuse_conv_bn(weights, identity_bn)?;

        w = (w + wi)?;
        b = (b + bi)?;
    }

    let reparam_conv = Conv2d::new(w, Some(b), conv2d_cfg);

    Ok(Func::new(move |xs| {
        let mut xs = xs.apply(&reparam_conv)?;
        if let Ok(f) = &se {
            xs = xs.apply(f)?;
        }
        xs = xs.relu()?;
        Ok(xs)
    }))
}

// Get the number of output channels per stage taking into account the multipliers
fn output_channels_per_stage(cfg: &Config, stage: usize) -> usize {
    let channels = STAGES[stage].channels as f32;
    let alpha = cfg.alphas[stage];

    match stage {
        0 => std::cmp::min(64, (channels * alpha) as usize),
        _ => (channels * alpha) as usize,
    }
}

// Each stage is made of blocks. The first layer always downsamples with stride 2.
// All but the first block have a residual connection.
fn mobileone_stage(cfg: &Config, idx: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let nblocks = STAGES[idx].blocks;
    let mut blocks = Vec::with_capacity(nblocks);

    let mut in_channels = output_channels_per_stage(cfg, idx - 1);

    for block_idx in 0..nblocks {
        let out_channels = output_channels_per_stage(cfg, idx);
        let (has_identity, stride) = if block_idx == 0 {
            (false, 2)
        } else {
            (true, 1)
        };

        // depthwise convolution layer
        blocks.push(mobileone_block(
            has_identity,
            cfg.k,
            in_channels,
            stride,
            1,
            in_channels,
            3,
            in_channels,
            in_channels,
            vb.pp(block_idx * 2),
        )?);

        // pointwise convolution layer
        blocks.push(mobileone_block(
            has_identity,
            cfg.k,
            out_channels,
            1, // stride
            0, // padding
            1, // groups
            1, // kernel
            in_channels,
            out_channels,
            vb.pp(block_idx * 2 + 1),
        )?);

        in_channels = out_channels;
    }

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for block in blocks.iter() {
            xs = xs.apply(block)?
        }
        Ok(xs)
    }))
}

// Build a mobileone model for a given configuration.
fn mobileone_model(
    config: &Config,
    nclasses: Option<usize>,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let outputs = output_channels_per_stage(config, 4);
            let linear = linear(outputs, nclasses, vb.pp("head.fc"))?;
            Some(linear)
        }
    };

    let stem_dim = output_channels_per_stage(config, 0);
    let stem = mobileone_block(false, 1, stem_dim, 2, 1, 1, 3, 3, stem_dim, vb.pp("stem"))?;
    let vb = vb.pp("stages");
    let stage1 = mobileone_stage(config, 1, vb.pp(0))?;
    let stage2 = mobileone_stage(config, 2, vb.pp(1))?;
    let stage3 = mobileone_stage(config, 3, vb.pp(2))?;
    let stage4 = mobileone_stage(config, 4, vb.pp(3))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&stem)?
            .apply(&stage1)?
            .apply(&stage2)?
            .apply(&stage3)?
            .apply(&stage4)?
            .mean(D::Minus2)?
            .mean(D::Minus1)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.apply(cls),
        }
    }))
}

pub fn mobileone(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    mobileone_model(cfg, Some(nclasses), vb)
}

pub fn mobileone_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    mobileone_model(cfg, None, vb)
}
