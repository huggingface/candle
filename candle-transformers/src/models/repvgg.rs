//! RepVGG inference implementation
//!
//! See "RepVGG: Making VGG-style ConvNets Great Again" Ding et al. 2021
//! https://arxiv.org/abs/2101.03697

use candle::{Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d_no_bias, linear, BatchNorm, Conv2d, Conv2dConfig, Func, VarBuilder,
};

const CHANNELS_PER_STAGE: [usize; 5] = [64, 64, 128, 256, 512];

#[derive(Clone)]
pub struct Config {
    a: f32,
    b: f32,
    groups: usize,
    stages: [usize; 4],
}

impl Config {
    pub fn a0() -> Self {
        Self {
            a: 0.75,
            b: 2.5,
            groups: 1,
            stages: [2, 4, 14, 1],
        }
    }

    pub fn a1() -> Self {
        Self {
            a: 1.0,
            b: 2.5,
            groups: 1,
            stages: [2, 4, 14, 1],
        }
    }

    pub fn a2() -> Self {
        Self {
            a: 1.5,
            b: 2.75,
            groups: 1,
            stages: [2, 4, 14, 1],
        }
    }

    pub fn b0() -> Self {
        Self {
            a: 1.0,
            b: 2.5,
            groups: 1,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b1() -> Self {
        Self {
            a: 2.0,
            b: 4.0,
            groups: 1,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b2() -> Self {
        Self {
            a: 2.5,
            b: 5.0,
            groups: 1,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b3() -> Self {
        Self {
            a: 3.0,
            b: 5.0,
            groups: 1,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b1g4() -> Self {
        Self {
            a: 2.0,
            b: 4.0,
            groups: 4,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b2g4() -> Self {
        Self {
            a: 2.5,
            b: 5.0,
            groups: 4,
            stages: [4, 6, 16, 1],
        }
    }

    pub fn b3g4() -> Self {
        Self {
            a: 3.0,
            b: 5.0,
            groups: 4,
            stages: [4, 6, 16, 1],
        }
    }
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

// A RepVGG layer has a different training time and inference time architecture.
// The latter is a simple and efficient equivalent transformation of the former
// realized by a structural reparameterization technique, where 3x3 and 1x1 convolutions
// along with identity branches and batchnorm layers are fused into a single 3x3 convolution.
fn repvgg_layer(
    has_identity: bool,
    dim: usize,
    stride: usize,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride,
        groups,
        padding: 1,
        ..Default::default()
    };

    // read and reparameterize the 1x1 conv and bn into w1 and b1
    // based on https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/byobnet.py#L543

    let conv1x1_bn = batch_norm(dim, 1e-5, vb.pp("conv_1x1.bn"))?;
    let conv1x1 = conv2d_no_bias(
        in_channels,
        out_channels,
        1,
        conv2d_cfg,
        vb.pp("conv_1x1.conv"),
    )?;

    let (mut w1, b1) = fuse_conv_bn(conv1x1.weight(), conv1x1_bn)?;

    // resize to 3x3
    w1 = w1.pad_with_zeros(D::Minus1, 1, 1)?;
    w1 = w1.pad_with_zeros(D::Minus2, 1, 1)?;

    // read and reparameterize the 3x3 conv and bn into w3 and b3
    let convkxk_bn = batch_norm(dim, 1e-5, vb.pp("conv_kxk.bn"))?;
    let conv3x3 = conv2d_no_bias(
        in_channels,
        out_channels,
        3,
        conv2d_cfg,
        vb.pp("conv_kxk.conv"),
    )?;

    let (w3, b3) = fuse_conv_bn(conv3x3.weight(), convkxk_bn)?;

    let mut w = (w1 + w3)?;
    let mut b = (b1 + b3)?;

    // read and reparameterize the identity bn into wi and bi
    if has_identity {
        let identity_bn = batch_norm(dim, 1e-5, vb.pp("identity"))?;

        // create a 3x3 convolution equivalent to the identity branch
        let mut weights: Vec<f32> = vec![0.0; conv3x3.weight().elem_count()];

        // https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/byobnet.py#L620
        let in_dim = in_channels / groups;
        for i in 0..in_channels {
            weights[i * in_dim * 3 * 3 + (i % in_dim) * 3 * 3 + 4] = 1.0;
        }

        let weights = &Tensor::from_vec(weights, w.shape(), w.device())?;
        let (wi, bi) = fuse_conv_bn(weights, identity_bn)?;

        w = (w + wi)?;
        b = (b + bi)?;
    }

    // create the 3x3 conv equivalent to the sum of 3x3, 1x1 and identity branches
    let reparam_conv = Conv2d::new(w, Some(b), conv2d_cfg);

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&reparam_conv)?.relu()?;
        Ok(xs)
    }))
}

// Get the number of output channels per stage taking into account the multipliers
fn output_channels_per_stage(a: f32, b: f32, stage: usize) -> usize {
    let channels = CHANNELS_PER_STAGE[stage] as f32;

    match stage {
        0 => std::cmp::min(64, (channels * a) as usize),
        4 => (channels * b) as usize,
        _ => (channels * a) as usize,
    }
}

// Each stage is made of layers. The first layer always downsamples with stride 2.
// All but the first layer have a residual connection.
// The G4 variants have a groupwise convolution instead of a dense one on odd layers
// counted across stage boundaries, so we keep track of which layer we are in the
// full model.
fn repvgg_stage(cfg: &Config, idx: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let nlayers = cfg.stages[idx - 1];
    let mut layers = Vec::with_capacity(nlayers);
    let prev_layers: usize = cfg.stages[..idx - 1].iter().sum();
    let out_channels_prev = output_channels_per_stage(cfg.a, cfg.b, idx - 1);
    let out_channels = output_channels_per_stage(cfg.a, cfg.b, idx);

    for layer_idx in 0..nlayers {
        let (has_identity, stride, in_channels) = if layer_idx == 0 {
            (false, 2, out_channels_prev)
        } else {
            (true, 1, out_channels)
        };

        let groups = if (prev_layers + layer_idx) % 2 == 1 {
            cfg.groups
        } else {
            1
        };

        layers.push(repvgg_layer(
            has_identity,
            out_channels,
            stride,
            in_channels,
            out_channels,
            groups,
            vb.pp(layer_idx),
        )?)
    }

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for layer in layers.iter() {
            xs = xs.apply(layer)?
        }
        Ok(xs)
    }))
}

// Build a RepVGG model for a given configuration.
fn repvgg_model(config: &Config, nclasses: Option<usize>, vb: VarBuilder) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let outputs = output_channels_per_stage(config.a, config.b, 4);
            let linear = linear(outputs, nclasses, vb.pp("head.fc"))?;
            Some(linear)
        }
    };

    let stem_dim = output_channels_per_stage(config.a, config.b, 0);
    let stem = repvgg_layer(false, stem_dim, 2, 3, stem_dim, 1, vb.pp("stem"))?;
    let vb = vb.pp("stages");
    let stage1 = repvgg_stage(config, 1, vb.pp(0))?;
    let stage2 = repvgg_stage(config, 2, vb.pp(1))?;
    let stage3 = repvgg_stage(config, 3, vb.pp(2))?;
    let stage4 = repvgg_stage(config, 4, vb.pp(3))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&stem)?
            .apply(&stage1)?
            .apply(&stage2)?
            .apply(&stage3)?
            .apply(&stage4)?
            .mean(D::Minus1)?
            .mean(D::Minus1)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.apply(cls),
        }
    }))
}

pub fn repvgg(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    repvgg_model(cfg, Some(nclasses), vb)
}

pub fn repvgg_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    repvgg_model(cfg, None, vb)
}
