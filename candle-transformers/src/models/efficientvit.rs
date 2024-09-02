//! EfficientViT (MSRA) inference implementation based on timm.
//!
//! See "EfﬁcientViT: Memory Efﬁcient Vision Transformer with Cascaded Group Attention"
//! https://arxiv.org/abs/2305.07027

//! https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvit_msra.py

use candle::{Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, ops::sigmoid, ops::softmax, Conv2dConfig, Func,
    VarBuilder,
};

#[derive(Clone)]
pub struct Config {
    channels: [usize; 3],
    blocks: [usize; 3],
    heads: [usize; 3],
    kernels: [usize; 4],
}

impl Config {
    pub fn m0() -> Self {
        Self {
            channels: [64, 128, 192],
            blocks: [1, 2, 3],
            heads: [4, 4, 4],
            kernels: [5, 5, 5, 5],
        }
    }
    pub fn m1() -> Self {
        Self {
            channels: [128, 144, 192],
            blocks: [1, 2, 3],
            heads: [2, 3, 3],
            kernels: [7, 5, 3, 3],
        }
    }
    pub fn m2() -> Self {
        Self {
            channels: [128, 192, 224],
            blocks: [1, 2, 3],
            heads: [4, 3, 2],
            kernels: [7, 5, 3, 3],
        }
    }
    pub fn m3() -> Self {
        Self {
            channels: [128, 240, 320],
            blocks: [1, 2, 3],
            heads: [4, 3, 4],
            kernels: [5, 5, 5, 5],
        }
    }
    pub fn m4() -> Self {
        Self {
            channels: [128, 256, 384],
            blocks: [1, 2, 3],
            heads: [4, 4, 4],
            kernels: [7, 5, 3, 3],
        }
    }

    pub fn m5() -> Self {
        Self {
            channels: [192, 288, 384],
            blocks: [1, 3, 4],
            heads: [3, 3, 4],
            kernels: [7, 5, 3, 3],
        }
    }
}

fn efficientvit_stemblock(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 2,
        padding: 1,
        ..Default::default()
    };

    let bn = batch_norm(out_channels, 1e-5, vb.pp("bn"))?;
    let conv = conv2d_no_bias(in_channels, out_channels, 3, conv2d_cfg, vb.pp("conv"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&conv)?.apply_t(&bn, false)?;
        Ok(xs)
    }))
}

fn efficientvit_stem(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv1 = efficientvit_stemblock(3, dim / 8, vb.pp("conv1"))?;
    let conv2 = efficientvit_stemblock(dim / 8, dim / 4, vb.pp("conv2"))?;
    let conv3 = efficientvit_stemblock(dim / 4, dim / 2, vb.pp("conv3"))?;
    let conv4 = efficientvit_stemblock(dim / 2, dim, vb.pp("conv4"))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv1)?
            .relu()?
            .apply(&conv2)?
            .relu()?
            .apply(&conv3)?
            .relu()?
            .apply(&conv4)?;

        Ok(xs)
    }))
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

fn conv_mlp(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let pw1 = pointwise_conv(in_channels, out_channels, vb.pp("pw1"))?;
    let pw2 = pointwise_conv(out_channels, in_channels, vb.pp("pw2"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&pw1)?.relu()?.apply(&pw2)?;
        Ok(xs)
    }))
}

// Fixed per-stage resolutions
const RESOLUTIONS: [usize; 3] = [14, 7, 4];

// Attention block
fn efficientvit_attn(
    cfg: &Config,
    stage: usize,
    in_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let cga = cascaded_group_attn(cfg, stage, in_channels, vb)?;

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();

        let (b, c, h, w) = xs.dims4()?;
        let win_res = 7; // Fixed window resolution
        let pad_b = (win_res - h % win_res) % win_res;
        let pad_r = (win_res - w % win_res) % win_res;
        let ph = h + pad_b;
        let pw = w + pad_r;
        let nh = ph / win_res;
        let nw = pw / win_res;

        if RESOLUTIONS[stage] > win_res {
            xs = xs.permute((0, 2, 3, 1))?;
            xs = xs.pad_with_zeros(D::Minus1, 0, pad_r)?;
            xs = xs.pad_with_zeros(D::Minus2, 0, pad_b)?;
            xs = xs
                .reshape((b, nh, win_res, nw, win_res, c))?
                .transpose(2, 3)?;
            xs = xs
                .reshape((b * nh * nw, win_res, win_res, c))?
                .permute((0, 3, 1, 2))?;
        }

        xs = xs.apply(&cga)?;

        if RESOLUTIONS[stage] > win_res {
            xs = xs
                .permute((0, 2, 3, 1))?
                .reshape((b, nh, nw, win_res, win_res, c))?;
            xs = xs.transpose(2, 3)?.reshape((b, ph, pw, c))?;
            xs = xs.permute((0, 3, 1, 2))?;
        }

        Ok(xs)
    }))
}

// Cascaded group attention
fn cascaded_group_attn(
    cfg: &Config,
    stage: usize,
    in_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let heads = cfg.heads[stage];
    let key_dim = 16;

    let val_dim = in_channels / heads;

    let scale = (key_dim as f64).powf(-0.5);

    let mut dws = Vec::with_capacity(heads);
    let mut qkvs = Vec::with_capacity(heads);
    for i in 0..heads {
        dws.push(depthwise_conv(
            key_dim,
            cfg.kernels[i],
            1,
            cfg.kernels[i] / 2,
            vb.pp(format!("dws.{i}")),
        )?);

        qkvs.push(pointwise_conv(
            in_channels / heads,
            in_channels / heads + 2 * key_dim,
            vb.pp(format!("qkvs.{i}")),
        )?);
    }
    let proj = pointwise_conv(in_channels, in_channels, vb.pp("proj.1"))?;

    Ok(Func::new(move |xs| {
        let (b, _, h, w) = xs.dims4()?;
        let feats_in = xs.chunk(heads, 1)?;
        let mut feats_out = Vec::with_capacity(heads);
        let mut feat = feats_in[0].clone();

        for i in 0..heads {
            if i > 0 {
                feat = (&feat + &feats_in[i])?;
            }
            feat = feat.apply(&qkvs[i])?;
            let res = feat.reshape((b, (), h, w))?;
            let q = res.narrow(1, 0, key_dim)?;
            let k = res.narrow(1, key_dim, key_dim)?;
            let v = res.narrow(1, 2 * key_dim, val_dim)?;

            let q = q.apply(&dws[i])?;

            let q = q.flatten_from(2)?;
            let k = k.flatten_from(2)?;
            let v = v.flatten_from(2)?;
            let q = (q * scale)?;

            let att = q.transpose(D::Minus2, D::Minus1)?.matmul(&k)?;
            let att = softmax(&att, D::Minus1)?;
            feat = v.matmul(&att.transpose(D::Minus2, D::Minus1)?)?;
            feat = feat.reshape((b, val_dim, h, w))?;
            feats_out.push(feat.clone());
        }

        let xs = Tensor::cat(&feats_out, 1)?;
        let xs = xs.relu()?.apply(&proj)?;

        Ok(xs)
    }))
}

// Used by the downsampling layer
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

// Used by the downsampling layer
fn patchmerge(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let dim = in_channels;
    let hid_dim = in_channels * 4;
    let conv1 = pointwise_conv(dim, hid_dim, vb.pp("conv1"))?;
    let conv2 = depthwise_conv(hid_dim, 3, 2, 1, vb.pp("conv2"))?;
    let conv3 = pointwise_conv(hid_dim, out_channels, vb.pp("conv3"))?;
    let se = squeeze_and_excitation(hid_dim, hid_dim / 4, vb.pp("se"))?;
    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv1)?
            .relu()?
            .apply(&conv2)?
            .relu()?
            .apply(&se)?
            .apply(&conv3)?;
        Ok(xs)
    }))
}

// Used by the downsampling layer
fn res(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let dw = depthwise_conv(dim, 3, 1, 1, vb.pp("0.m"))?;
    let mlp = conv_mlp(dim, dim * 2, vb.pp("1.m"))?;
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        xs = (&xs + &xs.apply(&dw)?)?;
        xs = (&xs + &xs.apply(&mlp)?)?;
        Ok(xs)
    }))
}

// Downsampling
fn efficientvit_downsample(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let res1 = res(in_channels, vb.pp("res1"))?;
    let res2 = res(out_channels, vb.pp("res2"))?;
    let patchmerge = patchmerge(in_channels, out_channels, vb.pp("patchmerge"))?;
    Ok(Func::new(move |xs| {
        let xs = xs.apply(&res1)?.apply(&patchmerge)?.apply(&res2)?;
        Ok(xs)
    }))
}

fn efficientvit_block(
    cfg: &Config,
    stage: usize,
    dim: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let dw0 = depthwise_conv(dim, 3, 1, 1, vb.pp("dw0.m"))?;
    let dw1 = depthwise_conv(dim, 3, 1, 1, vb.pp("dw1.m"))?;
    let ffn0 = conv_mlp(dim, dim * 2, vb.pp("ffn0.m"))?;
    let ffn1 = conv_mlp(dim, dim * 2, vb.pp("ffn1.m"))?;
    let attn = efficientvit_attn(cfg, stage, dim, vb.pp("mixer.m.attn"))?;
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        xs = (&xs + &xs.apply(&dw0)?)?;
        xs = (&xs + &xs.apply(&ffn0)?)?;
        xs = (&xs + &xs.apply(&attn)?)?;
        xs = (&xs + &xs.apply(&dw1)?)?;
        xs = (&xs + &xs.apply(&ffn1)?)?;
        Ok(xs)
    }))
}

// Each stage is made of blocks. There is a downsampling layer between stages.
fn efficientvit_stage(cfg: &Config, stage: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let nblocks = cfg.blocks[stage];
    let mut blocks = Vec::with_capacity(nblocks + 1);

    let in_channels = if stage > 0 {
        cfg.channels[stage - 1]
    } else {
        cfg.channels[0]
    };
    let out_channels = cfg.channels[stage];

    if stage > 0 {
        blocks.push(efficientvit_downsample(
            in_channels,
            out_channels,
            vb.pp("downsample"),
        )?);
    }

    for i in 0..nblocks {
        blocks.push(efficientvit_block(
            cfg,
            stage,
            out_channels,
            vb.pp(format!("blocks.{i}")),
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
fn efficientvit_head(outputs: usize, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let norm = batch_norm(outputs, 1e-6, vb.pp("bn"))?;
    let linear = linear(outputs, nclasses, vb.pp("linear"))?;
    Ok(Func::new(move |xs| {
        xs.apply_t(&norm, false)?.apply(&linear)
    }))
}

// Build a efficientvit model for a given configuration.
fn efficientvit_model(
    config: &Config,
    nclasses: Option<usize>,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let outputs = config.channels[2];
            let head = efficientvit_head(outputs, nclasses, vb.pp("head"))?;
            Some(head)
        }
    };

    let stem_dim = config.channels[0];
    let stem = efficientvit_stem(stem_dim, vb.pp("patch_embed"))?;

    let vb = vb.pp("stages");
    let stage1 = efficientvit_stage(config, 0, vb.pp(0))?;
    let stage2 = efficientvit_stage(config, 1, vb.pp(1))?;
    let stage3 = efficientvit_stage(config, 2, vb.pp(2))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&stem)?
            .apply(&stage1)?
            .apply(&stage2)?
            .apply(&stage3)?
            .mean(D::Minus2)?
            .mean(D::Minus1)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.apply(cls),
        }
    }))
}

pub fn efficientvit(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    efficientvit_model(cfg, Some(nclasses), vb)
}

pub fn efficientvit_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    efficientvit_model(cfg, None, vb)
}
