//! # FastViT inference implementation based on timm
//!
//! ## Description
//! See ["FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization"](https://arxiv.org/pdf/2303.14189)
//!
//! Implementation based on [timm model](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/fastvit.py)

use candle::{Context, DType, Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, linear_no_bias, ops::sigmoid, ops::softmax,
    BatchNorm, Conv2d, Conv2dConfig, Func, VarBuilder,
};

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub exp_ratio: usize,
    pub in_channels: usize,
    pub blocks: [usize; 4],
    pub attn: bool,
    pub lkc_use_act: bool,
}

impl Config {
    pub fn t8() -> Self {
        Self {
            exp_ratio: 3,
            in_channels: 48,
            blocks: [2, 2, 4, 2],
            attn: false,
            lkc_use_act: false,
        }
    }

    pub fn t12() -> Self {
        Self {
            exp_ratio: 3,
            in_channels: 64,
            blocks: [2, 2, 6, 2],
            attn: false,
            lkc_use_act: false,
        }
    }
    pub fn s12() -> Self {
        Self {
            exp_ratio: 4,
            in_channels: 64,
            blocks: [2, 2, 6, 2],
            attn: false,
            lkc_use_act: false,
        }
    }
    pub fn sa12() -> Self {
        Self {
            exp_ratio: 4,
            in_channels: 64,
            blocks: [2, 2, 6, 2],
            attn: true,
            lkc_use_act: false,
        }
    }
    pub fn sa24() -> Self {
        Self {
            exp_ratio: 4,
            in_channels: 64,
            blocks: [4, 4, 12, 4],
            attn: true,
            lkc_use_act: false,
        }
    }
    pub fn sa36() -> Self {
        Self {
            exp_ratio: 4,
            in_channels: 64,
            blocks: [6, 6, 18, 6],
            attn: true,
            lkc_use_act: false,
        }
    }
    pub fn ma36() -> Self {
        Self {
            exp_ratio: 4,
            in_channels: 76,
            blocks: [6, 6, 18, 6],
            attn: true,
            lkc_use_act: false,
        }
    }

    // configs used by MobileCLIP's image encoder
    pub fn mci0() -> Self {
        Self {
            exp_ratio: 3,
            in_channels: 64,
            blocks: [2, 6, 10, 2],
            attn: true,
            lkc_use_act: true,
        }
    }
    pub fn mci1() -> Self {
        Self {
            exp_ratio: 3,
            in_channels: 64,
            blocks: [4, 12, 20, 4],
            attn: true,
            lkc_use_act: true,
        }
    }
    pub fn mci2() -> Self {
        Self {
            exp_ratio: 3,
            in_channels: 80,
            blocks: [4, 12, 24, 4],
            attn: true,
            lkc_use_act: true,
        }
    }
}

fn conv_norm(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride,
        padding: kernel / 2,
        groups: in_channels,
        ..Default::default()
    };

    let bn = batch_norm(out_channels, 1e-5, vb.pp("bn"))?;
    let conv = conv2d_no_bias(in_channels, out_channels, kernel, conv2d_cfg, vb.pp("conv"))?;
    let conv = conv.absorb_bn(&bn)?;
    Ok(Func::new(move |xs| {
        let xs = xs.apply(&conv)?;
        Ok(xs)
    }))
}

fn conv_mlp(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        ..Default::default()
    };

    let conv = conv_norm(dim, dim, 7, 1, vb.pp("conv"))?;
    let fc1 = conv2d(dim, dim * exp_ratio, 1, conv2d_cfg, vb.pp("fc1"))?;
    let fc2 = conv2d(dim * exp_ratio, dim, 1, conv2d_cfg, vb.pp("fc2"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.apply(&conv)?.apply(&fc1)?.gelu_erf()?.apply(&fc2)?;
        Ok(xs)
    }))
}

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
    let (gamma, beta) = bn.weight_and_bias().context("no weight-bias")?;
    let mu = bn.running_mean();
    let sigma = (bn.running_var() + bn.eps())?.sqrt();
    let gps = (gamma / sigma)?;
    let bias = (beta - mu * &gps)?;
    let weights = weights.broadcast_mul(&gps.reshape(((), 1, 1, 1))?)?;

    Ok((weights, bias))
}

fn mobileone_block(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    group_size: usize,
    use_act: bool,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let groups = if group_size == 0 {
        1
    } else {
        in_channels / group_size
    };

    let padding = kernel / 2;
    let conv2d_cfg = Conv2dConfig {
        stride,
        groups,
        padding,
        ..Default::default()
    };

    let mut w = Tensor::zeros(
        (out_channels, in_channels / groups, kernel, kernel),
        DType::F32,
        vb.device(),
    )?;
    let dim = out_channels;

    let mut b = Tensor::zeros(dim, DType::F32, vb.device())?;

    let conv_kxk_bn = batch_norm(dim, 1e-5, vb.pp("conv_kxk.0.bn"));
    let conv_kxk = conv2d_no_bias(
        in_channels,
        out_channels,
        kernel,
        conv2d_cfg,
        vb.pp("conv_kxk.0.conv"),
    );

    if let (Ok(conv), Ok(bn)) = (conv_kxk, conv_kxk_bn) {
        let (wk, bk) = fuse_conv_bn(conv.weight(), bn)?;
        w = (w + wk)?;
        b = (b + bk)?;
    };

    let conv_scale_bn = batch_norm(dim, 1e-5, vb.pp("conv_scale.bn"));
    let conv_scale = conv2d_no_bias(
        in_channels,
        out_channels,
        1,
        conv2d_cfg,
        vb.pp("conv_scale.conv"),
    );

    if let (Ok(conv), Ok(bn)) = (conv_scale, conv_scale_bn) {
        let (ws, bs) = fuse_conv_bn(conv.weight(), bn)?;
        // pad to 3x3
        let ws = ws
            .pad_with_zeros(D::Minus1, 1, 1)?
            .pad_with_zeros(D::Minus2, 1, 1)?;

        w = (w + ws)?;
        b = (b + bs)?;
    };

    let se = squeeze_and_excitation(out_channels, out_channels / 16, vb.pp("se"));

    // read and reparameterize the identity bn into wi and bi
    let identity_bn = batch_norm(dim, 1e-5, vb.pp("identity"));

    if let Ok(id_bn) = identity_bn {
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
        let (wi, bi) = fuse_conv_bn(weights, id_bn)?;

        w = (w + wi)?;
        b = (b + bi)?;
    };
    let reparam_conv = Conv2d::new(w, Some(b), conv2d_cfg);

    Ok(Func::new(move |xs| {
        let mut xs = xs.apply(&reparam_conv)?;
        if let Ok(f) = &se {
            xs = xs.apply(f)?;
        }
        if use_act {
            xs = xs.gelu_erf()?;
        };
        Ok(xs)
    }))
}

fn repmixer(dim: usize, kernel: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let gamma = vb.get((dim, 1, 1), "layer_scale.gamma")?;
    let norm = mobileone_block(dim, dim, kernel, 1, 1, false, vb.pp("norm"))?;
    let mixer = mobileone_block(dim, dim, kernel, 1, 1, false, vb.pp("mixer"))?;

    Ok(Func::new(move |xs| {
        let residual = xs.clone();
        let xs = (xs.apply(&mixer)? - xs.apply(&norm)?)?;
        let xs = xs.broadcast_mul(&gamma.reshape((1, (), 1, 1))?)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }))
}

fn repmixer_block(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let gamma = vb.get((dim, 1, 1), "layer_scale.gamma")?;
    let token_mixer = repmixer(dim, 3, vb.pp("token_mixer"))?;
    let mlp = conv_mlp(dim, exp_ratio, vb.pp("mlp"))?;

    Ok(Func::new(move |xs| {
        let residual = xs.apply(&token_mixer)?;
        let mut xs = residual.apply(&mlp)?;
        xs = xs.broadcast_mul(&gamma.reshape((1, (), 1, 1))?)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }))
}

fn positional_encoding(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: 1,
        padding: 3,
        groups: dim,
        ..Default::default()
    };

    let conv = conv2d(dim, dim, 7, conv2d_cfg, vb.pp("pos_enc"))?;

    Ok(Func::new(move |xs| {
        let xs = (xs + xs.apply(&conv)?)?;
        Ok(xs)
    }))
}

fn attention(dim: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let qkv = linear_no_bias(dim, dim * 3, vb.pp("qkv"))?;
    let proj = linear(dim, dim, vb.pp("proj"))?;
    let head_dim = 32;
    let num_heads = dim / head_dim;
    let scale = (head_dim as f64).powf(-0.5);

    Ok(Func::new(move |xs| {
        let xs = xs.clone();
        let (b, c, h, w) = xs.dims4()?;
        let n = h * w;
        let xs = xs.flatten_from(2)?.transpose(D::Minus1, D::Minus2)?;
        let qkv = xs
            .apply(&qkv)?
            .reshape((b, n, 3, num_heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let q = (q * scale)?;

        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax(&att, D::Minus1)?;
        let xs = att.matmul(&v)?;

        let xs = xs.transpose(1, 2)?.reshape((b, n, c))?;
        let xs = xs.apply(&proj)?;
        let xs = xs.transpose(D::Minus1, D::Minus2)?.reshape((b, c, h, w))?;

        Ok(xs)
    }))
}

fn attention_block(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let gamma1 = vb.get((dim, 1, 1), "layer_scale_1.gamma")?;
    let gamma2 = vb.get((dim, 1, 1), "layer_scale_2.gamma")?;
    let norm = batch_norm(dim, 1e-5, vb.pp("norm"))?;
    let token_mixer = attention(dim, vb.pp("token_mixer"))?;
    let mlp = conv_mlp(dim, exp_ratio, vb.pp("mlp"))?;

    Ok(Func::new(move |xs| {
        let xs = xs.clone();
        let xs = (&xs
            + &xs
                .apply_t(&norm, false)?
                .apply(&token_mixer)?
                .broadcast_mul(&gamma1.reshape((1, (), 1, 1))?)?)?;

        let xs = (&xs
            + &xs
                .apply(&mlp)?
                .broadcast_mul(&gamma2.reshape((1, (), 1, 1))?)?)?;

        Ok(xs)
    }))
}

fn fastvit_stage(cfg: &Config, idx: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let nblocks = cfg.blocks[idx];
    let mut blocks = Vec::with_capacity(nblocks);

    let dim = cfg.in_channels << idx;
    let downsample = fastvit_patch_embed(dim / 2, dim, cfg.lkc_use_act, vb.pp("downsample"));
    for block_idx in 0..nblocks {
        let block = if cfg.attn && idx == 3 {
            attention_block(dim, cfg.exp_ratio, vb.pp(format!("blocks.{block_idx}")))?
        } else {
            repmixer_block(dim, cfg.exp_ratio, vb.pp(format!("blocks.{block_idx}")))?
        };
        blocks.push(block);
    }
    let pos_emb = positional_encoding(dim, vb.pp("pos_emb"));

    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        if let Ok(ds) = &downsample {
            xs = xs.apply(ds)?;
        }
        if let Ok(pos) = &pos_emb {
            xs = xs.apply(pos)?;
        }
        for block in blocks.iter() {
            xs = xs.apply(block)?;
        }
        Ok(xs)
    }))
}

fn fastvit_patch_embed(
    in_channels: usize,
    out_channels: usize,
    use_act: bool,
    vb: VarBuilder,
) -> Result<Func<'static>> {
    let lk = conv_norm(in_channels, out_channels, 7, 2, vb.pp("proj.0.large_conv"))?;
    let sk = conv_norm(in_channels, out_channels, 3, 2, vb.pp("proj.0.small_conv"))?;
    let se = squeeze_and_excitation(out_channels, out_channels / 4, vb.pp("proj.0.se"));
    let mb = mobileone_block(out_channels, out_channels, 1, 1, 0, true, vb.pp("proj.1"))?;

    Ok(Func::new(move |xs| {
        let mut xs = (xs.apply(&lk)? + xs.apply(&sk)?)?;
        if let Ok(f) = &se {
            xs = xs.apply(f)?;
        }
        if use_act {
            xs = xs.gelu_erf()?;
        };
        let xs = xs.apply(&mb)?;
        Ok(xs)
    }))
}

fn fastvit_stem(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Func<'static>> {
    let mb0 = mobileone_block(in_channels, out_channels, 3, 2, 0, true, vb.pp(0))?;
    let mb1 = mobileone_block(out_channels, out_channels, 3, 2, 1, true, vb.pp(1))?;
    let mb2 = mobileone_block(out_channels, out_channels, 1, 1, 0, true, vb.pp(2))?;
    Ok(Func::new(move |xs| {
        let xs = xs.apply(&mb0)?.apply(&mb1)?.apply(&mb2)?;
        Ok(xs)
    }))
}

// Build a fastvit model for a given configuration.
fn fastvit_model(cfg: &Config, nclasses: Option<usize>, vb: VarBuilder) -> Result<Func<'static>> {
    let cls = match nclasses {
        None => None,
        Some(nclasses) => {
            let linear = linear(cfg.in_channels * 16, nclasses, vb.pp("head.fc"))?;
            Some(linear)
        }
    };

    let stem = fastvit_stem(3, cfg.in_channels, vb.pp("stem"))?;
    let final_conv = mobileone_block(
        cfg.in_channels * 8,
        cfg.in_channels * 16,
        3,
        1,
        1,
        true,
        vb.pp("final_conv"),
    )?;

    let vb = vb.pp("stages");
    let stage1 = fastvit_stage(cfg, 0, vb.pp(0))?;
    let stage2 = fastvit_stage(cfg, 1, vb.pp(1))?;
    let stage3 = fastvit_stage(cfg, 2, vb.pp(2))?;
    let stage4 = fastvit_stage(cfg, 3, vb.pp(3))?;

    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&stem)?
            .apply(&stage1)?
            .apply(&stage2)?
            .apply(&stage3)?
            .apply(&stage4)?
            .apply(&final_conv)?;
        match &cls {
            None => Ok(xs),
            Some(cls) => xs.mean(D::Minus2)?.mean(D::Minus1)?.apply(cls),
        }
    }))
}

pub fn fastvit(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Func<'static>> {
    fastvit_model(cfg, Some(nclasses), vb)
}

pub fn fastvit_no_final_layer(cfg: &Config, vb: VarBuilder) -> Result<Func<'static>> {
    fastvit_model(cfg, None, vb)
}
