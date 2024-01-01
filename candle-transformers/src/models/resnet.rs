//! ResNet implementation.
//!
//! See "Deep Residual Learning for Image Recognition" He et al. 2015
//! <https://arxiv.org/abs/1512.03385>
use candle::{Result, D};
use candle_nn::{batch_norm, Conv2d, Func, VarBuilder};

fn conv2d(
    c_in: usize,
    c_out: usize,
    ksize: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let conv2d_cfg = candle_nn::Conv2dConfig {
        stride,
        padding,
        ..Default::default()
    };
    candle_nn::conv2d_no_bias(c_in, c_out, ksize, conv2d_cfg, vb)
}

fn downsample(c_in: usize, c_out: usize, stride: usize, vb: VarBuilder) -> Result<Func> {
    if stride != 1 || c_in != c_out {
        let conv = conv2d(c_in, c_out, 1, 0, stride, vb.pp(0))?;
        let bn = batch_norm(c_out, 1e-5, vb.pp(1))?;
        Ok(Func::new(move |xs| xs.apply(&conv)?.apply_t(&bn, false)))
    } else {
        Ok(Func::new(|xs| Ok(xs.clone())))
    }
}

fn basic_block(c_in: usize, c_out: usize, stride: usize, vb: VarBuilder) -> Result<Func> {
    let conv1 = conv2d(c_in, c_out, 3, 1, stride, vb.pp("conv1"))?;
    let bn1 = batch_norm(c_out, 1e-5, vb.pp("bn1"))?;
    let conv2 = conv2d(c_out, c_out, 3, 1, 1, vb.pp("conv2"))?;
    let bn2 = batch_norm(c_out, 1e-5, vb.pp("bn2"))?;
    let downsample = downsample(c_in, c_out, stride, vb.pp("downsample"))?;
    Ok(Func::new(move |xs| {
        let ys = xs
            .apply(&conv1)?
            .apply_t(&bn1, false)?
            .relu()?
            .apply(&conv2)?
            .apply_t(&bn2, false)?;
        (xs.apply(&downsample)? + ys)?.relu()
    }))
}

fn basic_layer(
    c_in: usize,
    c_out: usize,
    stride: usize,
    cnt: usize,
    vb: VarBuilder,
) -> Result<Func> {
    let mut layers = Vec::with_capacity(cnt);
    for index in 0..cnt {
        let l_in = if index == 0 { c_in } else { c_out };
        let stride = if index == 0 { stride } else { 1 };
        layers.push(basic_block(l_in, c_out, stride, vb.pp(index))?)
    }
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for layer in layers.iter() {
            xs = xs.apply(layer)?
        }
        Ok(xs)
    }))
}

fn resnet(
    nclasses: Option<usize>,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize,
    vb: VarBuilder,
) -> Result<Func> {
    let conv1 = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
    let bn1 = batch_norm(64, 1e-5, vb.pp("bn1"))?;
    let layer1 = basic_layer(64, 64, 1, c1, vb.pp("layer1"))?;
    let layer2 = basic_layer(64, 128, 2, c2, vb.pp("layer2"))?;
    let layer3 = basic_layer(128, 256, 2, c3, vb.pp("layer3"))?;
    let layer4 = basic_layer(256, 512, 2, c4, vb.pp("layer4"))?;
    let fc = match nclasses {
        None => None,
        Some(nclasses) => {
            let linear = candle_nn::linear(512, nclasses, vb.pp("fc"))?;
            Some(linear)
        }
    };
    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv1)?
            .apply_t(&bn1, false)?
            .relu()?
            .pad_with_same(D::Minus1, 1, 1)?
            .pad_with_same(D::Minus2, 1, 1)?
            .max_pool2d_with_stride(3, 2)?
            .apply(&layer1)?
            .apply(&layer2)?
            .apply(&layer3)?
            .apply(&layer4)?
            .mean(D::Minus1)?
            .mean(D::Minus1)?;
        match &fc {
            None => Ok(xs),
            Some(fc) => xs.apply(fc),
        }
    }))
}

/// Creates a ResNet-18 model.
pub fn resnet18(num_classes: usize, vb: VarBuilder) -> Result<Func> {
    resnet(Some(num_classes), 2, 2, 2, 2, vb)
}

pub fn resnet18_no_final_layer(vb: VarBuilder) -> Result<Func> {
    resnet(None, 2, 2, 2, 2, vb)
}

/// Creates a ResNet-34 model.
pub fn resnet34(num_classes: usize, vb: VarBuilder) -> Result<Func> {
    resnet(Some(num_classes), 3, 4, 6, 3, vb)
}

pub fn resnet34_no_final_layer(vb: VarBuilder) -> Result<Func> {
    resnet(None, 3, 4, 6, 3, vb)
}

// Bottleneck versions for ResNet 50, 101, and 152.
fn bottleneck_block(
    c_in: usize,
    c_out: usize,
    stride: usize,
    e: usize,
    vb: VarBuilder,
) -> Result<Func> {
    let e_dim = e * c_out;
    let conv1 = conv2d(c_in, c_out, 1, 0, 1, vb.pp("conv1"))?;
    let bn1 = batch_norm(c_out, 1e-5, vb.pp("bn1"))?;
    let conv2 = conv2d(c_out, c_out, 3, 1, stride, vb.pp("conv2"))?;
    let bn2 = batch_norm(c_out, 1e-5, vb.pp("bn2"))?;
    let conv3 = conv2d(c_out, e_dim, 1, 0, 1, vb.pp("conv3"))?;
    let bn3 = batch_norm(e_dim, 1e-5, vb.pp("bn3"))?;
    let downsample = downsample(c_in, e_dim, stride, vb.pp("downsample"))?;
    Ok(Func::new(move |xs| {
        let ys = xs
            .apply(&conv1)?
            .apply_t(&bn1, false)?
            .relu()?
            .apply(&conv2)?
            .apply_t(&bn2, false)?
            .relu()?
            .apply(&conv3)?
            .apply_t(&bn3, false)?;
        (xs.apply(&downsample)? + ys)?.relu()
    }))
}

fn bottleneck_layer(
    c_in: usize,
    c_out: usize,
    stride: usize,
    cnt: usize,
    vb: VarBuilder,
) -> Result<Func> {
    let mut layers = Vec::with_capacity(cnt);
    for index in 0..cnt {
        let l_in = if index == 0 { c_in } else { 4 * c_out };
        let stride = if index == 0 { stride } else { 1 };
        layers.push(bottleneck_block(l_in, c_out, stride, 4, vb.pp(index))?)
    }
    Ok(Func::new(move |xs| {
        let mut xs = xs.clone();
        for layer in layers.iter() {
            xs = xs.apply(layer)?
        }
        Ok(xs)
    }))
}

fn bottleneck_resnet(
    nclasses: Option<usize>,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize,
    vb: VarBuilder,
) -> Result<Func> {
    let conv1 = conv2d(3, 64, 7, 3, 2, vb.pp("conv1"))?;
    let bn1 = batch_norm(64, 1e-5, vb.pp("bn1"))?;
    let layer1 = bottleneck_layer(64, 64, 1, c1, vb.pp("layer1"))?;
    let layer2 = bottleneck_layer(4 * 64, 128, 2, c2, vb.pp("layer2"))?;
    let layer3 = bottleneck_layer(4 * 128, 256, 2, c3, vb.pp("layer3"))?;
    let layer4 = bottleneck_layer(4 * 256, 512, 2, c4, vb.pp("layer4"))?;
    let fc = match nclasses {
        None => None,
        Some(nclasses) => {
            let linear = candle_nn::linear(4 * 512, nclasses, vb.pp("fc"))?;
            Some(linear)
        }
    };
    Ok(Func::new(move |xs| {
        let xs = xs
            .apply(&conv1)?
            .apply_t(&bn1, false)?
            .relu()?
            .pad_with_same(D::Minus1, 1, 1)?
            .pad_with_same(D::Minus2, 1, 1)?
            .max_pool2d_with_stride(3, 2)?
            .apply(&layer1)?
            .apply(&layer2)?
            .apply(&layer3)?
            .apply(&layer4)?
            .mean(D::Minus1)?
            .mean(D::Minus1)?;
        match &fc {
            None => Ok(xs),
            Some(fc) => xs.apply(fc),
        }
    }))
}

pub fn resnet50(num_classes: usize, vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(Some(num_classes), 3, 4, 6, 3, vb)
}

pub fn resnet50_no_final_layer(vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(None, 3, 4, 6, 3, vb)
}

pub fn resnet101(num_classes: usize, vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(Some(num_classes), 3, 4, 23, 3, vb)
}

pub fn resnet101_no_final_layer(vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(None, 3, 4, 23, 3, vb)
}

pub fn resnet152(num_classes: usize, vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(Some(num_classes), 3, 8, 36, 3, vb)
}

pub fn resnet152_no_final_layer(vb: VarBuilder) -> Result<Func> {
    bottleneck_resnet(None, 3, 8, 36, 3, vb)
}
