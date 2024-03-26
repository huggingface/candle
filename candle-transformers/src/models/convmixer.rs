use candle::Result;
use candle_nn::{batch_norm, Conv2dConfig, Module, VarBuilder};

#[allow(clippy::many_single_char_names)]
fn conv2d_same(
    i: usize,
    o: usize,
    k: usize,
    c: Conv2dConfig,
    vb: VarBuilder,
) -> Result<impl Module> {
    let conv2d = candle_nn::conv2d(i, o, k, c, vb)?;
    let s = c.stride;
    let module = candle_nn::func(move |xs| {
        let ih = xs.dim(2)?;
        let iw = xs.dim(3)?;
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = usize::max((oh - 1) * s + k - ih, 0);
        let pad_w = usize::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            xs.pad_with_zeros(3, pad_w / 2, pad_w - pad_w / 2)?
                .pad_with_zeros(2, pad_h / 2, pad_h - pad_h / 2)?
                .apply(&conv2d)
        } else {
            xs.apply(&conv2d)
        }
    });
    Ok(module)
}

fn block(dim: usize, kernel_size: usize, vb: VarBuilder) -> Result<impl Module> {
    let conv2d_cfg = Conv2dConfig {
        groups: dim,
        ..Default::default()
    };
    let vb_fn = vb.pp(0).pp("fn");
    let conv1 = conv2d_same(dim, dim, kernel_size, conv2d_cfg, vb_fn.pp(0))?;
    let bn1 = batch_norm(dim, 1e-5, vb_fn.pp(2))?;
    let conv2 = candle_nn::conv2d(dim, dim, 1, Default::default(), vb.pp(1))?;
    let bn2 = batch_norm(dim, 1e-5, vb.pp(3))?;
    Ok(candle_nn::func(move |xs| {
        let ys = xs.apply(&conv1)?.gelu_erf()?.apply_t(&bn1, false)?;
        (xs + ys)?.apply(&conv2)?.gelu_erf()?.apply_t(&bn2, false)
    }))
}

fn convmixer(
    nclasses: usize,
    dim: usize,
    depth: usize,
    kernel_size: usize,
    patch_size: usize,
    vb: VarBuilder,
) -> Result<candle_nn::Func<'static>> {
    let conv2d_cfg = Conv2dConfig {
        stride: patch_size,
        ..Default::default()
    };
    let conv1 = candle_nn::conv2d(3, dim, patch_size, conv2d_cfg, vb.pp(0))?;
    let bn1 = batch_norm(dim, 1e-5, vb.pp(2))?;
    let blocks: Vec<_> = (0..depth)
        .map(|index| block(dim, kernel_size, vb.pp(3 + index)))
        .collect::<Result<Vec<_>>>()?;
    let fc = candle_nn::linear(dim, nclasses, vb.pp(25))?;
    Ok(candle_nn::func(move |xs| {
        let mut xs = xs.apply(&conv1)?.gelu_erf()?.apply_t(&bn1, false)?;
        for block in blocks.iter() {
            xs = xs.apply(block)?
        }
        // This performs the adaptive average pooling with a target size of (1, 1).
        xs.mean(3)?.mean(2)?.apply(&fc)
    }))
}

pub fn c1536_20(nclasses: usize, vb: VarBuilder) -> Result<candle_nn::Func<'static>> {
    convmixer(nclasses, 1536, 20, 9, 7, vb)
}

pub fn c1024_20(nclasses: usize, vb: VarBuilder) -> Result<candle_nn::Func<'static>> {
    convmixer(nclasses, 1024, 20, 9, 14, vb)
}
