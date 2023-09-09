// Adapted from:
// https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/modeling/tiny_vit_sam.py
#![allow(unused)]
use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Conv2dConfig, Module, VarBuilder};

#[derive(Debug)]
struct Conv2dBN {
    c: candle_nn::Conv2d,
    bn: candle_nn::BatchNorm,
}

impl Conv2dBN {
    fn new(in_: usize, out: usize, ks: usize, cfg: Conv2dConfig, vb: VarBuilder) -> Result<Self> {
        let c = candle_nn::conv2d(in_, out, ks, cfg, vb.pp("c"))?;
        let bn = candle_nn::batch_norm(out, 1e-5, vb.pp("bn"))?;
        Ok(Self { c, bn })
    }
}

impl Module for Conv2dBN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.c)?.apply(&self.bn)
    }
}

#[derive(Debug)]
struct PatchEmbed {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
}

impl PatchEmbed {
    fn new(in_chans: usize, embed_dim: usize, resolution: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv1 = Conv2dBN::new(in_chans, embed_dim / 2, 3, cfg, vb.pp("seq.0"))?;
        let conv2 = Conv2dBN::new(embed_dim / 2, embed_dim, 3, cfg, vb.pp("seq.2"))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv1)?.gelu()?.apply(&self.conv2)
    }
}

#[derive(Debug)]
struct MBConv {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    conv3: Conv2dBN,
}

impl MBConv {
    fn new(in_: usize, out: usize, expand_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = in_ * expand_ratio;
        let cfg2 = candle_nn::Conv2dConfig {
            padding: 1,
            groups: hidden,
            ..Default::default()
        };
        let conv1 = Conv2dBN::new(in_, hidden, 1, Default::default(), vb.pp("conv1"))?;
        let conv2 = Conv2dBN::new(hidden, hidden, 3, cfg2, vb.pp("conv2"))?;
        let conv3 = Conv2dBN::new(hidden, out, 1, Default::default(), vb.pp("conv3"))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
        })
    }
}

impl Module for MBConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs;
        let xs = xs
            .apply(&self.conv1)?
            .gelu()?
            .apply(&self.conv2)?
            .gelu()?
            .apply(&self.conv3)?;
        (xs + shortcut)?.gelu()
    }
}

#[derive(Debug)]
struct PatchMerging {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    conv3: Conv2dBN,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    fn new(
        input_resolution: (usize, usize),
        dim: usize,
        out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stride = if [320, 448, 576].contains(&out) { 1 } else { 2 };
        let cfg2 = candle_nn::Conv2dConfig {
            padding: 1,
            stride,
            groups: out,
            ..Default::default()
        };
        let conv1 = Conv2dBN::new(dim, out, 1, Default::default(), vb.pp("conv1"))?;
        let conv2 = Conv2dBN::new(out, out, 3, cfg2, vb.pp("conv2"))?;
        let conv3 = Conv2dBN::new(out, out, 1, Default::default(), vb.pp("conv3"))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            input_resolution,
        })
    }
}

impl Module for PatchMerging {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if xs.rank() == 3 {
            let (h, w) = self.input_resolution;
            let b = xs.dim(0)?;
            xs.reshape((b, h, w, ()))?.permute((0, 3, 1, 2))?
        } else {
            xs.clone()
        };
        xs.apply(&self.conv1)?
            .gelu()?
            .apply(&self.conv2)?
            .gelu()?
            .apply(&self.conv3)?
            .flatten_from(2)?
            .transpose(1, 2)
    }
}

#[derive(Debug)]
struct ConvLayer {
    blocks: Vec<MBConv>,
    downsample: Option<PatchMerging>,
}

impl ConvLayer {
    fn new(
        dim: usize,
        out: usize,
        input_resolution: (usize, usize),
        depth: usize,
        downsample: bool,
        conv_expand_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            let block = MBConv::new(dim, dim, conv_expand_ratio, vb_b.pp(index))?;
            blocks.push(block)
        }
        let downsample = if downsample {
            let downsample = PatchMerging::new(input_resolution, dim, out, vb.pp("downsample"))?;
            Some(downsample)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }
}

impl Module for ConvLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?
        }
        match &self.downsample {
            None => Ok(xs),
            Some(downsample) => downsample.forward(&xs),
        }
    }
}

#[derive(Debug)]
struct Mlp {
    norm: candle_nn::LayerNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl Mlp {
    fn new(in_: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::layer_norm(in_, 1e-5, vb.pp("norm"))?;
        let fc1 = candle_nn::linear(in_, hidden, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden, in_, vb.pp("fc2"))?;
        Ok(Self { norm, fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.norm)?
            .apply(&self.fc1)?
            .gelu()?
            .apply(&self.fc2)
    }
}
