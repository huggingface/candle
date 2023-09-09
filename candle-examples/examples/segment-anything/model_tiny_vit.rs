// Adapted from:
// https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/modeling/tiny_vit_sam.py
#![allow(unused)]
use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2dConfig, Module, VarBuilder};

const MBCONV_EXPAND_RATIO: usize = 4;
const MLP_RATIO: usize = 4;
const LOCAL_CONV_SIZE: usize = 3;

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
    fn new(in_chans: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
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

#[derive(Debug)]
struct Attention {
    norm: candle_nn::LayerNorm,
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    attention_biases: Tensor,
    ab: Tensor,
    key_dim: usize,
    num_heads: usize,
    d: usize,
    dh: usize,
    scale: f64,
}

impl Attention {
    fn new(
        dim: usize,
        key_dim: usize,
        num_heads: usize,
        attn_ratio: usize,
        resolution: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let d = attn_ratio * key_dim;
        let dh = d * num_heads;
        let nh_kd = key_dim * num_heads;
        let h = dh + nh_kd * 2;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        let qkv = candle_nn::linear(dim, h, vb.pp("qkv"))?;
        let proj = candle_nn::linear(dh, dim, vb.pp("proj"))?;
        // TODO: replace 0, get the proper ab
        let attention_biases = vb.get((num_heads, 0), "attention_biases")?;
        let ab = attention_biases.clone();
        Ok(Self {
            norm,
            qkv,
            proj,
            attention_biases,
            ab,
            key_dim,
            num_heads,
            d,
            dh,
            scale: 1f64 / (key_dim as f64).sqrt(),
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let xs = xs.apply(&self.norm)?;
        let qkv = xs.apply(&self.qkv)?.reshape((b, n, self.num_heads, ()))?;
        let q = qkv
            .narrow(D::Minus1, 0, self.key_dim)?
            .permute((0, 2, 1, 3))?;
        let k = qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .permute((0, 2, 1, 3))?;
        let v = qkv
            .narrow(D::Minus1, 2 * self.key_dim, self.d)?
            .permute((0, 2, 1, 3))?;
        let attn = (q.matmul(&k.t()?)? * self.scale)?;
        let attn = (attn + &self.ab)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, n, self.dh))?
            .apply(&self.proj)
    }
}

#[derive(Debug)]
struct TinyViTBlock {
    attn: Attention,
    local_conv: Conv2dBN,
    mlp: Mlp,
    window_size: usize,
}

impl TinyViTBlock {
    fn new(
        dim: usize,
        input_resolution: (usize, usize),
        num_heads: usize,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let attn = Attention::new(
            dim,
            head_dim,
            num_heads,
            1,
            (window_size, window_size),
            vb.pp("attn"),
        )?;
        let mlp = Mlp::new(dim, dim * MLP_RATIO, vb.pp("mlp"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: LOCAL_CONV_SIZE / 2,
            ..Default::default()
        };
        let local_conv = Conv2dBN::new(dim, dim, LOCAL_CONV_SIZE, cfg, vb.pp("local_conv"))?;
        Ok(Self {
            attn,
            local_conv,
            mlp,
            window_size,
        })
    }
}

impl Module for TinyViTBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
struct BasicLayer {
    blocks: Vec<TinyViTBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        input_resolution: (usize, usize),
        depth: usize,
        num_heads: usize,
        window_size: usize,
        downsample: bool,
        out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_b = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            let block = TinyViTBlock::new(
                dim,
                input_resolution,
                num_heads,
                window_size,
                vb_b.pp(index),
            )?;
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

impl Module for BasicLayer {
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
struct TinyViT {
    patch_embed: PatchEmbed,
    layer0: ConvLayer,
    layers: Vec<BasicLayer>,
    norm_head: candle_nn::LayerNorm,
    head: candle_nn::Linear,
    neck_conv1: candle_nn::Conv2d,
    neck_ln1: crate::LayerNorm2d,
    neck_conv2: candle_nn::Conv2d,
    neck_ln2: crate::LayerNorm2d,
}

impl TinyViT {
    #[allow(clippy::too_many_arguments)]
    fn new(
        img_size: usize,
        in_: usize,
        embed_dims: &[usize],
        depths: &[usize],
        num_heads: &[usize],
        window_sizes: &[usize],
        num_classes: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(in_, embed_dims[0], vb.pp("patch_embed"))?;
        let patches_resolution = img_size / 4;

        let vb_l = vb.pp("layers");
        let layer0 = ConvLayer::new(
            /* dim */ embed_dims[0],
            /* out */ embed_dims[1],
            /* input_resolution */ (patches_resolution, patches_resolution),
            /* depth */ depths[0],
            /* downsample */ true,
            /* conv_expand_ratio */ MBCONV_EXPAND_RATIO,
            vb_l.pp(0),
        )?;

        let num_layers = embed_dims.len();
        let mut layers = Vec::with_capacity(num_layers - 1);
        for i_layer in 1..num_layers {
            let patches_resolution = patches_resolution / (1 << usize::min(i_layer, 2));
            let layer = BasicLayer::new(
                /* dim */ embed_dims[i_layer],
                /* input_resolution */ (patches_resolution, patches_resolution),
                /* depth */ depths[i_layer],
                /* num_heads */ num_heads[i_layer],
                /* window_size */ window_sizes[i_layer],
                /* downsample */ i_layer < num_layers - 1,
                /* out */ embed_dims[usize::min(i_layer + 1, num_layers - 1)],
                vb_l.pp(i_layer),
            )?;
            layers.push(layer)
        }

        let last_embed_dim = embed_dims[embed_dims.len() - 1];
        let norm_head = candle_nn::layer_norm(last_embed_dim, 1e-5, vb.pp("norm_head"))?;
        let head = candle_nn::linear(last_embed_dim, num_classes, vb.pp("head"))?;
        let neck_conv1 =
            candle_nn::conv2d_no_bias(last_embed_dim, 256, 1, Default::default(), vb.pp("neck.0"))?;
        let neck_ln1 = crate::LayerNorm2d::new(256, 1e-6, vb.pp("neck.1"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck_conv2 = candle_nn::conv2d_no_bias(256, 256, 3, cfg, vb.pp("neck.2"))?;
        let neck_ln2 = crate::LayerNorm2d::new(256, 1e-6, vb.pp("neck.3"))?;

        Ok(Self {
            patch_embed,
            layer0,
            layers,
            norm_head,
            head,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
        })
    }
}

impl Module for TinyViT {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.patch_embed.forward(xs)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        let (b, _, c) = xs.dims3()?;
        xs.reshape((b, 64, 64, c))?
            .permute((0, 3, 1, 2))?
            .apply(&self.neck_conv1)?
            .apply(&self.neck_ln1)?
            .apply(&self.neck_conv2)?
            .apply(&self.neck_ln2)
    }
}
