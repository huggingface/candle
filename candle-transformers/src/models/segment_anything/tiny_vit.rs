// Adapted from:
// https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/modeling/tiny_vit_sam.py
use candle::{IndexOp, Result, Tensor, D};
use candle_nn::{Conv2dConfig, Module, VarBuilder};

const MBCONV_EXPAND_RATIO: usize = 4;
const MLP_RATIO: usize = 4;
const LOCAL_CONV_SIZE: usize = 3;
const IMG_SIZE: usize = 1024;
const IN_CHANNELS: usize = 3;

#[derive(Debug)]
struct Conv2dBN {
    c: candle_nn::Conv2d,
    bn: candle_nn::BatchNorm,
    span: tracing::Span,
}

impl Conv2dBN {
    fn new(in_: usize, out: usize, ks: usize, cfg: Conv2dConfig, vb: VarBuilder) -> Result<Self> {
        let c = candle_nn::conv2d_no_bias(in_, out, ks, cfg, vb.pp("c"))?;
        let bn = candle_nn::batch_norm(out, 1e-5, vb.pp("bn"))?;
        let span = tracing::span!(tracing::Level::TRACE, "conv2d-bn");
        Ok(Self { c, bn, span })
    }
}

impl Module for Conv2dBN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.c)?.apply_t(&self.bn, false)
    }
}

#[derive(Debug)]
struct PatchEmbed {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    span: tracing::Span,
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
        let span = tracing::span!(tracing::Level::TRACE, "patch-embed");
        Ok(Self { conv1, conv2, span })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.conv1)?.gelu()?.apply(&self.conv2)
    }
}

#[derive(Debug)]
struct MBConv {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    conv3: Conv2dBN,
    span: tracing::Span,
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
        let span = tracing::span!(tracing::Level::TRACE, "mb-conv");
        Ok(Self {
            conv1,
            conv2,
            conv3,
            span,
        })
    }
}

impl Module for MBConv {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    span: tracing::Span,
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
        let span = tracing::span!(tracing::Level::TRACE, "patch-merging");
        Ok(Self {
            conv1,
            conv2,
            conv3,
            input_resolution,
            span,
        })
    }
}

impl Module for PatchMerging {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    span: tracing::Span,
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
        let span = tracing::span!(tracing::Level::TRACE, "conv-layer");
        Ok(Self {
            blocks,
            downsample,
            span,
        })
    }
}

impl Module for ConvLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    fc1: super::Linear,
    fc2: super::Linear,
    span: tracing::Span,
}

impl Mlp {
    fn new(in_: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::layer_norm(in_, 1e-5, vb.pp("norm"))?;
        let fc1 = super::linear(vb.pp("fc1"), in_, hidden, true)?;
        let fc2 = super::linear(vb.pp("fc2"), hidden, in_, true)?;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            norm,
            fc1,
            fc2,
            span,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.norm)?
            .apply(&self.fc1)?
            .gelu()?
            .apply(&self.fc2)
    }
}

#[derive(Debug)]
struct Attention {
    norm: candle_nn::LayerNorm,
    qkv: super::Linear,
    proj: super::Linear,
    ab: Tensor,
    key_dim: usize,
    num_heads: usize,
    d: usize,
    dh: usize,
    scale: f64,
    span: tracing::Span,
    span_matmul: tracing::Span,
    span_softmax: tracing::Span,
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
        let qkv = super::linear(vb.pp("qkv"), dim, h, true)?;
        let proj = super::linear(vb.pp("proj"), dh, dim, true)?;

        let points = (0..resolution.0)
            .flat_map(|x| (0..resolution.1).map(move |y| (x as i64, y as i64)))
            .collect::<Vec<_>>();
        let mut idxs = Vec::with_capacity(points.len() * points.len());
        let mut attention_offsets = std::collections::HashMap::new();
        for &(x1, y1) in points.iter() {
            for &(x2, y2) in points.iter() {
                let offset = ((x2 - x1).abs(), (y2 - y1).abs());
                let l = attention_offsets.len();
                let idx = attention_offsets.entry(offset).or_insert(l);
                idxs.push(*idx as u32)
            }
        }
        let attention_biases = vb.get((num_heads, attention_offsets.len()), "attention_biases")?;
        let idxs = Tensor::new(idxs, attention_biases.device())?;
        let ab =
            attention_biases
                .index_select(&idxs, 1)?
                .reshape(((), points.len(), points.len()))?;
        let span = tracing::span!(tracing::Level::TRACE, "attention");
        let span_matmul = tracing::span!(tracing::Level::TRACE, "attn-matmul");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "attn-sm");
        Ok(Self {
            norm,
            qkv,
            proj,
            ab,
            key_dim,
            num_heads,
            d,
            dh,
            scale: 1f64 / (key_dim as f64).sqrt(),
            span,
            span_matmul,
            span_softmax,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, n, _) = xs.dims3()?;
        let xs = xs.apply(&self.norm)?;
        let qkv = xs.apply(&self.qkv)?.reshape((b, n, self.num_heads, ()))?;
        let q = qkv
            .narrow(D::Minus1, 0, self.key_dim)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = qkv
            .narrow(D::Minus1, 2 * self.key_dim, self.d)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let attn = {
            let _enter = self.span_matmul.enter();
            (q.matmul(&k.t()?)? * self.scale)?
        };
        let attn = attn.broadcast_add(&self.ab)?;
        let attn = {
            let _enter = self.span_softmax.enter();
            candle_nn::ops::softmax_last_dim(&attn)?
        };
        let attn = {
            let _enter = self.span_matmul.enter();
            attn.matmul(&v)?
        };
        attn.transpose(1, 2)?
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
    input_resolution: (usize, usize),
    span: tracing::Span,
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
            groups: dim,
            ..Default::default()
        };
        let local_conv = Conv2dBN::new(dim, dim, LOCAL_CONV_SIZE, cfg, vb.pp("local_conv"))?;
        let span = tracing::span!(tracing::Level::TRACE, "attention");
        Ok(Self {
            attn,
            local_conv,
            mlp,
            window_size,
            input_resolution,
            span,
        })
    }
}

impl Module for TinyViTBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (h, w) = self.input_resolution;
        let (b, l, c) = xs.dims3()?;
        let res_x = xs;
        let xs = if h == self.window_size && w == self.window_size {
            self.attn.forward(xs)?
        } else {
            let xs = xs.reshape((b, h, w, c))?;
            let pad_b = (self.window_size - h % self.window_size) % self.window_size;
            let pad_r = (self.window_size - w % self.window_size) % self.window_size;

            let xs = if pad_b > 0 {
                xs.pad_with_zeros(1, 0, pad_b)?
            } else {
                xs
            };
            let xs = if pad_r > 0 {
                xs.pad_with_zeros(2, 0, pad_r)?
            } else {
                xs
            };
            let (p_h, p_w) = (h + pad_b, w + pad_r);
            let n_h = p_h / self.window_size;
            let n_w = p_w / self.window_size;
            let xs = xs
                .reshape((b, n_h, self.window_size, n_w, self.window_size, c))?
                .transpose(2, 3)?
                .reshape((b * n_h * n_w, self.window_size * self.window_size, c))?;
            let xs = self.attn.forward(&xs)?;
            let xs = xs
                .reshape((b, n_h, n_w, self.window_size, self.window_size, c))?
                .transpose(2, 3)?
                .reshape((b, p_h, p_w, c))?;
            let xs = if pad_r > 0 {
                xs.i((.., .., ..w))?.contiguous()?
            } else {
                xs
            };
            let xs = if pad_b > 0 {
                xs.i((.., ..h, ..))?.contiguous()?
            } else {
                xs
            };
            xs.reshape((b, l, c))?
        };
        let xs = (xs + res_x)?;
        let xs = xs
            .transpose(1, 2)?
            .reshape((b, c, h, w))?
            .apply(&self.local_conv)?
            .reshape((b, c, l))?
            .transpose(1, 2)?;
        &xs + self.mlp.forward(&xs)?
    }
}

#[derive(Debug)]
struct BasicLayer {
    blocks: Vec<TinyViTBlock>,
    downsample: Option<PatchMerging>,
    span: tracing::Span,
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
        let span = tracing::span!(tracing::Level::TRACE, "basic-layer");
        Ok(Self {
            blocks,
            downsample,
            span,
        })
    }
}

impl Module for BasicLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
pub struct TinyViT {
    patch_embed: PatchEmbed,
    layer0: ConvLayer,
    layers: Vec<BasicLayer>,
    // norm_head: candle_nn::LayerNorm,
    // head: candle_nn::Linear,
    neck_conv1: candle_nn::Conv2d,
    neck_ln1: super::LayerNorm2d,
    neck_conv2: candle_nn::Conv2d,
    neck_ln2: super::LayerNorm2d,
    span: tracing::Span,
    span_neck: tracing::Span,
}

impl TinyViT {
    pub fn new(
        embed_dims: &[usize],
        depths: &[usize],
        num_heads: &[usize],
        window_sizes: &[usize],
        _num_classes: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(IN_CHANNELS, embed_dims[0], vb.pp("patch_embed"))?;
        let patches_resolution = IMG_SIZE / 4;

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
        // let norm_head = candle_nn::layer_norm(last_embed_dim, 1e-5, vb.pp("norm_head"))?;
        // let head = candle_nn::linear(last_embed_dim, num_classes, vb.pp("head"))?;
        let neck_conv1 =
            candle_nn::conv2d_no_bias(last_embed_dim, 256, 1, Default::default(), vb.pp("neck.0"))?;
        let neck_ln1 = super::LayerNorm2d::new(256, 1e-6, vb.pp("neck.1"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck_conv2 = candle_nn::conv2d_no_bias(256, 256, 3, cfg, vb.pp("neck.2"))?;
        let neck_ln2 = super::LayerNorm2d::new(256, 1e-6, vb.pp("neck.3"))?;

        let span = tracing::span!(tracing::Level::TRACE, "tiny-vit");
        let span_neck = tracing::span!(tracing::Level::TRACE, "neck");
        Ok(Self {
            patch_embed,
            layer0,
            layers,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
            span,
            span_neck,
        })
    }
}

impl Module for TinyViT {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.patch_embed.forward(xs)?;
        let mut xs = self.layer0.forward(&xs)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        let (b, _, c) = xs.dims3()?;
        let _enter = self.span_neck.enter();
        xs.reshape((b, 64, 64, c))?
            .permute((0, 3, 1, 2))?
            .apply(&self.neck_conv1)?
            .apply(&self.neck_ln1)?
            .apply(&self.neck_conv2)?
            .apply(&self.neck_ln2)
    }
}

pub fn tiny_vit_5m(vb: VarBuilder) -> Result<TinyViT> {
    TinyViT::new(
        /* embed_dims */ &[64, 128, 160, 320],
        /* depths */ &[2, 2, 6, 2],
        /* num_heads */ &[2, 4, 5, 10],
        /* window_sizes */ &[7, 7, 14, 7],
        /* num_classes */ 1000,
        vb,
    )
}
