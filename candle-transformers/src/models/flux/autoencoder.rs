use candle::{Result, Tensor, D};
use candle_nn::{conv2d, group_norm, Conv2d, GroupNorm, VarBuilder};

// https://github.com/black-forest-labs/flux/blob/727e3a71faf37390f318cf9434f0939653302b60/src/flux/modules/autoencoder.py#L9
#[derive(Debug, Clone)]
pub struct Config {
    pub resolution: usize,
    pub in_channels: usize,
    pub ch: usize,
    pub out_ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub z_channels: usize,
    pub scale_factor: f64,
    pub shift_factor: f64,
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

#[derive(Debug, Clone)]
struct AttnBlock {
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
    norm: GroupNorm,
}

impl AttnBlock {
    fn new(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let q = conv2d(in_c, in_c, 1, Default::default(), vb.pp("q"))?;
        let k = conv2d(in_c, in_c, 1, Default::default(), vb.pp("k"))?;
        let v = conv2d(in_c, in_c, 1, Default::default(), vb.pp("v"))?;
        let proj_out = conv2d(in_c, in_c, 1, Default::default(), vb.pp("proj_out"))?;
        let norm = group_norm(32, in_c, 1e-6, vb.pp("norm"))?;
        Ok(Self {
            q,
            k,
            v,
            proj_out,
            norm,
        })
    }
}

impl candle::Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let init_xs = xs;
        let xs = xs.apply(&self.norm)?;
        let q = xs.apply(&self.q)?;
        let k = xs.apply(&self.k)?;
        let v = xs.apply(&self.v)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let xs = scaled_dot_product_attention(&q, &k, &v)?;
        let xs = xs.squeeze(1)?.t()?.reshape((b, c, h, w))?;
        xs.apply(&self.proj_out)? + init_xs
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(32, in_c, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(32, out_c, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let nin_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("nin_shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }
}

impl candle::Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.norm1)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv1)?
            .apply(&self.norm2)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv2)?;
        match self.nin_shortcut.as_ref() {
            None => xs + h,
            Some(c) => xs.apply(c)? + h,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let block_in = cfg.ch;
        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block_1"))?;
        let mid_attn_1 = AttnBlock::new(block_in, vb.pp("mid_attn_1"))?;
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block_2"))?;
        let conv_in = conv2d(cfg.in_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;
        let conv_out = conv2d(cfg.in_channels, block_in, 3, conv_cfg, vb.pp("conv_out"))?;
        let norm_out = group_norm(32, block_in, 1e-6, vb.pp("norm_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
        })
    }
}

impl candle_nn::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.conv_in)?;
        // TODO: blocks for downsampling
        h.apply(&self.mid_block_1)?
            .apply(&self.mid_attn_1)?
            .apply(&self.mid_block_2)?
            .apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let block_in = cfg.ch * cfg.ch_mult.last().unwrap_or(&1);
        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block_1"))?;
        let mid_attn_1 = AttnBlock::new(block_in, vb.pp("mid_attn_1"))?;
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid_block_2"))?;
        let conv_in = conv2d(cfg.z_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;
        let norm_out = group_norm(32, block_in, 1e-6, vb.pp("norm_out"))?;
        let conv_out = conv2d(block_in, cfg.out_ch, 3, conv_cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
        })
    }
}

impl candle_nn::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.conv_in)?;
        let h = h
            .apply(&self.mid_block_1)?
            .apply(&self.mid_attn_1)?
            .apply(&self.mid_block_2)?;
        // TODO: blocks for upsampling.
        h.apply(&self.norm_out)?
            .apply(&candle_nn::Activation::Swish)?
            .apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
    chunk_dim: usize,
}

impl DiagonalGaussian {
    pub fn new(sample: bool, chunk_dim: usize) -> Result<Self> {
        Ok(Self { sample, chunk_dim })
    }
}

impl candle_nn::Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, self.chunk_dim)?;
        if self.sample {
            let std = (&chunks[1] * 0.5)?.exp()?;
            &chunks[0] + (std * chunks[0].randn_like(0., 1.))?
        } else {
            Ok(chunks[0].clone())
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    shift_factor: f64,
    scale_factor: f64,
}

impl AutoEncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true, 1)?;
        Ok(Self {
            encoder,
            decoder,
            reg,
            scale_factor: cfg.scale_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let z = xs.apply(&self.encoder)?.apply(&self.reg)?;
        (z - self.shift_factor)? * self.scale_factor
    }
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        xs.apply(&self.decoder)
    }
}

impl candle::Module for AutoEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}
