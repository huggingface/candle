//! Implementation of the Multi-Scale Neural Audio Codec (SNAC)
//!
//! See: [SNAC](https://github.com/hubertsiuzdak/snac)
//!
/// Multi-Scale Neural Audio Codec (SNAC) compresses audio into discrete codes at a low bitrate.
/// For more information, read the paper: https://arxiv.org/abs/2410.14411
///
use crate::models::encodec;
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, VarBuilder,
};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub sampling_rate: usize,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub attn_window_size: Option<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vq_strides: Vec<usize>,
    pub noise: bool,
    pub depthwise: bool,
}

#[allow(unused)]
#[derive(Debug, Clone)]
struct SinusoidalEmbeddings {
    inv_freq: Tensor,
}

#[allow(unused)]
#[derive(Debug, Clone)]
struct LocalMHA {
    norm: LayerNorm,
    to_qkv: Linear,
    rel_pos: Option<SinusoidalEmbeddings>,
}

#[derive(Debug, Clone)]
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn new(
        dim: usize,
        dilation: usize,
        kernel: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad = ((kernel - 1) * dilation) / 2;
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            groups,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = encodec::conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp(3))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        let pad = (xs.dim(D::Minus1)? - ys.dim(D::Minus1)?) / 2;
        if pad > 0 {
            &ys + xs.narrow(D::Minus1, pad, ys.dim(D::Minus1)?)
        } else {
            ys + xs
        }
    }
}

#[derive(Debug, Clone)]
struct NoiseBlock {
    linear: Conv1d,
}

impl NoiseBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // TODO: remove bias
        let linear = encodec::conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl Module for NoiseBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _c, t) = xs.dims3()?;
        let noise = Tensor::randn(0f32, 1f32, (b, 1, t), xs.device())?;
        let h = xs.apply(&self.linear)?;
        let n = noise.broadcast_mul(&h)?;
        let xs = (xs + n)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTranspose1d,
    noise: Option<NoiseBlock>,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    fn new(
        in_dim: usize,
        out_dim: usize,
        stride: usize,
        noise: bool,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            output_padding: stride % 2,
            ..Default::default()
        };
        let conv_tr1 = encodec::conv_transpose1d_weight_norm(
            in_dim,
            out_dim,
            2 * stride,
            true,
            cfg,
            vb.pp(1),
        )?;
        let (n, noise) = if noise {
            let noise = NoiseBlock::new(out_dim, vb.pp(2))?;
            (1, Some(noise))
        } else {
            (0, None)
        };
        let res1 = ResidualUnit::new(out_dim, 1, 7, groups, vb.pp(2 + n))?;
        let res2 = ResidualUnit::new(out_dim, 3, 7, groups, vb.pp(3 + n))?;
        let res3 = ResidualUnit::new(out_dim, 9, 7, groups, vb.pp(4 + n))?;
        Ok(Self {
            snake1,
            conv_tr1,
            noise,
            res1,
            res2,
            res3,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.noise.as_ref())?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl EncoderBlock {
    fn new(
        out_dim: usize,
        in_dim: Option<usize>,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let in_dim = in_dim.unwrap_or(out_dim / 2);
        let res1 = ResidualUnit::new(in_dim, 1, 7, groups, vb.pp(0))?;
        let res2 = ResidualUnit::new(in_dim, 3, 7, groups, vb.pp(1))?;
        let res3 = ResidualUnit::new(in_dim, 9, 7, groups, vb.pp(2))?;
        let snake1 = Snake1d::new(in_dim, vb.pp(3))?;
        let cfg1 = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(in_dim, out_dim, 2 * stride, cfg1, vb.pp(4))?;
        Ok(Self {
            res1,
            res2,
            res3,
            snake1,
            conv1,
        })
    }
}

impl candle::Module for EncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    conv2: Conv1d,
}

impl candle::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.conv2)
    }
}

impl Encoder {
    fn new(
        mut d_model: usize,
        strides: &[usize],
        depthwise: bool,
        attn_window_size: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let groups = if depthwise { d_model / 2 } else { 1 };
        let conv1 = encodec::conv1d_weight_norm(1, d_model, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (block_idx, &stride) in strides.iter().enumerate() {
            d_model *= 2;
            let block = EncoderBlock::new(d_model, None, stride, groups, vb.pp(block_idx + 1))?;
            blocks.push(block)
        }
        if let Some(_) = attn_window_size {
            todo!()
        }
        let cfg2 = Conv1dConfig {
            padding: 3,
            groups,
            ..Default::default()
        };
        let conv2 =
            encodec::conv1d_weight_norm(d_model, d_model, 7, cfg2, vb.pp(strides.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            conv2,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        noise: bool,
        depthwise: bool,
        attn_window: Option<usize>,
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(rates.len());
        if let Some(_) = attn_window {
            todo!()
        }
        for (idx, stride) in rates.iter().enumerate() {
            let groups = if depthwise { channels / 2 } else { 1 };
            let block = DecoderBlock::new(
                channels,
                channels / 2,
                *stride,
                noise,
                groups,
                vb.pp(idx + 1),
            )?;
            channels /= 2;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(channels, vb.pp(rates.len() + 1))?;
        let conv2 = encodec::conv1d_weight_norm(channels, d_out, 7, cfg1, vb.pp(rates.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl candle::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

// https://github.com/hubertsiuzdak/snac/blob/main/snac/vq.py
#[allow(unused)]
#[derive(Clone, Debug)]
struct VectorQuantizer {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: candle_nn::Embedding,
    stride: usize,
}

impl VectorQuantizer {
    fn new(
        in_dim: usize,
        cb_size: usize,
        cb_dim: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_proj =
            encodec::conv1d_weight_norm(in_dim, cb_dim, 1, Default::default(), vb.pp("in_proj"))?;
        let out_proj =
            encodec::conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
        let codebook = candle_nn::embedding(cb_size, cb_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
            stride,
        })
    }

    fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        embed_id.apply(&self.codebook)
    }

    fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    fn new(
        input_dim: usize,
        cb_size: usize,
        cb_dim: usize,
        vq_strides: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = vq_strides
            .iter()
            .enumerate()
            .map(|(i, stride)| VectorQuantizer::new(input_dim, cb_size, cb_dim, *stride, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let z_p_i = quantizer.decode_code(&codes.i((.., idx))?)?;
            let z_q_i = z_p_i.apply(&quantizer.out_proj)?;
            let s = match sum {
                None => z_q_i,
                Some(s) => (s + z_q_i)?,
            };
            sum = Some(s)
        }
        match sum {
            Some(s) => Ok(s),
            None => candle::bail!("empty codebooks"),
        }
    }
}

// https://github.com/hubertsiuzdak/snac/blob/main/snac/snac.py
#[derive(Debug, Clone)]
pub struct Model {
    pub encoder: Encoder,
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(
            cfg.encoder_dim,
            &cfg.encoder_rates,
            cfg.depthwise,
            cfg.attn_window_size,
            vb.pp("encoder"),
        )?;
        let latent_dim = cfg.encoder_dim * 2usize.pow(cfg.encoder_rates.len() as u32);
        let quantizer = ResidualVectorQuantizer::new(
            latent_dim,
            cfg.codebook_size,
            cfg.codebook_dim,
            &cfg.vq_strides,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(
            latent_dim,
            cfg.decoder_dim,
            &cfg.decoder_rates,
            cfg.noise,
            cfg.depthwise,
            cfg.attn_window_size,
            /* d_out */ 1,
            vb.pp("decoder"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }

    pub fn decode_codes(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }
}
