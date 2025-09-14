//! Implementation of the Descript Audio Codec (DAC) model
//!
//! See: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
//!
/// An efficient neural codec for compressing/decompressing audio
///
use crate::models::encodec;
use candle::{BackendStorage, IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub num_codebooks: usize,
    pub model_bitrate: u32,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub frame_rate: u32,
    pub sampling_rate: u32,
}

#[derive(Debug, Clone)]
pub struct Snake1d<B: BackendStorage> {
    alpha: Tensor<B>,
}

impl<B: BackendStorage> Snake1d<B> {
    pub fn new(channels: usize, vb: VarBuilder<B>) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl<B: BackendStorage> candle::Module<B> for Snake1d<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit<B: BackendStorage> {
    snake1: Snake1d<B>,
    conv1: Conv1d<B>,
    snake2: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: BackendStorage> ResidualUnit<B> {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder<B>) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
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

impl<B: BackendStorage> candle::Module<B> for ResidualUnit<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
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
pub struct EncoderBlock<B: BackendStorage> {
    res1: ResidualUnit<B>,
    res2: ResidualUnit<B>,
    res3: ResidualUnit<B>,
    snake1: Snake1d<B>,
    conv1: Conv1d<B>,
}

impl<B: BackendStorage> EncoderBlock<B> {
    pub fn new(dim: usize, stride: usize, vb: VarBuilder<B>) -> Result<Self> {
        let vb = vb.pp("block");
        let res1 = ResidualUnit::new(dim / 2, 1, vb.pp(0))?;
        let res2 = ResidualUnit::new(dim / 2, 3, vb.pp(1))?;
        let res3 = ResidualUnit::new(dim / 2, 9, vb.pp(2))?;
        let snake1 = Snake1d::new(dim / 2, vb.pp(3))?;
        let cfg1 = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim / 2, dim, 2 * stride, cfg1, vb.pp(4))?;
        Ok(Self {
            res1,
            res2,
            res3,
            snake1,
            conv1,
        })
    }
}

impl<B: BackendStorage> candle::Module<B> for EncoderBlock<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder<B: BackendStorage> {
    conv1: Conv1d<B>,
    blocks: Vec<EncoderBlock<B>>,
    snake1: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: BackendStorage> candle::Module<B> for Encoder<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

impl<B: BackendStorage> Encoder<B> {
    pub fn new(
        mut d_model: usize,
        strides: &[usize],
        d_latent: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(1, d_model, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (block_idx, stride) in strides.iter().enumerate() {
            d_model *= 2;
            let block = EncoderBlock::new(d_model, *stride, vb.pp(block_idx + 1))?;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(d_model, vb.pp(strides.len() + 1))?;
        let cfg2 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv2 =
            encodec::conv1d_weight_norm(d_model, d_latent, 3, cfg2, vb.pp(strides.len() + 2))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlock<B: BackendStorage> {
    snake1: Snake1d<B>,
    conv_tr1: ConvTranspose1d<B>,
    res1: ResidualUnit<B>,
    res2: ResidualUnit<B>,
    res3: ResidualUnit<B>,
}

impl<B: BackendStorage> DecoderBlock<B> {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder<B>) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
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
        let res1 = ResidualUnit::new(out_dim, 1, vb.pp(2))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp(3))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp(4))?;
        Ok(Self {
            snake1,
            conv_tr1,
            res1,
            res2,
            res3,
        })
    }
}

impl<B: BackendStorage> candle::Module<B> for DecoderBlock<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
    }
}

#[derive(Debug, Clone)]
pub struct Decoder<B: BackendStorage> {
    conv1: Conv1d<B>,
    blocks: Vec<DecoderBlock<B>>,
    snake1: Snake1d<B>,
    conv2: Conv1d<B>,
}

impl<B: BackendStorage> Decoder<B> {
    pub fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(0))?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (idx, stride) in rates.iter().enumerate() {
            let block = DecoderBlock::new(channels, channels / 2, *stride, vb.pp(idx + 1))?;
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

impl<B: BackendStorage> candle::Module<B> for Decoder<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

#[allow(unused)]
#[derive(Clone, Debug)]
pub struct VectorQuantizer<B: BackendStorage> {
    in_proj: Conv1d<B>,
    out_proj: Conv1d<B>,
    codebook: candle_nn::Embedding<B>,
}

impl<B: BackendStorage> VectorQuantizer<B> {
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder<B>) -> Result<Self> {
        let in_proj =
            encodec::conv1d_weight_norm(in_dim, cb_dim, 1, Default::default(), vb.pp("in_proj"))?;
        let out_proj =
            encodec::conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
        let codebook = candle_nn::embedding(cb_size, cb_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
        })
    }

    pub fn embed_code(&self, embed_id: &Tensor<B>) -> Result<Tensor<B>> {
        embed_id.apply(&self.codebook)
    }

    pub fn decode_code(&self, embed_id: &Tensor<B>) -> Result<Tensor<B>> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer<B: BackendStorage> {
    quantizers: Vec<VectorQuantizer<B>>,
}

impl<B: BackendStorage> ResidualVectorQuantizer<B> {
    pub fn new(
        input_dim: usize,
        n_codebooks: usize,
        cb_size: usize,
        cb_dim: usize,
        vb: VarBuilder<B>,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = (0..n_codebooks)
            .map(|i| VectorQuantizer::new(input_dim, cb_size, cb_dim, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &Tensor<B>) -> Result<Tensor<B>> {
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

#[derive(Debug, Clone)]
pub struct Model<B: BackendStorage> {
    pub encoder: Encoder<B>,
    pub quantizer: ResidualVectorQuantizer<B>,
    pub decoder: Decoder<B>,
}

impl<B: BackendStorage> Model<B> {
    pub fn new(cfg: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let encoder = Encoder::new(64, &[2, 4, 8, 8], cfg.latent_dim, vb.pp("encoder"))?;
        let quantizer = ResidualVectorQuantizer::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            8,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(cfg.latent_dim, 1536, &[8, 8, 4, 2], 1, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            quantizer,
        })
    }

    pub fn decode_codes(&self, audio_codes: &Tensor<B>) -> Result<Tensor<B>> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }
}
