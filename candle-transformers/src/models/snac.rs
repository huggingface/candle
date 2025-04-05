#![allow(unused)]
//! Implementation of the Multi-Scale Neural Audio Codec (SNAC)
//!
//! See: [SNAC](https://github.com/hubertsiuzdak/snac)
//!
/// Multi-Scale Neural Audio Codec (SNAC) compresses audio into discrete codes at a low bitrate.
/// For more information, read the paper: https://arxiv.org/abs/2410.14411
///
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    linear_b, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear,
    VarBuilder,
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

pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = {
        let name = "parametrizations.weight.original1";
        match vb.get((out_c, in_c, kernel_size), name) {
            Ok(v) => v,
            Err(_) => vb.get((out_c, 1, kernel_size), name)?,
        }
    };
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

pub fn conv1d_weight_norm_no_bias(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = {
        let name = "parametrizations.weight.original1";
        match vb.get((out_c, in_c, kernel_size), name) {
            Ok(v) => v,
            Err(_) => vb.get((out_c, 1, kernel_size), name)?,
        }
    };
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    Ok(Conv1d::new(weight, None, config))
}

pub fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: candle_nn::ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight_g = vb.get((in_c, 1, 1), "parametrizations.weight.original0")?;
    let weight_v = vb.get(
        (in_c, out_c, kernel_size),
        "parametrizations.weight.original1",
    )?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

// https://github.com/hubertsiuzdak/snac/blob/main/snac/attention.py
#[allow(unused)]
#[derive(Debug, Clone)]
struct SinusoidalEmbeddings {
    inv_freq: Tensor,
    scale: Tensor,
    scale_base: f32,
    use_xpos: bool,
}

impl SinusoidalEmbeddings {
    fn new(dim: usize, scale_base: f32, use_xpos: bool, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10_000f32.powf(i as f32 / dim as f32))
            .collect();
        let len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, len, dev)?.to_dtype(DType::F32)?;
        let scale: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| (i as f32 + 0.4 * dim as f32) / (1.4 * dim as f32))
            .collect();
        let scale = Tensor::from_vec(scale, len, dev)?.to_dtype(DType::F32)?;
        Ok(Self {
            inv_freq,
            scale,
            scale_base,
            use_xpos,
        })
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
struct LocalMHA {
    norm: LayerNorm,
    to_qkv: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
    rel_pos: Option<SinusoidalEmbeddings>,
}

impl LocalMHA {
    fn new(
        dim: usize,
        window_size: usize,
        dim_head: usize,
        use_rotary_pos_emb: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        let to_qkv = linear_b(dim, dim * 3, false, vb.pp("to_qkv"))?;
        let to_out = linear_b(dim, dim, false, vb.pp("to_out"))?;
        let rel_pos = if use_rotary_pos_emb {
            let rel_pos =
                SinusoidalEmbeddings::new(dim_head, window_size as f32 / 2.0, false, vb.device())?;
            Some(rel_pos)
        } else {
            None
        };
        Ok(Self {
            norm,
            to_qkv,
            to_out,
            rel_pos,
            num_heads: dim / dim_head,
            head_dim: dim_head,
        })
    }
}

impl Module for LocalMHA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        let residual = xs.clone();
        let xs = xs.transpose(1, 2)?.apply(&self.norm)?;
        let qkv = xs.apply(&self.to_qkv)?;
        let q = qkv.narrow(D::Minus1, 0, c)?;
        let k = qkv.narrow(D::Minus1, c, c)?;
        let v = qkv.narrow(D::Minus1, 2 * c, c)?;
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (q, k) = match self.rel_pos {
            Some(_) => todo!(),
            None => (q, k),
        };
        let out = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            // Non-causal attention
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?
            .apply(&self.to_out)?;
        out.transpose(1, 2)? + residual
    }
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
        let conv1 = conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp(1))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp(3))?;
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
        let linear = conv1d_weight_norm_no_bias(dim, dim, 1, Default::default(), vb.pp("linear"))?;
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
        let conv_tr1 =
            conv_transpose1d_weight_norm(in_dim, out_dim, 2 * stride, true, cfg, vb.pp(1))?;
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
        let conv1 = conv1d_weight_norm(in_dim, out_dim, 2 * stride, cfg1, vb.pp(4))?;
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
    local_mha: Option<LocalMHA>,
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
        let mut idx = 0;
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let groups = if depthwise { d_model / 2 } else { 1 };
        let conv1 = conv1d_weight_norm(1, d_model, 7, cfg1, vb.pp(idx))?;
        idx += 1;
        let mut blocks = Vec::with_capacity(strides.len());
        for &stride in strides.iter() {
            d_model *= 2;
            let block = EncoderBlock::new(d_model, None, stride, groups, vb.pp(idx))?;
            idx += 1;
            blocks.push(block)
        }
        let local_mha = match attn_window_size {
            Some(w) => {
                let mha = LocalMHA::new(d_model, w, 64, true, vb.pp(idx))?;
                idx += 1;
                Some(mha)
            }
            None => None,
        };
        let cfg2 = Conv1dConfig {
            padding: 3,
            groups,
            ..Default::default()
        };
        let conv2 = conv1d_weight_norm(d_model, d_model, 7, cfg2, vb.pp(idx))?;
        idx += 1;
        Ok(Self {
            conv1,
            blocks,
            local_mha,
            conv2,
        })
    }
}

#[derive(Debug, Clone)]
enum ConvInit {
    Depthwise(Conv1d, Conv1d),
    Standard(Conv1d),
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv1: ConvInit,
    local_mha: Option<LocalMHA>,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        noise: bool,
        depthwise: bool,
        attn_window_size: Option<usize>,
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("model");
        let mut idx = 0;
        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 = if depthwise {
            let conv1 = conv1d_weight_norm(in_c, in_c, 7, cfg1, vb.pp(idx))?;
            idx += 1;
            let conv2 = conv1d_weight_norm(in_c, channels, 1, Default::default(), vb.pp(idx))?;
            idx += 1;
            ConvInit::Depthwise(conv1, conv2)
        } else {
            let conv1 = conv1d_weight_norm(in_c, channels, 7, cfg1, vb.pp(idx))?;
            idx += 1;
            ConvInit::Standard(conv1)
        };
        let mut blocks = Vec::with_capacity(rates.len());
        let local_mha = match attn_window_size {
            Some(w) => {
                let mha = LocalMHA::new(channels, w, 64, true, vb.pp(idx))?;
                idx += 1;
                Some(mha)
            }
            None => None,
        };
        for stride in rates.iter() {
            let groups = if depthwise { channels / 2 } else { 1 };
            let block =
                DecoderBlock::new(channels, channels / 2, *stride, noise, groups, vb.pp(idx))?;
            idx += 1;
            channels /= 2;
            blocks.push(block)
        }
        let snake1 = Snake1d::new(channels, vb.pp(idx))?;
        idx += 1;
        let conv2 = conv1d_weight_norm(channels, d_out, 7, cfg1, vb.pp(idx))?;
        idx += 1;
        Ok(Self {
            conv1,
            local_mha,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl candle::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = match &self.conv1 {
            ConvInit::Standard(c) => xs.apply(c)?,
            ConvInit::Depthwise(c1, c2) => xs.apply(c1)?.apply(c2)?,
        };
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
        let in_proj = conv1d_weight_norm(in_dim, cb_dim, 1, Default::default(), vb.pp("in_proj"))?;
        let out_proj =
            conv1d_weight_norm(cb_dim, in_dim, 1, Default::default(), vb.pp("out_proj"))?;
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
