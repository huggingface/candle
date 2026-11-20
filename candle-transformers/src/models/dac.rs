//! Implementation of the Descript Audio Codec (DAC) model
//!
//! See: [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
//!
/// An efficient neural codec for compressing/decompressing audio
///
use crate::models::encodec;
use candle::{IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Central code frames per cancellable decoder chunk (the halo is additional).
pub const DEFAULT_DECODE_CHUNK_SIZE: usize = 32;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(alias = "n_codebooks")]
    pub num_codebooks: usize,
    #[serde(default)]
    pub model_bitrate: u32,
    pub codebook_size: usize,
    #[serde(alias = "hidden_size")]
    pub latent_dim: usize,
    #[serde(default)]
    pub frame_rate: u32,
    pub sampling_rate: u32,
}

#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl candle::Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
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

impl candle::Module for ResidualUnit {
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
pub struct EncoderBlock {
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl EncoderBlock {
    pub fn new(dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
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
    snake1: Snake1d,
    conv2: Conv1d,
}

impl candle::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

impl Encoder {
    pub fn new(
        mut d_model: usize,
        strides: &[usize],
        d_latent: usize,
        vb: VarBuilder,
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
pub struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: ConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
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

impl candle_nn::Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_tr1)?
            .apply(&self.res1)?
            .apply(&self.res2)?
            .apply(&self.res3)
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
    pub fn new(
        in_c: usize,
        mut channels: usize,
        rates: &[usize],
        d_out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if rates
            .iter()
            .any(|rate| *rate == 0 || rate.checked_mul(2).is_none())
        {
            candle::bail!("invalid DAC decoder stride");
        }
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

    // Reverse the decoder's finite receptive field for one complete output
    // code-cell. Translation by `hop` samples corresponds to one input code,
    // so this covers every output phase without enumerating all hop phases.
    // Snake, tanh, codebook embeddings and 1x1 projections are pointwise.
    fn chunk_geometry(&self) -> Result<(usize, usize, i128)> {
        let mut hop = 1usize;
        let mut length_offset = 0i128;
        for block in &self.blocks {
            let config = block.conv_tr1.config();
            hop = hop
                .checked_mul(config.stride)
                .ok_or_else(|| candle::Error::Msg("DAC decoder hop overflow".into()))?;
            let kernel = block.conv_tr1.weight().dim(2)? as i128;
            length_offset = length_offset * config.stride as i128
                + config.dilation as i128 * (kernel - 1)
                + config.output_padding as i128
                + 1
                - config.stride as i128
                - 2 * config.padding as i128;
        }
        let mut interval = reverse_conv((0, hop as i128 - 1), &self.conv2)?;
        for block in self.blocks.iter().rev() {
            for residual in [&block.res3, &block.res2, &block.res1] {
                let main = reverse_conv(reverse_conv(interval, &residual.conv2)?, &residual.conv1)?;
                interval = (interval.0.min(main.0), interval.1.max(main.1));
            }
            let config = block.conv_tr1.config();
            let extent = config.dilation as i128 * (block.conv_tr1.weight().dim(2)? as i128 - 1);
            let stride = config.stride as i128;
            let left = interval.0 + config.padding as i128 - extent;
            interval = (
                -(-left).div_euclid(stride),
                (interval.1 + config.padding as i128).div_euclid(stride),
            );
        }
        interval = reverse_conv(interval, &self.conv1)?;
        let halo = (-interval.0.min(0)).max(interval.1.max(0));
        let halo = usize::try_from(halo).map_err(candle::Error::wrap)?;
        Ok((hop, halo, length_offset))
    }
}

fn reverse_conv(interval: (i128, i128), conv: &Conv1d) -> Result<(i128, i128)> {
    let config = conv.config();
    let extent = config.dilation as i128 * (conv.weight().dim(2)? as i128 - 1);
    // Every regular convolution in this decoder preserves length. Keeping that
    // invariant explicit prevents a future architecture edit silently changing
    // the chunk placement or residual skip alignment.
    if config.stride != 1 || 2 * config.padding as i128 != extent {
        candle::bail!("chunked DAC decoding requires length-preserving regular convolutions");
    }
    Ok((
        interval.0 - config.padding as i128,
        interval.1 - config.padding as i128 + extent,
    ))
}

impl candle::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)?.tanh()
    }
}

#[allow(unused)]
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: candle_nn::Embedding,
}

fn normalize_code_vectors(xs: &Tensor) -> Result<Tensor> {
    xs.broadcast_div(
        &xs.sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12f32, f32::MAX)?,
    )
}

impl VectorQuantizer {
    pub fn new(in_dim: usize, cb_size: usize, cb_dim: usize, vb: VarBuilder) -> Result<Self> {
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

    /// Normalized nearest-neighbor assignment as in Descript's reference VQ.
    pub fn encode(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let latents = xs.apply(&self.in_proj)?;
        let (batch, dim, time) = latents.dims3()?;
        let encodings = latents.transpose(1, 2)?.reshape((batch * time, dim))?;
        let encodings = normalize_code_vectors(&encodings)?;
        let codebook = normalize_code_vectors(self.codebook.embeddings())?;
        let distances = encodings
            .sqr()?
            .sum_keepdim(1)?
            .broadcast_add(&codebook.sqr()?.sum_keepdim(1)?.t()?)?
            .broadcast_sub(&(encodings.matmul(&codebook.t()?)? * 2.)?)?;
        let indices = distances.argmin(1)?.reshape((batch, time))?;
        let quantized = self.decode_code(&indices)?.apply(&self.out_proj)?;
        Ok((quantized, indices))
    }
    pub fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        embed_id.apply(&self.codebook)
    }

    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        self.embed_code(embed_id)?.transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        input_dim: usize,
        n_codebooks: usize,
        cb_size: usize,
        cb_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = &vb.pp("quantizers");
        let quantizers = (0..n_codebooks)
            .map(|i| VectorQuantizer::new(input_dim, cb_size, cb_dim, vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { quantizers })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut residual = xs.clone();
        let mut codes = Vec::with_capacity(self.quantizers.len());
        for quantizer in &self.quantizers {
            let (quantized, indices) = quantizer.encode(&residual)?;
            residual = (residual - quantized)?;
            codes.push(indices);
        }
        Tensor::stack(&codes, 1)
    }
    #[allow(clippy::wrong_self_convention)]
    pub fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
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
pub struct Model {
    pub encoder: Encoder,
    pub quantizer: ResidualVectorQuantizer,
    pub decoder: Decoder,
    hop_length: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::with_architecture(cfg, 64, &[2, 4, 8, 8], 1536, &[8, 8, 4, 2], 8, vb)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn with_architecture(
        cfg: &Config,
        encoder_dim: usize,
        encoder_rates: &[usize],
        decoder_dim: usize,
        decoder_rates: &[usize],
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if cfg.num_codebooks == 0
            || cfg.codebook_size < 2
            || cfg.latent_dim == 0
            || encoder_dim == 0
            || decoder_dim == 0
            || codebook_dim == 0
            || encoder_rates.is_empty()
            || decoder_rates.is_empty()
            || encoder_rates.contains(&0)
            || decoder_rates.contains(&0)
        {
            candle::bail!("invalid DAC configuration");
        }
        let hop_length = encoder_rates
            .iter()
            .try_fold(1usize, |n, value| n.checked_mul(*value))
            .ok_or_else(|| candle::Error::Msg("DAC hop overflow".into()))?;
        let output_hop = decoder_rates
            .iter()
            .try_fold(1usize, |n, value| n.checked_mul(*value))
            .ok_or_else(|| candle::Error::Msg("DAC output hop overflow".into()))?;
        if hop_length != output_hop {
            candle::bail!("DAC encoder/decoder hops differ");
        }
        let vb = if vb.contains_tensor("encoder.conv1.weight")
            || vb.contains_tensor("encoder.conv1.weight_g")
        {
            let (enc, dec) = (encoder_rates.len(), decoder_rates.len());
            vb.rename_f(move |name| transformers_name(name, enc, dec))
        } else {
            vb
        };
        let encoder = Encoder::new(encoder_dim, encoder_rates, cfg.latent_dim, vb.pp("encoder"))?;
        let quantizer = ResidualVectorQuantizer::new(
            cfg.latent_dim,
            cfg.num_codebooks,
            cfg.codebook_size,
            codebook_dim,
            vb.pp("quantizer"),
        )?;
        let decoder = Decoder::new(
            cfg.latent_dim,
            decoder_dim,
            decoder_rates,
            1,
            vb.pp("decoder"),
        )?;
        Ok(Self {
            encoder,
            quantizer,
            decoder,
            hop_length,
        })
    }
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let (_, channels, length) = audio.dims3()?;
        if channels != 1 || length == 0 {
            candle::bail!("DAC expects nonempty mono audio");
        }
        let padding = length.div_ceil(self.hop_length) * self.hop_length - length;
        let audio = audio.pad_with_zeros(D::Minus1, 0, padding)?;
        self.quantizer.encode(&audio.apply(&self.encoder)?)
    }

    pub fn decode_codes(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let audio_values = self.quantizer.from_codes(audio_codes)?;
        audio_values.apply(&self.decoder)
    }

    /// Receptive-field halo in input code frames for the loaded decoder.
    /// Parler's [8,8,4,2] decoder yields ten frames; other rates are derived.
    pub fn decoder_code_frame_halo(&self) -> Result<usize> {
        Ok(self.decoder.chunk_geometry()?.1)
    }

    pub fn decode_codes_chunked(&self, audio_codes: &Tensor) -> Result<Tensor> {
        self.decode_codes_chunked_with_callback(audio_codes, DEFAULT_DECODE_CHUNK_SIZE, |_| Ok(()))
    }

    /// Decode exact central chunks with a derived input halo on each side.
    /// True recording boundaries retain the full decoder's zero padding. Only
    /// central samples are concatenated, including a short final chunk. Floating
    /// point kernels may differ by a few ULP when their input shape changes.
    /// `check` receives the next central code offset, before each chunk and once
    /// after its decoder work (before accepting output); its error stops decoding.
    pub fn decode_codes_chunked_with_callback(
        &self,
        audio_codes: &Tensor,
        chunk_code_frames: usize,
        mut check: impl FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        let (batch, books, frames) = audio_codes.dims3()?;
        if batch == 0
            || frames == 0
            || books != self.quantizer.quantizers.len()
            || chunk_code_frames == 0
        {
            candle::bail!("chunked DAC decode needs nonempty batch/codes, matching codebooks and nonzero chunk size");
        }
        let (hop, halo, offset) = self.decoder.chunk_geometry()?;
        let length = frames as i128 * hop as i128 + offset;
        let length = usize::try_from(length).map_err(candle::Error::wrap)?;
        if length == 0 {
            candle::bail!("DAC code sequence is too short for its decoder");
        }
        let mut chunks = Vec::with_capacity(frames.div_ceil(chunk_code_frames));
        for start in (0..frames).step_by(chunk_code_frames) {
            let audio_start = start
                .checked_mul(hop)
                .ok_or_else(|| candle::Error::Msg("DAC chunk clock overflow".into()))?;
            if audio_start >= length {
                break;
            }
            check(start)?;
            let end = start.saturating_add(chunk_code_frames).min(frames);
            let extended_start = start.saturating_sub(halo);
            let extended_end = end.saturating_add(halo).min(frames);
            let codes = audio_codes
                .narrow(2, extended_start, extended_end - extended_start)?
                .contiguous()?;
            let decoded = self.decode_codes(&codes)?;
            // GPU backends may defer execution. Complete this bounded chunk
            // before checking cancellation instead of enqueueing the full utterance.
            audio_codes.device().synchronize()?;
            check(end)?;
            let audio_end = end
                .checked_mul(hop)
                .ok_or_else(|| candle::Error::Msg("DAC chunk clock overflow".into()))?
                .min(length);
            let local_start = (start - extended_start)
                .checked_mul(hop)
                .ok_or_else(|| candle::Error::Msg("DAC chunk clock overflow".into()))?;
            chunks.push(
                decoded
                    .narrow(2, local_start, audio_end - audio_start)?
                    .contiguous()?,
            );
        }
        Tensor::cat(&chunks, 2)
    }
}

// Legacy Descript/Parler module names -> Transformers DacModel module names.
fn transformers_name(name: &str, encoder_blocks: usize, decoder_blocks: usize) -> String {
    let map = |name: &str, prefix: &str, blocks: usize, is_encoder: bool| -> Option<String> {
        let rest = name.strip_prefix(prefix)?;
        let (index, suffix) = rest.split_once('.')?;
        let index: usize = index.parse().ok()?;
        let root = if index == 0 {
            format!("{}.conv1", if is_encoder { "encoder" } else { "decoder" })
        } else if index == blocks + 1 {
            format!("{}.snake1", if is_encoder { "encoder" } else { "decoder" })
        } else if index == blocks + 2 {
            format!("{}.conv2", if is_encoder { "encoder" } else { "decoder" })
        } else {
            format!(
                "{}.block.{}",
                if is_encoder { "encoder" } else { "decoder" },
                index - 1
            )
        };
        let mut suffix = suffix.to_owned();
        if (1..=blocks).contains(&index) {
            if let Some(rest) = suffix.strip_prefix("block.") {
                if let Some((sub, tail)) = rest.split_once('.') {
                    let component = match (is_encoder, sub) {
                        (true, "0") => "res_unit1",
                        (true, "1") => "res_unit2",
                        (true, "2") => "res_unit3",
                        (true, "3") => "snake1",
                        (true, "4") => "conv1",
                        (false, "0") => "snake1",
                        (false, "1") => "conv_t1",
                        (false, "2") => "res_unit1",
                        (false, "3") => "res_unit2",
                        (false, "4") => "res_unit3",
                        _ => return None,
                    };
                    suffix = format!("{component}.{tail}");
                }
            }
            for (i, component) in [(0, "snake1"), (1, "conv1"), (2, "snake2"), (3, "conv2")] {
                suffix = suffix.replace(&format!(".block.{i}."), &format!(".{component}."));
            }
        }
        Some(format!("{root}.{suffix}"))
    };
    map(name, "encoder.block.", encoder_blocks, true)
        .or_else(|| map(name, "decoder.model.", decoder_blocks, false))
        .unwrap_or_else(|| name.to_owned())
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    struct VariedWeights;
    impl candle_nn::var_builder::SimpleBackend for VariedWeights {
        fn get(
            &self,
            shape: candle::Shape,
            name: &str,
            _: candle_nn::Init,
            dtype: candle::DType,
            device: &candle::Device,
        ) -> Result<Tensor> {
            let hash = name.bytes().fold(0u32, |hash, byte| {
                hash.wrapping_mul(31).wrapping_add(byte as u32)
            });
            let values = (0..shape.elem_count())
                .map(|index| {
                    if name.ends_with("alpha") {
                        1.
                    } else if name.ends_with("bias") {
                        0.003 * ((index + hash as usize % 97) as f32 * 0.17).cos()
                    } else {
                        let scale = if name.contains("codebook") { 0.5 } else { 0.12 };
                        scale * ((index + hash as usize % 997) as f32 * 0.13).sin()
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
        }
        fn get_unchecked(&self, _: &str, _: candle::DType, _: &candle::Device) -> Result<Tensor> {
            candle::bail!("fixture requires explicit shapes")
        }
        fn contains_tensor(&self, name: &str) -> bool {
            name.ends_with("weight")
        }
    }
    fn small_chunk_model(
        encoder_rates: &[usize],
        decoder_rates: &[usize],
        decoder_dim: usize,
    ) -> Result<Model> {
        let cfg = Config {
            num_codebooks: 2,
            model_bitrate: 0,
            codebook_size: 8,
            latent_dim: 4,
            frame_rate: 0,
            sampling_rate: 32000,
        };
        let vb = VarBuilder::from_backend(
            Box::new(VariedWeights),
            candle::DType::F32,
            candle::Device::Cpu,
        );
        Model::with_architecture(&cfg, 2, encoder_rates, decoder_dim, decoder_rates, 2, vb)
    }
    fn compare_chunk_sizes(model: &Model, frames: usize, sizes: &[usize]) -> Result<()> {
        let codes = (0..2 * frames)
            .map(|index| ((index * 3 + index / frames * 5) % 8) as u32)
            .collect::<Vec<_>>();
        let codes = Tensor::from_vec(codes, (1, 2, frames), &candle::Device::Cpu)?;
        let expected = model.decode_codes(&codes)?;
        let expected_values = expected.flatten_all()?.to_vec1::<f32>()?;
        let range = expected_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - expected_values
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
        assert!(
            range > 1e-4,
            "nonconstant fixture lost its input dependence: range={range}"
        );
        for size in sizes {
            let actual = model.decode_codes_chunked_with_callback(&codes, *size, |_| Ok(()))?;
            assert_eq!(actual.dims(), expected.dims());
            let difference = actual
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .zip(&expected_values)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                difference < 2e-5,
                "chunk={size}, halo={}, max_abs={difference}",
                model.decoder_code_frame_halo()?
            );
        }
        let mut checks = 0;
        let result = model.decode_codes_chunked_with_callback(&codes, 7, |offset| {
            checks += 1;
            if offset >= 7 {
                candle::bail!("stop chunk fixture");
            }
            Ok(())
        });
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("stop chunk fixture"));
        assert_eq!(checks, 2);
        assert!(model
            .decode_codes_chunked_with_callback(&codes, 0, |_| Ok(()))
            .is_err());
        Ok(())
    }
    #[test]
    fn chunked_decoder_matches_whole_with_default_rates_and_true_boundaries() -> Result<()> {
        let model = small_chunk_model(&[2, 4, 8, 8], &[8, 8, 4, 2], 32)?;
        assert_eq!(model.decoder_code_frame_halo()?, 10);
        compare_chunk_sizes(&model, 41, &[1, 7, 32, 64])
    }
    #[test]
    fn chunked_decoder_derives_other_halos_and_odd_stride_lengths() -> Result<()> {
        let model = small_chunk_model(&[2], &[2], 4)?;
        assert_eq!(model.decoder_code_frame_halo()?, 25);
        compare_chunk_sizes(&model, 73, &[1, 7, 32, 64])?;
        let model = small_chunk_model(&[2, 3], &[3, 2], 8)?;
        assert_ne!(model.decoder_code_frame_halo()?, 10);
        compare_chunk_sizes(&model, 73, &[1, 7, 32, 64])
    }
    #[test]
    fn encoder_matches_independent_descript_torch_fixture() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dac_encoder.json"))
                .map_err(candle::Error::wrap)?;
        let values = |value: &serde_json::Value| {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let device = candle::Device::Cpu;
        let weights = fixture["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, tensor)| {
                let shape = tensor["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                Ok((
                    name.clone(),
                    Tensor::from_vec(values(&tensor["values"]), shape, &device)?,
                ))
            })
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let encoder = Encoder::new(
            2,
            &[2],
            4,
            VarBuilder::from_tensors(weights, candle::DType::F32, &device),
        )?;
        let input = values(&fixture["input"]);
        let length = input.len();
        let actual = Tensor::from_vec(input, (1, 1, length), &device)?
            .apply(&encoder)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = values(&fixture["expected"]);
        assert_eq!(actual.len(), expected.len());
        let difference = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(difference < 1e-6, "encoder difference={difference}");
        Ok(())
    }
    #[test]
    fn normalized_assignment_uses_direction_not_unnormalized_distance() -> Result<()> {
        let device = candle::Device::Cpu;
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/dac_encoder.json"))
                .map_err(candle::Error::wrap)?;
        let values = |name: &str| {
            fixture["quantizer"][name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let mut weights = std::collections::HashMap::new();
        let identity = Tensor::new(&[[[1f32], [0.]], [[0.], [1.]]], &device)?;
        for name in ["in_proj", "out_proj"] {
            weights.insert(format!("{name}.weight"), identity.clone());
            weights.insert(
                format!("{name}.bias"),
                Tensor::zeros(2, candle::DType::F32, &device)?,
            );
        }
        weights.insert(
            "codebook.weight".into(),
            Tensor::from_vec(values("codebook"), (4, 2), &device)?,
        );
        let quantizer = VectorQuantizer::new(
            2,
            4,
            2,
            VarBuilder::from_tensors(weights, candle::DType::F32, &device),
        )?;
        let latent = Tensor::from_vec(values("input"), (1, 2, 4), &device)?;
        let (quantized, indices) = quantizer.encode(&latent)?;
        let expected = fixture["quantizer"]["expected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(indices.flatten_all()?.to_vec1::<u32>()?, expected);
        assert_eq!(
            quantized.flatten_all()?.to_vec1::<f32>()?,
            values("expected_quantized")
        );
        Ok(())
    }
    #[test]
    fn decoder_applies_final_tanh() -> Result<()> {
        let device = candle::Device::Cpu;
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "model.0.weight".into(),
            Tensor::zeros((2, 1, 7), candle::DType::F32, &device)?,
        );
        weights.insert(
            "model.0.bias".into(),
            Tensor::zeros(2, candle::DType::F32, &device)?,
        );
        weights.insert(
            "model.1.alpha".into(),
            Tensor::ones((1, 2, 1), candle::DType::F32, &device)?,
        );
        weights.insert(
            "model.2.weight".into(),
            Tensor::zeros((1, 2, 7), candle::DType::F32, &device)?,
        );
        weights.insert("model.2.bias".into(), Tensor::new(&[2f32], &device)?);
        let decoder = Decoder::new(
            1,
            2,
            &[],
            1,
            VarBuilder::from_tensors(weights, candle::DType::F32, &device),
        )?;
        let output = Tensor::zeros((1, 1, 5), candle::DType::F32, &device)?
            .apply(&decoder)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(output
            .iter()
            .all(|value| (*value - 2f32.tanh()).abs() < 1e-6));
        Ok(())
    }
}
