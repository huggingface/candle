//! AutoencoderOobleck VAE for the ACE-Step music generation model.
//!
//! This module implements the Oobleck audio VAE that encodes 48kHz stereo audio
//! into a compact latent representation and decodes it back. The architecture
//! uses Snake1d activations, weight-normalized convolutions, and a progressive
//! channel structure with strided up/downsampling.
//!
//! The encoder compresses (B, 2, T_audio) to (B, 128, T_latent) which is then
//! split into mean and log-variance (each 64 channels) for the diagonal Gaussian.
//! The decoder reconstructs (B, 64, T_latent) back to (B, 2, T_audio) where
//! T_audio is approximately T_latent * 1920 (the product of downsampling ratios).
//!
//! Reference: diffusers `AutoencoderOobleck`

use candle::{Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

use super::VaeConfig;
use crate::models::encodec;

// ---------------------------------------------------------------------------
// Snake1d activation
// ---------------------------------------------------------------------------

/// Learnable Snake activation with alpha and beta parameters stored in log-scale.
///
/// Computes: `x + (exp(beta) + 1e-9).recip() * sin(exp(alpha) * x)^2`
///
/// Both `alpha` and `beta` have shape `(1, channels, 1)` and are exponentiated
/// before use to ensure positivity.
#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
    beta: Tensor,
}

impl Snake1d {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        let beta = vb.get((1, channels, 1), "beta")?;
        Ok(Self { alpha, beta })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.exp()?;
        let beta = self.beta.exp()?;
        let sin = alpha.broadcast_mul(xs)?.sin()?;
        let sin_sq = (&sin * &sin)?;
        xs + ((&beta + 1e-9)?.recip()?.broadcast_mul(&sin_sq)?)
    }
}

// ---------------------------------------------------------------------------
// OobleckResidualUnit
// ---------------------------------------------------------------------------

/// Residual unit with dilated convolutions and Snake1d activations.
///
/// Structure: `snake1 -> conv1(k=7, dilation=d) -> snake2 -> conv2(k=1) + skip`
///
/// All convolutions use weight normalization. The skip connection trims the
/// input if the output is shorter due to dilation-induced size differences.
#[derive(Debug, Clone)]
pub struct OobleckResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl OobleckResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = ((7 - 1) * dilation) / 2;
        let snake1 = Snake1d::new(dim, vb.pp("snake1"))?;
        let cfg1 = Conv1dConfig {
            dilation,
            padding: pad,
            ..Default::default()
        };
        let conv1 = encodec::conv1d_weight_norm(dim, dim, 7, cfg1, vb.pp("conv1"))?;
        let snake2 = Snake1d::new(dim, vb.pp("snake2"))?;
        let conv2 = encodec::conv1d_weight_norm(dim, dim, 1, Default::default(), vb.pp("conv2"))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }
}

impl Module for OobleckResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs
            .apply(&self.snake1)?
            .apply(&self.conv1)?
            .apply(&self.snake2)?
            .apply(&self.conv2)?;
        let xs_len = xs.dim(D::Minus1)?;
        let ys_len = ys.dim(D::Minus1)?;
        if xs_len >= ys_len {
            let pad = (xs_len - ys_len) / 2;
            if pad > 0 {
                &ys + xs.narrow(D::Minus1, pad, ys_len)?
            } else {
                ys + xs
            }
        } else {
            let pad = (ys_len - xs_len) / 2;
            &ys.narrow(D::Minus1, pad, xs_len)? + xs
        }
    }
}

// ---------------------------------------------------------------------------
// OobleckEncoderBlock
// ---------------------------------------------------------------------------

/// Encoder block: 3 residual units (dilations 1, 3, 9) followed by Snake1d
/// activation and a strided downsampling convolution.
#[derive(Debug, Clone)]
pub struct OobleckEncoderBlock {
    res_unit1: OobleckResidualUnit,
    res_unit2: OobleckResidualUnit,
    res_unit3: OobleckResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl OobleckEncoderBlock {
    pub fn new(input_dim: usize, output_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let res_unit1 = OobleckResidualUnit::new(input_dim, 1, vb.pp("res_unit1"))?;
        let res_unit2 = OobleckResidualUnit::new(input_dim, 3, vb.pp("res_unit2"))?;
        let res_unit3 = OobleckResidualUnit::new(input_dim, 9, vb.pp("res_unit3"))?;
        let snake1 = Snake1d::new(input_dim, vb.pp("snake1"))?;
        let cfg = Conv1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv1 =
            encodec::conv1d_weight_norm(input_dim, output_dim, 2 * stride, cfg, vb.pp("conv1"))?;
        Ok(Self {
            res_unit1,
            res_unit2,
            res_unit3,
            snake1,
            conv1,
        })
    }
}

impl Module for OobleckEncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.res_unit1)?
            .apply(&self.res_unit2)?
            .apply(&self.res_unit3)?
            .apply(&self.snake1)?
            .apply(&self.conv1)
    }
}

// ---------------------------------------------------------------------------
// OobleckDecoderBlock
// ---------------------------------------------------------------------------

/// Decoder block: Snake1d activation and transposed convolution for upsampling,
/// followed by 3 residual units (dilations 1, 3, 9).
#[derive(Debug, Clone)]
pub struct OobleckDecoderBlock {
    snake1: Snake1d,
    conv_t1: ConvTranspose1d,
    res_unit1: OobleckResidualUnit,
    res_unit2: OobleckResidualUnit,
    res_unit3: OobleckResidualUnit,
}

impl OobleckDecoderBlock {
    pub fn new(input_dim: usize, output_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let snake1 = Snake1d::new(input_dim, vb.pp("snake1"))?;
        let cfg = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            ..Default::default()
        };
        let conv_t1 = encodec::conv_transpose1d_weight_norm(
            input_dim,
            output_dim,
            2 * stride,
            true,
            cfg,
            vb.pp("conv_t1"),
        )?;
        let res_unit1 = OobleckResidualUnit::new(output_dim, 1, vb.pp("res_unit1"))?;
        let res_unit2 = OobleckResidualUnit::new(output_dim, 3, vb.pp("res_unit2"))?;
        let res_unit3 = OobleckResidualUnit::new(output_dim, 9, vb.pp("res_unit3"))?;
        Ok(Self {
            snake1,
            conv_t1,
            res_unit1,
            res_unit2,
            res_unit3,
        })
    }
}

impl Module for OobleckDecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.snake1)?
            .apply(&self.conv_t1)?
            .apply(&self.res_unit1)?
            .apply(&self.res_unit2)?
            .apply(&self.res_unit3)
    }
}

// ---------------------------------------------------------------------------
// OobleckEncoder
// ---------------------------------------------------------------------------

/// Oobleck encoder that compresses stereo audio to a latent representation.
///
/// Channel progression (default config):
/// `2 -> 128 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 128`
#[derive(Debug, Clone)]
pub struct OobleckEncoder {
    conv1: Conv1d,
    blocks: Vec<OobleckEncoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl OobleckEncoder {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let enc_hs = config.encoder_hidden_size;

        // Prepend 1 to channel_multiples: [1, 1, 2, 4, 8, 16]
        let mut multiples = vec![1usize];
        multiples.extend_from_slice(&config.channel_multiples);

        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let conv1 =
            encodec::conv1d_weight_norm(config.audio_channels, enc_hs, 7, cfg1, vb.pp("conv1"))?;

        let mut blocks = Vec::with_capacity(config.downsampling_ratios.len());
        for (i, &stride) in config.downsampling_ratios.iter().enumerate() {
            let in_channels = enc_hs * multiples[i];
            let out_channels = enc_hs * multiples[i + 1];
            let block =
                OobleckEncoderBlock::new(in_channels, out_channels, stride, vb.pp("block").pp(i))?;
            blocks.push(block);
        }

        let final_channels = enc_hs * multiples[multiples.len() - 1];
        let snake1 = Snake1d::new(final_channels, vb.pp("snake1"))?;

        let cfg2 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv2 = encodec::conv1d_weight_norm(final_channels, enc_hs, 3, cfg2, vb.pp("conv2"))?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }
}

impl Module for OobleckEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

// ---------------------------------------------------------------------------
// OobleckDecoder
// ---------------------------------------------------------------------------

/// Oobleck decoder that reconstructs stereo audio from a latent representation.
///
/// Channel progression (default config):
/// `64 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 2`
#[derive(Debug, Clone)]
pub struct OobleckDecoder {
    conv1: Conv1d,
    blocks: Vec<OobleckDecoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl OobleckDecoder {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let dec_ch = config.decoder_channels;

        // Prepend 1 to channel_multiples: [1, 1, 2, 4, 8, 16]
        let mut multiples = vec![1usize];
        multiples.extend_from_slice(&config.channel_multiples);
        let n = multiples.len(); // 6

        // Upsampling ratios are reversed downsampling: [10, 6, 4, 4, 2]
        let upsampling_ratios: Vec<usize> =
            config.downsampling_ratios.iter().copied().rev().collect();

        let cfg1 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        // conv1: decoder_input_channels -> dec_ch * mult[-1] (e.g. 64 -> 2048)
        let first_channels = dec_ch * multiples[n - 1];
        let conv1 = encodec::conv1d_weight_norm(
            config.decoder_input_channels,
            first_channels,
            7,
            cfg1,
            vb.pp("conv1"),
        )?;

        let mut blocks = Vec::with_capacity(upsampling_ratios.len());
        for (i, &stride) in upsampling_ratios.iter().enumerate() {
            // Channels go from mult[N-1-i] to mult[N-2-i]
            let in_channels = dec_ch * multiples[n - 1 - i];
            let out_channels = dec_ch * multiples[n - 2 - i];
            let block =
                OobleckDecoderBlock::new(in_channels, out_channels, stride, vb.pp("block").pp(i))?;
            blocks.push(block);
        }

        let snake1 = Snake1d::new(dec_ch, vb.pp("snake1"))?;

        let cfg2 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        // Final conv: dec_ch -> audio_channels, NO bias
        let conv2 = encodec::conv1d_weight_norm_no_bias(
            dec_ch,
            config.audio_channels,
            7,
            cfg2,
            vb.pp("conv2"),
        )?;

        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }

    /// Tiled decoding for long latents that may not fit in memory.
    ///
    /// Splits the latent along the time axis into overlapping chunks, decodes
    /// each chunk independently via `forward`, trims the overlap regions in
    /// audio space, and concatenates the cores. The decoder is fully
    /// convolutional with local padding, so cores away from chunk boundaries
    /// are numerically identical to a single full-length decode (up to the
    /// decoder's finite receptive field, absorbed by the overlap).
    ///
    /// For short latents (`T_latent <= chunk_latent_frames`) this falls back
    /// to a plain `forward` call.
    ///
    /// - `chunk_latent_frames`: latent frames per chunk (default 128 ≈ 5.1s of
    ///   audio at 48 kHz with a 1920× upsample ratio).
    /// - `overlap_latent_frames`: overlap on each side in latent frames
    ///   (default 16). Must satisfy `chunk > 2 * overlap`.
    ///
    /// Input: `(B, decoder_input_channels, T_latent)`.
    /// Output: `(B, audio_channels, T_latent * upsample_ratio)`.
    pub fn tiled_decode(
        &self,
        z: &Tensor,
        chunk_latent_frames: Option<usize>,
        overlap_latent_frames: Option<usize>,
    ) -> Result<Tensor> {
        let chunk_size = chunk_latent_frames.unwrap_or(128);
        let overlap = overlap_latent_frames.unwrap_or(16);
        let total_frames = z.dim(2)?;

        if total_frames <= chunk_size {
            return self.forward(z);
        }

        if chunk_size <= 2 * overlap {
            candle::bail!(
                "tiled_decode: chunk_latent_frames ({chunk_size}) must be > 2 * overlap_latent_frames ({overlap})"
            );
        }

        let stride = chunk_size - 2 * overlap;
        let num_steps = total_frames.div_ceil(stride);
        let mut upsample_factor: Option<usize> = None;
        let mut audio_chunks = Vec::with_capacity(num_steps);

        for i in 0..num_steps {
            let core_start = i * stride;
            let core_end = (core_start + stride).min(total_frames);
            if core_start >= core_end {
                break;
            }

            // Expand the window with overlap on both sides, clipped at the edges.
            let win_start = core_start.saturating_sub(overlap);
            let win_end = (core_end + overlap).min(total_frames);

            let latent_chunk = z.narrow(2, win_start, win_end - win_start)?;
            let audio_chunk = self.forward(&latent_chunk)?;

            // Determine the upsample factor from the first chunk. The decoder
            // is strided-conv-transpose only, so audio_len is an integer
            // multiple of latent_len.
            let up = match upsample_factor {
                Some(up) => up,
                None => {
                    let audio_len = audio_chunk.dim(2)?;
                    let latent_len = latent_chunk.dim(2)?;
                    if latent_len == 0 || audio_len % latent_len != 0 {
                        candle::bail!(
                            "tiled_decode: audio length {audio_len} is not a multiple of latent length {latent_len}"
                        );
                    }
                    let up = audio_len / latent_len;
                    upsample_factor = Some(up);
                    up
                }
            };

            // Trim the overlap in audio space to recover only the core.
            let trim_start = (core_start - win_start) * up;
            let trim_end = (win_end - core_end) * up;
            let audio_len = audio_chunk.dim(2)?;
            let core_len = audio_len - trim_start - trim_end;
            let audio_core = audio_chunk.narrow(2, trim_start, core_len)?;
            audio_chunks.push(audio_core);
        }

        Tensor::cat(&audio_chunks.iter().collect::<Vec<_>>(), 2)
    }
}

impl Module for OobleckDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv1)?;
        for block in &self.blocks {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.snake1)?.apply(&self.conv2)
    }
}

// ---------------------------------------------------------------------------
// AutoencoderOobleck
// ---------------------------------------------------------------------------

/// AutoencoderOobleck: full VAE combining encoder and decoder for audio.
///
/// Encodes stereo 48kHz audio `(B, 2, T)` to latent `(B, 64, T/1920)` and
/// decodes back. Uses a diagonal Gaussian latent distribution with
/// reparameterization sampling.
#[derive(Debug, Clone)]
pub struct AutoencoderOobleck {
    encoder: OobleckEncoder,
    decoder: OobleckDecoder,
    config: VaeConfig,
}

impl AutoencoderOobleck {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = OobleckEncoder::new(config, vb.pp("encoder"))?;
        let decoder = OobleckDecoder::new(config, vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            config: config.clone(),
        })
    }

    /// Encode audio to a latent sample using the diagonal Gaussian.
    ///
    /// The encoder output `(B, encoder_hidden_size, T_latent)` is split along
    /// the channel dimension into mean and log-variance (each with
    /// `decoder_input_channels` channels), then sampled via reparameterization:
    /// `z = mean + exp(0.5 * logvar) * noise`.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(xs)?;
        let latent_dim = self.config.decoder_input_channels;
        let mean = h.narrow(1, 0, latent_dim)?;
        let logvar = h.narrow(1, latent_dim, latent_dim)?;
        let std = (logvar * 0.5)?.exp()?;
        let noise = std.randn_like(0., 1.)?;
        &mean + std.mul(&noise)?
    }

    /// Encode audio to the latent mean (no sampling noise).
    ///
    /// For deterministic encoding (e.g. when exact reconstruction matters),
    /// this returns only the mean of the diagonal Gaussian, without
    /// reparameterization noise.
    pub fn encode_mean(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(xs)?;
        let latent_dim = self.config.decoder_input_channels;
        h.narrow(1, 0, latent_dim)
    }

    /// Decode a latent representation back to audio waveform.
    ///
    /// Input shape: `(B, decoder_input_channels, T_latent)`
    /// Output shape: `(B, audio_channels, T_audio)`
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.decoder.forward(z)
    }

    /// Tiled decoding for long latents; see [`OobleckDecoder::tiled_decode`].
    pub fn tiled_decode(
        &self,
        z: &Tensor,
        chunk_latent_frames: Option<usize>,
        overlap_latent_frames: Option<usize>,
    ) -> Result<Tensor> {
        self.decoder
            .tiled_decode(z, chunk_latent_frames, overlap_latent_frames)
    }

    /// Tiled encoding for long audio that may not fit in GPU memory.
    ///
    /// Splits the audio into overlapping chunks, encodes each independently,
    /// trims the overlap regions, and concatenates. Uses the overlap-discard
    /// strategy to avoid boundary artifacts.
    ///
    /// - `chunk_samples`: audio samples per chunk (default: 48000 * 30 = 30s)
    /// - `overlap_samples`: overlap in audio samples (default: 48000 * 2 = 2s)
    ///
    /// Input: `(B, channels, T_audio)`, output: `(B, latent_dim, T_latent)`
    pub fn tiled_encode(
        &self,
        xs: &Tensor,
        chunk_samples: Option<usize>,
        overlap_samples: Option<usize>,
    ) -> Result<Tensor> {
        let chunk_size = chunk_samples.unwrap_or(48000 * 30);
        let overlap = overlap_samples.unwrap_or(48000 * 2);
        let total_samples = xs.dim(2)?;

        // Short audio: encode directly
        if total_samples <= chunk_size {
            return self.encode(xs);
        }

        if chunk_size <= 2 * overlap {
            candle::bail!(
                "tiled_encode: chunk_size ({chunk_size}) must be > 2 * overlap ({overlap})"
            );
        }
        let stride = chunk_size - 2 * overlap;
        let num_steps = total_samples.div_ceil(stride);
        let mut downsample_factor: Option<f64> = None;
        let mut latent_chunks = Vec::with_capacity(num_steps);

        for i in 0..num_steps {
            let core_start = i * stride;
            let core_end = (core_start + stride).min(total_samples);

            // Window with overlap
            let win_start = core_start.saturating_sub(overlap);
            let win_end = (core_end + overlap).min(total_samples);

            let audio_chunk = xs.narrow(2, win_start, win_end - win_start)?;
            let latent_chunk = self.encode(&audio_chunk)?;

            // Determine downsample factor from first chunk
            let df = match downsample_factor {
                Some(df) => df,
                None => {
                    let df = audio_chunk.dim(2)? as f64 / latent_chunk.dim(2)? as f64;
                    downsample_factor = Some(df);
                    df
                }
            };

            // Trim overlap in latent space
            let added_start = core_start - win_start;
            let trim_start = (added_start as f64 / df).round() as usize;

            let added_end = win_end - core_end;
            let trim_end = (added_end as f64 / df).round() as usize;

            let latent_len = latent_chunk.dim(2)?;
            let end_idx = if trim_end > 0 {
                latent_len - trim_end
            } else {
                latent_len
            };
            let core_len = end_idx - trim_start;
            let latent_core = latent_chunk.narrow(2, trim_start, core_len)?;
            latent_chunks.push(latent_core);
        }

        Tensor::cat(&latent_chunks.iter().collect::<Vec<_>>(), 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    /// Build a small OobleckDecoder with random weights on CPU. The
    /// architecture is identical to the production decoder; we just shrink the
    /// channel count so the test runs quickly.
    fn tiny_decoder() -> Result<(OobleckDecoder, Device)> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = VaeConfig {
            decoder_channels: 8,
            decoder_input_channels: 4,
            audio_channels: 2,
            // Keep the real upsampling structure to exercise all 5 blocks.
            downsampling_ratios: vec![2, 4, 4, 6, 10],
            channel_multiples: vec![1, 1, 1, 1, 1],
            ..VaeConfig::default()
        };
        let decoder = OobleckDecoder::new(&config, vb)?;
        Ok((decoder, device))
    }

    /// Elementwise compare two tensors tolerant to matching NaN/Inf values.
    /// The decoder has random weights with no normalization, so activations
    /// can explode past fp32 range — but `forward` and `tiled_decode` do the
    /// same computation on the same slice, so whatever NaN/Inf pattern
    /// `forward` produces, `tiled_decode` must produce in the same places.
    fn tensors_agree(a: &Tensor, b: &Tensor, atol: f32) -> Result<bool> {
        let a = a.flatten_all()?.to_vec1::<f32>()?;
        let b = b.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len(), "shape mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            let ok = match (x.is_nan(), y.is_nan()) {
                (true, true) => true,
                (false, false) => {
                    if x.is_infinite() || y.is_infinite() {
                        x == y
                    } else {
                        let scale = x.abs().max(y.abs()).max(1.0);
                        (x - y).abs() <= atol * scale
                    }
                }
                _ => false,
            };
            if !ok {
                eprintln!("mismatch: forward={x} tiled={y}");
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[test]
    fn tiled_decode_short_input_delegates_to_forward() -> Result<()> {
        let (decoder, device) = tiny_decoder()?;
        // total < chunk → tiled_decode must short-circuit to forward.
        let z = Tensor::randn(0f32, 1.0, (1, 4, 64), &device)?;

        let y_direct = decoder.forward(&z)?;
        let y_tiled = decoder.tiled_decode(&z, Some(128), Some(16))?;

        assert_eq!(y_tiled.dims(), y_direct.dims());
        assert!(tensors_agree(&y_direct, &y_tiled, 1e-5)?);
        Ok(())
    }

    #[test]
    fn tiled_decode_matches_forward_across_chunks() -> Result<()> {
        let (decoder, device) = tiny_decoder()?;
        // Force multiple chunk boundaries.
        let t_latent = 300;
        let z = Tensor::randn(0f32, 1.0, (1, 4, t_latent), &device)?;

        let y_direct = decoder.forward(&z)?;
        let y_tiled = decoder.tiled_decode(&z, Some(64), Some(16))?;

        assert_eq!(
            y_tiled.dims(),
            y_direct.dims(),
            "tiled output shape must equal full-forward shape"
        );
        // Receptive field fits inside the 16-frame overlap, so cores should
        // match within fp32 rounding slack. Values can be extreme due to
        // random untrained weights; compare relative.
        assert!(tensors_agree(&y_direct, &y_tiled, 1e-3)?);
        Ok(())
    }

    #[test]
    fn tiled_decode_rejects_bad_overlap() -> Result<()> {
        let (decoder, device) = tiny_decoder()?;
        let z = Tensor::randn(0f32, 1.0, (1, 4, 200), &device)?;

        // chunk <= 2 * overlap must fail.
        let err = decoder.tiled_decode(&z, Some(32), Some(16));
        assert!(err.is_err(), "expected tiled_decode to reject chunk<=2*overlap");
        Ok(())
    }
}
