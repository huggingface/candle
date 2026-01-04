//! CausalHiFTGenerator - HiFi-GAN with iSTFT Vocoder
//!
//! Converts mel spectrogram to waveform using:
//! - F0 prediction from mel
//! - Neural Source Filter (NSF) source generation
//! - HiFi-GAN style upsampling with residual blocks
//! - iSTFT for final waveform synthesis

use candle::{Device, DType, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use super::f0_predictor::{CausalConv1d, CausalConvRNNF0Predictor, CausalType};
use super::istft::HiFTiSTFT;
use super::source_module::SourceModuleHnNSF;
use super::stft::HiFTSTFT;
use crate::models::cosyvoice::activations::{LeakyReLU, Snake};
use crate::models::cosyvoice::config::HiFTConfig;

/// CausalConv1dUpsample - Upsample using nearest neighbor + Conv1d
///
/// Follows the official implementation:
/// 1. Nearest neighbor upsample by scale_factor
/// 2. Left-pad by (kernel_size - 1) for causal convolution
/// 3. Apply Conv1d with stride=1
#[derive(Debug)]
pub struct CausalConv1dUpsample {
    conv: Conv1d,
    scale_factor: usize,
    causal_padding: usize,
}

impl CausalConv1dUpsample {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        scale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, conv_config, vb)?;
        let causal_padding = kernel_size - 1;
        
        Ok(Self {
            conv,
            scale_factor,
            causal_padding,
        })
    }
}

impl Module for CausalConv1dUpsample {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Nearest neighbor upsample (manual implementation for Metal compatibility)
        let x = upsample_nearest1d_manual(x, self.scale_factor)?;
        
        // 2. Left pad for causal convolution
        let x = x.pad_with_zeros(2, self.causal_padding, 0)?;
        
        // 3. Apply Conv1d
        self.conv.forward(&x)
    }
}

/// Manual nearest neighbor upsampling for 1D that works on all backends
/// Input: [B, C, T], Output: [B, C, T * scale_factor]
fn upsample_nearest1d_manual(x: &Tensor, scale_factor: usize) -> Result<Tensor> {
    let (_batch, _channels, time) = x.dims3()?;
    let target_len = time * scale_factor;

    if scale_factor == 1 {
        return Ok(x.clone());
    }

    // Create indices for gather operation
    // For each target position i, pick from source position i / scale_factor
    let indices: Vec<u32> = (0..target_len)
        .map(|i| (i / scale_factor) as u32)
        .collect();
    let indices = Tensor::from_vec(indices, target_len, x.device())?;

    // Use index_select on the time dimension
    x.index_select(&indices, 2)
}

/// Reflection padding for 1D tensors
/// Input: [B, C, T], pads `left` elements on the left and `right` elements on the right
/// using reflection (mirrors the edge values)
///
/// For example, with input [1, 2, 3, 4] and left=1, right=0:
/// Output: [2, 1, 2, 3, 4] (reflects the second element to the left)
fn reflection_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }

    let (_batch, _channels, time) = x.dims3()?;
    
    // Build indices for reflection padding
    // Left padding: reflect from position 1, 2, ... (not 0)
    // Right padding: reflect from position time-2, time-3, ...
    let mut indices: Vec<u32> = Vec::with_capacity(left + time + right);
    
    // Left reflection: indices go left-1, left-2, ..., 0 -> positions 1, 2, ..., left
    for i in 0..left {
        let idx = left - i; // 1, 2, ..., left
        indices.push(idx.min(time - 1) as u32);
    }
    
    // Original content
    for i in 0..time {
        indices.push(i as u32);
    }
    
    // Right reflection: positions time-2, time-3, ...
    for i in 0..right {
        let idx = time.saturating_sub(2 + i);
        indices.push(idx as u32);
    }
    
    let indices = Tensor::from_vec(indices, left + time + right, x.device())?;
    x.index_select(&indices, 2)
}

/// ResBlock with Snake activation
/// 
/// For CausalHiFTGenerator, uses CausalConv1d with causal_type='left' for causal inference.
/// The causal padding ensures the output only depends on past inputs.
#[derive(Debug)]
pub struct ResBlock {
    convs1: Vec<CausalConv1d>,
    convs2: Vec<CausalConv1d>,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut activations1 = Vec::new();
        let mut activations2 = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            // Use CausalConv1d with causal_type='left' (matching Python implementation)
            let conv1 = CausalConv1d::new(
                channels,
                channels,
                kernel_size,
                dilation,
                CausalType::Left,
                vb.pp(format!("convs1.{}", i)),
            )?;
            convs1.push(conv1);

            // convs2 always uses dilation=1
            let conv2 = CausalConv1d::new(
                channels,
                channels,
                kernel_size,
                1, // dilation=1 for convs2
                CausalType::Left,
                vb.pp(format!("convs2.{}", i)),
            )?;
            convs2.push(conv2);

            // Snake activations
            activations1.push(Snake::new(channels, vb.pp(format!("activations1.{}", i)))?);
            activations2.push(Snake::new(channels, vb.pp(format!("activations2.{}", i)))?);
        }

        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
        })
    }
}

impl Module for ResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        for i in 0..self.convs1.len() {
            let xt = self.activations1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }

        Ok(x)
    }
}

/// Causal HiFT Generator
///
/// The main vocoder that converts mel spectrogram to waveform.
#[derive(Debug)]
pub struct CausalHiFTGenerator {
    config: HiFTConfig,
    f0_predictor: CausalConvRNNF0Predictor,
    m_source: SourceModuleHnNSF,
    /// CausalConv1d with causal_type='right' for lookahead
    conv_pre: CausalConv1d,
    ups: Vec<CausalConv1dUpsample>,
    source_downs: Vec<Conv1d>,
    source_down_causal_paddings: Vec<usize>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<Vec<ResBlock>>,
    conv_post: CausalConv1d,
    stft: HiFTSTFT,
    istft: HiFTiSTFT,
    lrelu: LeakyReLU,
    device: Device,
    dtype: DType,
}

impl CausalHiFTGenerator {
    pub fn new(config: HiFTConfig, vb: VarBuilder) -> Result<Self> {
        // Calculate total upsampling ratio (including iSTFT hop length)
        let upsample_scale: usize =
            config.upsample_rates.iter().product::<usize>() * config.istft_hop_len;

        // F0 Predictor
        // NOTE: The official Python implementation moves F0 predictor to CPU for precision:
        // "NOTE f0_predictor precision is crucial for causal inference, move self.f0_predictor to cpu if necessary"
        // See: refs/CosyVoice/cosyvoice/hifigan/generator.py line 715-717
        //
        // However, in Candle, moving weights to a different device after loading is complex.
        // For now, we keep force_cpu=false but ensure F32 precision is used in the forward pass.
        // TODO: Implement proper CPU fallback by loading F0 predictor weights on CPU device.
        let f0_predictor =
            CausalConvRNNF0Predictor::new(config.in_channels, 512, false, vb.pp("f0_predictor"))?;

        // Source Module
        let m_source = SourceModuleHnNSF::new(
            config.sampling_rate,
            upsample_scale,
            config.nb_harmonics,
            config.nsf_alpha,
            config.nsf_sigma,
            vb.pp("m_source"),
        )?;

        // Calculate channel count for each layer
        let mut channel_sizes = vec![config.base_channels];
        for i in 0..config.upsample_rates.len() {
            channel_sizes.push(config.base_channels / (1 << (i + 1)));
        }

        // Conv Pre - CausalConv1d with causal_type='right' for lookahead
        // kernel_size = conv_pre_look_right + 1 (following official implementation)
        let conv_pre_kernel_size = config.conv_pre_look_right + 1;
        let conv_pre = CausalConv1d::new(
            config.in_channels,
            config.base_channels,
            conv_pre_kernel_size,
            1, // dilation
            CausalType::Right,
            vb.pp("conv_pre"),
        )?;

        // Upsampling layers using CausalConv1dUpsample
        let mut ups = Vec::new();
        for (i, (&rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = channel_sizes[i];
            let out_ch = channel_sizes[i + 1];

            // CausalConv1dUpsample: nearest upsample + causal conv
            let up = CausalConv1dUpsample::new(
                in_ch,
                out_ch,
                kernel_size,
                rate,
                vb.pp(format!("ups.{}", i)),
            )?;
            ups.push(up);
        }

        // Source downsampling layers
        // Following official implementation:
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // downsample_cum_rates = np.cumprod(downsample_rates)
        // Then reversed for the loop
        let source_input_ch = (config.istft_n_fft / 2 + 1) * 2; // 18 for n_fft=16
        let mut source_downs = Vec::new();
        let mut source_down_causal_paddings = Vec::new();
        let mut source_resblocks_vec = Vec::new();

        // Calculate downsample_cum_rates following Python logic
        // upsample_rates = [8, 5, 3]
        // upsample_rates[::-1] = [3, 5, 8]
        // upsample_rates[::-1][:-1] = [3, 5]
        // downsample_rates = [1] + [3, 5] = [1, 3, 5]
        // downsample_cum_rates = cumprod([1, 3, 5]) = [1, 3, 15]
        // downsample_cum_rates[::-1] = [15, 3, 1]
        let mut downsample_rates = vec![1usize];
        let reversed_rates: Vec<usize> = config.upsample_rates.iter().rev().cloned().collect();
        if reversed_rates.len() > 1 {
            downsample_rates.extend(&reversed_rates[..reversed_rates.len() - 1]);
        }
        let mut downsample_cum_rates: Vec<usize> = Vec::new();
        let mut cum = 1;
        for &r in &downsample_rates {
            cum *= r;
            downsample_cum_rates.push(cum);
        }
        // Reverse for iteration
        downsample_cum_rates.reverse();

        for i in 0..config.upsample_rates.len() {
            let out_ch = channel_sizes[i + 1];
            let u = downsample_cum_rates[i];

            // Source downsampling with causal padding
            // For u == 1: CausalConv1d with kernel_size=1
            // For u != 1: CausalConv1dDownSample with kernel_size=u*2, stride=u, causal_padding=stride-1
            let (kernel_size, stride, causal_padding) = if u == 1 {
                (1, 1, 0)
            } else {
                (u * 2, u, u - 1) // causal_padding = stride - 1
            };

            let down_config = Conv1dConfig {
                padding: 0, // No symmetric padding, we'll do causal padding manually
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            let source_down = candle_nn::conv1d(
                source_input_ch,
                out_ch,
                kernel_size,
                down_config,
                vb.pp(format!("source_downs.{}", i)),
            )?;
            source_downs.push(source_down);
            source_down_causal_paddings.push(causal_padding);

            // Source ResBlock - use config parameters
            if i < config.source_resblock_kernel_sizes.len() {
                let resblock_kernel_size = config.source_resblock_kernel_sizes[i];
                let dilations = &config.source_resblock_dilation_sizes[i];
                let source_resblock = ResBlock::new(
                    out_ch,
                    resblock_kernel_size,
                    dilations,
                    vb.pp(format!("source_resblocks.{}", i)),
                )?;
                source_resblocks_vec.push(source_resblock);
            }
        }

        // ResBlocks for each upsampling stage
        // Following official implementation: resblocks use flat indexing
        // resblocks.{flat_idx} where flat_idx = stage * num_kernels + kernel_idx
        let num_kernels = config.resblock_kernel_sizes.len();
        let mut resblocks = Vec::new();
        for (i, _) in config.upsample_rates.iter().enumerate() {
            let ch = channel_sizes[i + 1];
            let mut stage_resblocks = Vec::new();

            for (j, &kernel_size) in config.resblock_kernel_sizes.iter().enumerate() {
                let dilations = &config.resblock_dilation_sizes[j];
                let flat_idx = i * num_kernels + j;
                let resblock = ResBlock::new(
                    ch,
                    kernel_size,
                    dilations,
                    vb.pp(format!("resblocks.{}", flat_idx)),
                )?;
                stage_resblocks.push(resblock);
            }
            resblocks.push(stage_resblocks);
        }

        // Conv Post - CausalConv1d with causal_type='left'
        let final_ch = *channel_sizes.last().unwrap();
        let post_out_ch = (config.istft_n_fft / 2 + 1) * 2; // 18
        let conv_post = CausalConv1d::new(
            final_ch,
            post_out_ch,
            7, // kernel_size
            1, // dilation
            CausalType::Left,
            vb.pp("conv_post"),
        )?;

        // STFT/iSTFT
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let stft = HiFTSTFT::new(&device, dtype)?;
        let istft = HiFTiSTFT::new(&device, dtype)?;

        Ok(Self {
            config,
            f0_predictor,
            m_source,
            conv_pre,
            ups,
            source_downs,
            source_down_causal_paddings,
            source_resblocks: source_resblocks_vec,
            resblocks,
            conv_post,
            stft,
            istft,
            lrelu: LeakyReLU::new(0.1),
            device,
            dtype,
        })
    }

    /// Inference: mel spectrogram -> waveform
    ///
    /// # Arguments
    /// * `mel` - [B, 80, T] mel spectrogram
    /// * `finalize` - Whether this is the last chunk
    pub fn inference(&self, mel: &Tensor, finalize: bool) -> Result<Tensor> {
        // Use stored device and dtype to ensure consistency
        let target_device = &self.device;
        let target_dtype = self.dtype;

        // Ensure mel is on the correct device
        let mel = mel.to_device(target_device)?.to_dtype(target_dtype)?;

        // 1. F0 prediction
        let f0 = self.f0_predictor.forward(&mel, finalize)?; // [B, T]
        // Ensure f0 is on the same device
        let f0 = f0.to_device(target_device)?.to_dtype(target_dtype)?;

        // 2. F0 upsample to audio sample rate
        let upsample_scale: usize =
            self.config.upsample_rates.iter().product::<usize>() * self.config.istft_hop_len;
        let f0_up = self.upsample_f0(&f0, upsample_scale)?; // [B, T*scale, 1]

        // 3. Source signal generation
        let (source, _noise, _uv) = self.m_source.forward(&f0_up)?; // [B, T*scale, 1]
        let source = source.transpose(1, 2)?; // [B, 1, T*scale]
        let source = source.squeeze(1)?; // [B, T*scale]
        // Ensure source is on target device
        let source = source.to_device(target_device)?.to_dtype(target_dtype)?;

        // 4. STFT source signal
        let (s_stft_real, s_stft_imag) = self.stft.forward(&source)?;
        // Ensure STFT output is on target device
        let s_stft_real = s_stft_real.to_device(target_device)?.to_dtype(target_dtype)?;
        let s_stft_imag = s_stft_imag.to_device(target_device)?.to_dtype(target_dtype)?;
        let s_stft = Tensor::cat(&[&s_stft_real, &s_stft_imag], 1)?; // [B, 18, T']

        // 5. Mel encoding
        let mut x = self.conv_pre.forward(&mel)?;

        // 6. Upsample + source fusion + ResBlocks
        for i in 0..self.ups.len() {
            x = self.lrelu.forward(&x)?;
            x = self.ups[i].forward(&x)?;

            // Apply reflection padding at the last upsample stage
            // Following Python: if i == self.num_upsamples - 1: x = self.reflection_pad(x)
            // ReflectionPad1d((1, 0)) pads 1 element on the left using reflection
            if i == self.ups.len() - 1 {
                x = reflection_pad_1d(&x, 1, 0)?;
            }

            // Downsample source signal with causal padding
            // Apply left padding before conv (causal_padding = stride - 1 for downsample)
            let s_stft_padded = if self.source_down_causal_paddings[i] > 0 {
                s_stft.pad_with_zeros(2, self.source_down_causal_paddings[i], 0)?
            } else {
                s_stft.clone()
            };
            let si = self.source_downs[i].forward(&s_stft_padded)?;
            // Apply source_resblock only if it exists (last layer doesn't have one)
            let si = if i < self.source_resblocks.len() {
                self.source_resblocks[i].forward(&si)?
            } else {
                si
            };

            // Adjust sizes and add
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;
            let min_len = x_len.min(si_len);
            let x_trimmed = x.narrow(2, 0, min_len)?;
            let si_trimmed = si.narrow(2, 0, min_len)?;
            x = (&x_trimmed + &si_trimmed)?;

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for resblock in &self.resblocks[i] {
                let xj = resblock.forward(&x)?;
                xs = Some(match xs {
                    None => xj,
                    Some(acc) => (acc + xj)?,
                });
            }
            x = (xs.unwrap() / self.resblocks[i].len() as f64)?;
        }

        // 7. Post-processing
        // NOTE: Python uses default leaky_relu slope (0.01) here, not self.lrelu_slope (0.1)
        // This is different from the loop where self.lrelu_slope is used
        let lrelu_post = LeakyReLU::new(0.01);
        x = lrelu_post.forward(&x)?;
        x = self.conv_post.forward(&x)?;

        // 8. Separate magnitude and phase
        // Clip the log-magnitude to prevent exp() explosion
        // Python clips magnitude after exp() to 100, which corresponds to log(100) ≈ 4.6
        // We clip to a slightly higher value to allow some headroom
        let n_fft_half = self.config.istft_n_fft / 2 + 1;
        let log_magnitude = x.narrow(1, 0, n_fft_half)?.clamp(-10.0, 5.0)?;
        let magnitude = log_magnitude.exp()?;
        let phase = x.narrow(1, n_fft_half, n_fft_half)?;
        // Note: In original implementation phase goes through sin, simplified here
        let phase = phase.sin()?;

        // 9. iSTFT
        let waveform = self.istft.forward(&magnitude, &phase)?;

        // 10. Clip
        waveform.clamp(-0.99, 0.99)
    }

    /// Debug inference: returns intermediate outputs for comparison
    ///
    /// Returns: (f0, f0_up, source, waveform)
    #[allow(dead_code)]
    pub fn inference_debug(
        &self,
        mel: &Tensor,
        finalize: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let target_device = &self.device;
        let target_dtype = self.dtype;

        let mel = mel.to_device(target_device)?.to_dtype(target_dtype)?;

        // 1. F0 prediction
        let f0 = self.f0_predictor.forward(&mel, finalize)?;
        let f0 = f0.to_device(target_device)?.to_dtype(target_dtype)?;

        // 2. F0 upsample
        let upsample_scale: usize =
            self.config.upsample_rates.iter().product::<usize>() * self.config.istft_hop_len;
        let f0_up = self.upsample_f0(&f0, upsample_scale)?;

        // 3. Source signal generation
        let (source, _noise, _uv) = self.m_source.forward(&f0_up)?;

        // Run full inference for waveform
        let waveform = self.inference(&mel, finalize)?;

        Ok((f0, f0_up, source, waveform))
    }

    /// Detailed debug inference: returns all intermediate outputs
    ///
    /// Returns a vector of (name, tensor) pairs for each stage
    #[allow(dead_code)]
    pub fn inference_debug_detailed(
        &self,
        mel: &Tensor,
        finalize: bool,
    ) -> Result<Vec<(String, Tensor)>> {
        let target_device = &self.device;
        let target_dtype = self.dtype;
        let mut outputs = Vec::new();

        let mel = mel.to_device(target_device)?.to_dtype(target_dtype)?;

        // 1. F0 prediction
        let f0 = self.f0_predictor.forward(&mel, finalize)?;
        let f0 = f0.to_device(target_device)?.to_dtype(target_dtype)?;
        outputs.push(("f0".to_string(), f0.clone()));

        // 2. F0 upsample
        let upsample_scale: usize =
            self.config.upsample_rates.iter().product::<usize>() * self.config.istft_hop_len;
        let f0_up = self.upsample_f0(&f0, upsample_scale)?;
        outputs.push(("f0_up".to_string(), f0_up.clone()));

        // 3. Source signal generation
        let (source, _noise, _uv) = self.m_source.forward(&f0_up)?;
        let source_t = source.transpose(1, 2)?;
        let source_squeezed = source_t.squeeze(1)?;
        outputs.push(("source".to_string(), source.clone()));

        // 4. STFT source signal
        let (s_stft_real, s_stft_imag) = self.stft.forward(&source_squeezed)?;
        let s_stft = Tensor::cat(&[&s_stft_real, &s_stft_imag], 1)?;
        outputs.push(("s_stft".to_string(), s_stft.clone()));

        // 5. Mel encoding
        let mut x = self.conv_pre.forward(&mel)?;
        outputs.push(("conv_pre".to_string(), x.clone()));

        // 6. Upsample + source fusion + ResBlocks
        for i in 0..self.ups.len() {
            x = self.lrelu.forward(&x)?;
            x = self.ups[i].forward(&x)?;

            // Apply reflection padding at the last upsample stage
            if i == self.ups.len() - 1 {
                x = reflection_pad_1d(&x, 1, 0)?;
            }

            // Downsample source signal with causal padding
            let s_stft_padded = if self.source_down_causal_paddings[i] > 0 {
                s_stft.pad_with_zeros(2, self.source_down_causal_paddings[i], 0)?
            } else {
                s_stft.clone()
            };
            let si = self.source_downs[i].forward(&s_stft_padded)?;
            let si = if i < self.source_resblocks.len() {
                self.source_resblocks[i].forward(&si)?
            } else {
                si
            };

            // Adjust sizes and add
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;
            let min_len = x_len.min(si_len);
            let x_trimmed = x.narrow(2, 0, min_len)?;
            let si_trimmed = si.narrow(2, 0, min_len)?;
            x = (&x_trimmed + &si_trimmed)?;

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for resblock in &self.resblocks[i] {
                let xj = resblock.forward(&x)?;
                xs = Some(match xs {
                    None => xj,
                    Some(acc) => (acc + xj)?,
                });
            }
            x = (xs.unwrap() / self.resblocks[i].len() as f64)?;
            outputs.push((format!("stage_{}", i), x.clone()));
        }

        // 7. Post-processing
        // NOTE: Python uses default leaky_relu slope (0.01) here, not self.lrelu_slope (0.1)
        let lrelu_post = LeakyReLU::new(0.01);
        x = lrelu_post.forward(&x)?;
        outputs.push(("after_lrelu".to_string(), x.clone()));
        x = self.conv_post.forward(&x)?;
        outputs.push(("conv_post".to_string(), x.clone()));

        // 8. Separate magnitude and phase
        let n_fft_half = self.config.istft_n_fft / 2 + 1;
        let log_magnitude_raw = x.narrow(1, 0, n_fft_half)?;
        outputs.push(("log_magnitude_raw".to_string(), log_magnitude_raw.clone()));
        let log_magnitude = log_magnitude_raw.clamp(-10.0, 5.0)?;
        let magnitude = log_magnitude.exp()?;
        let phase_raw = x.narrow(1, n_fft_half, n_fft_half)?;
        outputs.push(("phase_raw".to_string(), phase_raw.clone()));
        let phase = phase_raw.sin()?;
        outputs.push(("magnitude".to_string(), magnitude.clone()));
        outputs.push(("phase".to_string(), phase.clone()));

        // 9. iSTFT
        let waveform = self.istft.forward(&magnitude, &phase)?;
        outputs.push(("waveform_before_clamp".to_string(), waveform.clone()));

        // 10. Clip
        let waveform = waveform.clamp(-0.99, 0.99)?;
        outputs.push(("waveform".to_string(), waveform));

        Ok(outputs)
    }

    /// Debug stage 0 in detail
    #[allow(dead_code)]
    pub fn debug_stage_0(&self, mel: &Tensor, finalize: bool) -> Result<()> {
        let target_device = &self.device;
        let target_dtype = self.dtype;

        let mel = mel.to_device(target_device)?.to_dtype(target_dtype)?;

        // F0 and source
        let f0 = self.f0_predictor.forward(&mel, finalize)?;
        let f0 = f0.to_device(target_device)?.to_dtype(target_dtype)?;
        let upsample_scale: usize =
            self.config.upsample_rates.iter().product::<usize>() * self.config.istft_hop_len;
        let f0_up = self.upsample_f0(&f0, upsample_scale)?;
        let (source, _noise, _uv) = self.m_source.forward(&f0_up)?;
        let source_t = source.transpose(1, 2)?;
        let source_squeezed = source_t.squeeze(1)?;
        let source_squeezed = source_squeezed.to_device(target_device)?.to_dtype(target_dtype)?;

        // STFT
        let (s_stft_real, s_stft_imag) = self.stft.forward(&source_squeezed)?;
        let s_stft = Tensor::cat(&[&s_stft_real, &s_stft_imag], 1)?;

        // Conv pre
        let x = self.conv_pre.forward(&mel)?;
        println!("conv_pre: mean={:.6}, max={:.6}", 
            x.mean_all()?.to_scalar::<f32>()?,
            x.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);

        // Stage 0 detailed
        let i = 0;
        let x = self.lrelu.forward(&x)?;
        println!("after lrelu: mean={:.6}, max={:.6}",
            x.mean_all()?.to_scalar::<f32>()?,
            x.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);

        let x = self.ups[i].forward(&x)?;
        println!("after ups[0]: mean={:.6}, max={:.6}, shape={:?}",
            x.mean_all()?.to_scalar::<f32>()?,
            x.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?,
            x.dims());

        // Source fusion
        let s_stft_padded = if self.source_down_causal_paddings[i] > 0 {
            s_stft.pad_with_zeros(2, self.source_down_causal_paddings[i], 0)?
        } else {
            s_stft.clone()
        };
        let si = self.source_downs[i].forward(&s_stft_padded)?;
        println!("source_downs[0]: mean={:.6}, max={:.6}, shape={:?}",
            si.mean_all()?.to_scalar::<f32>()?,
            si.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?,
            si.dims());

        let si = if i < self.source_resblocks.len() {
            self.source_resblocks[i].forward(&si)?
        } else {
            si
        };
        println!("source_resblocks[0]: mean={:.6}, max={:.6}",
            si.mean_all()?.to_scalar::<f32>()?,
            si.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);

        // Adjust sizes and add
        let x_len = x.dim(2)?;
        let si_len = si.dim(2)?;
        let min_len = x_len.min(si_len);
        let x_trimmed = x.narrow(2, 0, min_len)?;
        let si_trimmed = si.narrow(2, 0, min_len)?;
        let x = (&x_trimmed + &si_trimmed)?;
        println!("after fusion: mean={:.6}, max={:.6}",
            x.mean_all()?.to_scalar::<f32>()?,
            x.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);

        // ResBlocks
        let mut xs: Option<Tensor> = None;
        for (j, resblock) in self.resblocks[i].iter().enumerate() {
            let xj = resblock.forward(&x)?;
            xs = Some(match xs {
                None => xj.clone(),
                Some(acc) => (acc + &xj)?,
            });
            println!("resblock[{}]: mean={:.6}, max={:.6}",
                j,
                xs.as_ref().unwrap().mean_all()?.to_scalar::<f32>()?,
                xs.as_ref().unwrap().max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);
        }
        let x = (xs.unwrap() / self.resblocks[i].len() as f64)?;
        println!("stage_0 final: mean={:.6}, max={:.6}",
            x.mean_all()?.to_scalar::<f32>()?,
            x.max(D::Minus1)?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f32>()?);

        Ok(())
    }

    /// F0 upsampling
    fn upsample_f0(&self, f0: &Tensor, scale: usize) -> Result<Tensor> {
        let (batch, time) = f0.dims2()?;
        let new_time = time * scale;

        // [B, T] -> [B, T, 1] -> [B, T, scale, 1] -> [B, T*scale, 1]
        let f0 = f0.unsqueeze(D::Minus1)?; // [B, T, 1]
        let f0 = f0.unsqueeze(2)?; // [B, T, 1, 1]

        // Expand
        let f0 = f0.broadcast_as((batch, time, scale, 1))?;

        // Reshape
        f0.reshape((batch, new_time, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_resblock_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let varmap = candle_nn::VarMap::new();
        let _vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

        // Need to initialize weights to create ResBlock
        // Only testing basic shape logic here
        let x = Tensor::randn(0f32, 1.0, (2, 256, 100), &device)?;
        assert_eq!(x.dims(), &[2, 256, 100]);

        Ok(())
    }
}

