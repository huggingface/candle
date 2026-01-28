//! Qwen3-TTS speaker encoder (ECAPA-TDNN) and mel-spectrogram helper.

use candle::{Result, Tensor, D};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, VarBuilder};

use super::config::Qwen3TtsSpeakerEncoderConfig;

const MEL_N_FFT: usize = 1024;
const MEL_HOP: usize = 256;
const MEL_WIN: usize = 1024;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = 12_000.0;

fn reflect_pad_1d(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(xs.clone());
    }
    let dims = xs.dims();
    let last_dim = dims.len() - 1;
    let len = xs.dim(last_dim)?;
    if (left > 0 && len < left + 1) || (right > 0 && len < right + 1) {
        candle::bail!("input too short for reflect pad (len={len}, left={left}, right={right})");
    }
    let mut parts: Vec<Tensor> = Vec::new();
    if left > 0 {
        let slice = xs.narrow(last_dim, 1, left)?;
        parts.push(slice.flip(&[last_dim])?);
    }
    parts.push(xs.clone());
    if right > 0 {
        let slice = xs.narrow(last_dim, len - right - 1, right)?;
        parts.push(slice.flip(&[last_dim])?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, D::Minus1)
}

#[derive(Debug, Clone)]
struct TimeDelayNetBlock {
    conv: Conv1d,
    padding: usize,
}

impl TimeDelayNetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            stride: 1,
            dilation,
            groups: 1,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        let padding = (kernel_size.saturating_sub(1) * dilation) / 2;
        Ok(Self { conv, padding })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = reflect_pad_1d(xs, self.padding, self.padding)?;
        xs.apply(&self.conv)?.relu()
    }
}

#[derive(Debug, Clone)]
struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;
        let mut blocks = Vec::with_capacity(scale.saturating_sub(1));
        for idx in 0..scale.saturating_sub(1) {
            blocks.push(TimeDelayNetBlock::new(
                in_channel,
                hidden_channel,
                kernel_size,
                dilation,
                vb.pp(format!("blocks.{idx}")),
            )?);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let parts = xs.chunk(self.scale, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor> = None;
        for (i, part) in parts.into_iter().enumerate() {
            let out = if i == 0 {
                part
            } else if i == 1 {
                self.blocks[i - 1].forward(&part)?
            } else {
                let sum = (&part + prev.as_ref().unwrap())?;
                self.blocks[i - 1].forward(&sum)?
            };
            prev = Some(out.clone());
            outputs.push(out);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 1)
    }
}

#[derive(Debug, Clone)]
struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn new(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            stride: 1,
            dilation: 1,
            groups: 1,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(in_channels, se_channels, 1, cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(se_channels, out_channels, 1, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
        let hidden = mean.apply(&self.conv1)?.relu()?.apply(&self.conv2)?;
        let hidden = candle_nn::ops::sigmoid(&hidden)?;
        xs.broadcast_mul(&hidden)
    }
}

#[derive(Debug, Clone)]
struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock,
    conv: Conv1d,
    eps: f64,
}

impl AttentiveStatisticsPooling {
    fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        let tdnn = TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?;
        let cfg = Conv1dConfig {
            stride: 1,
            dilation: 1,
            groups: 1,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(attention_channels, channels, 1, cfg, vb.pp("conv"))?;
        Ok(Self {
            tdnn,
            conv,
            eps: 1e-12,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        let mean = xs.mean(D::Minus1)?;
        let mean_b = mean.unsqueeze(D::Minus1)?.broadcast_as((b, c, t))?;
        let var = ((xs - &mean_b)?.sqr()?).mean(D::Minus1)?;
        let std = var.clamp(self.eps as f32, f32::INFINITY)?.sqrt()?;
        let std_b = std.unsqueeze(D::Minus1)?.broadcast_as((b, c, t))?;
        let attention_in = Tensor::cat(&[xs, &mean_b, &std_b], 1)?;
        let mut attn = self.tdnn.forward(&attention_in)?;
        attn = attn.tanh()?.apply(&self.conv)?;
        let attn = candle_nn::ops::softmax(&attn, 2)?;
        let mean_w = xs.broadcast_mul(&attn)?.sum(D::Minus1)?;
        let mean_w_b = mean_w.unsqueeze(D::Minus1)?.broadcast_as((b, c, t))?;
        let var_w = ((xs - &mean_w_b)?.sqr()?.broadcast_mul(&attn)?).sum(D::Minus1)?;
        let std_w = var_w.clamp(self.eps as f32, f32::INFINITY)?.sqrt()?;
        let pooled = Tensor::cat(&[&mean_w, &std_w], 1)?.unsqueeze(D::Minus1)?;
        Ok(pooled)
    }
}

#[derive(Debug, Clone)]
struct SqueezeExcitationRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
}

impl SqueezeExcitationRes2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TimeDelayNetBlock::new(in_channels, out_channels, 1, 1, vb.pp("tdnn1"))?,
            res2net_block: Res2NetBlock::new(
                out_channels,
                out_channels,
                res2net_scale,
                kernel_size,
                dilation,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TimeDelayNetBlock::new(out_channels, out_channels, 1, 1, vb.pp("tdnn2"))?,
            se_block: SqueezeExcitationBlock::new(
                out_channels,
                se_channels,
                out_channels,
                vb.pp("se_block"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.tdnn1.forward(xs)?;
        let xs = self.res2net_block.forward(&xs)?;
        let xs = self.tdnn2.forward(&xs)?;
        let xs = self.se_block.forward(&xs)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
enum SpeakerBlock {
    Tdnn(TimeDelayNetBlock),
    SeRes2Net(SqueezeExcitationRes2NetBlock),
}

impl SpeakerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            SpeakerBlock::Tdnn(block) => block.forward(xs),
            SpeakerBlock::SeRes2Net(block) => block.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3TtsSpeakerEncoder {
    blocks: Vec<SpeakerBlock>,
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
}

impl Qwen3TtsSpeakerEncoder {
    pub fn new(cfg: &Qwen3TtsSpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.enc_channels.len() != cfg.enc_kernel_sizes.len()
            || cfg.enc_channels.len() != cfg.enc_dilations.len()
        {
            candle::bail!("enc_channels, enc_kernel_sizes and enc_dilations must have same length");
        }
        let mut blocks: Vec<SpeakerBlock> = Vec::new();
        let vb_blocks = vb.pp("blocks");
        blocks.push(SpeakerBlock::Tdnn(TimeDelayNetBlock::new(
            cfg.mel_dim,
            cfg.enc_channels[0],
            cfg.enc_kernel_sizes[0],
            cfg.enc_dilations[0],
            vb_blocks.pp(0),
        )?));
        for i in 1..cfg.enc_channels.len() - 1 {
            blocks.push(SpeakerBlock::SeRes2Net(SqueezeExcitationRes2NetBlock::new(
                cfg.enc_channels[i - 1],
                cfg.enc_channels[i],
                cfg.enc_res2net_scale,
                cfg.enc_se_channels,
                cfg.enc_kernel_sizes[i],
                cfg.enc_dilations[i],
                vb_blocks.pp(i),
            )?));
        }
        let mfa = TimeDelayNetBlock::new(
            cfg.enc_channels[cfg.enc_channels.len() - 1],
            cfg.enc_channels[cfg.enc_channels.len() - 1],
            cfg.enc_kernel_sizes[cfg.enc_kernel_sizes.len() - 1],
            cfg.enc_dilations[cfg.enc_dilations.len() - 1],
            vb.pp("mfa"),
        )?;
        let asp = AttentiveStatisticsPooling::new(
            cfg.enc_channels[cfg.enc_channels.len() - 1],
            cfg.enc_attention_channels,
            vb.pp("asp"),
        )?;
        let cfg_fc = Conv1dConfig {
            stride: 1,
            dilation: 1,
            groups: 1,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let fc = conv1d(
            cfg.enc_channels[cfg.enc_channels.len() - 1] * 2,
            cfg.enc_dim,
            1,
            cfg_fc,
            vb.pp("fc"),
        )?;
        Ok(Self {
            blocks,
            mfa,
            asp,
            fc,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.transpose(1, 2)?; // [B, mel, T]
        let mut hidden_states: Vec<Tensor> = Vec::new();
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?;
            hidden_states.push(xs.clone());
        }
        if hidden_states.len() > 1 {
            let refs: Vec<&Tensor> = hidden_states.iter().skip(1).collect();
            xs = Tensor::cat(&refs, 1)?;
        }
        xs = self.mfa.forward(&xs)?;
        xs = self.asp.forward(&xs)?;
        xs = xs.apply(&self.fc)?;
        xs.squeeze(D::Minus1)
    }
}

fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filter_bank(
    num_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = if fmax > 0.0 {
        fmax
    } else {
        sample_rate as f32 / 2.0
    };
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mut mel_points = Vec::with_capacity(num_mels + 2);
    for i in 0..num_mels + 2 {
        let mel = mel_min + (mel_max - mel_min) * (i as f32) / ((num_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }
    let mut bins = Vec::with_capacity(num_mels + 2);
    for hz in mel_points.iter() {
        bins.push((((n_fft + 1) as f32) * hz / sample_rate as f32).floor() as usize);
    }
    let mut filters = vec![0f32; num_mels * n_freqs];
    for m in 0..num_mels {
        let f_left = bins[m];
        let f_center = bins[m + 1];
        let f_right = bins[m + 2];
        if f_center <= f_left || f_right <= f_center {
            continue;
        }
        for k in f_left..f_center {
            if k < n_freqs {
                filters[m * n_freqs + k] = (k - f_left) as f32 / (f_center - f_left) as f32;
            }
        }
        for k in f_center..f_right {
            if k < n_freqs {
                filters[m * n_freqs + k] = (f_right - k) as f32 / (f_right - f_center) as f32;
            }
        }
        let enorm = 2.0 / (mel_points[m + 2] - mel_points[m]).max(1e-6);
        for k in f_left..f_right {
            if k < n_freqs {
                filters[m * n_freqs + k] *= enorm;
            }
        }
    }
    filters
}

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    if n == 1 {
        return vec![inp[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![0f32; n * 2];
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);
    for (i, &v) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(v);
        } else {
            odd.push(v);
        }
    }
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);
    let two_pi = std::f32::consts::PI * 2.0;
    for k in 0..n / 2 {
        let theta = two_pi * k as f32 / n as f32;
        let re = theta.cos();
        let im = -theta.sin();
        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];
        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn dft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    let two_pi = std::f32::consts::PI * 2.0;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &v) in inp.iter().enumerate() {
            let angle = two_pi * k as f32 * j as f32 / n as f32;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

pub fn mel_spectrogram(
    audio: &[f32],
    cfg: &Qwen3TtsSpeakerEncoderConfig,
) -> Result<(Vec<f32>, usize)> {
    let pad = (MEL_N_FFT - MEL_HOP) / 2;
    if audio.len() <= pad + 1 {
        candle::bail!("audio too short for mel spectrogram");
    }
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);
    for i in 0..pad {
        padded.push(audio[pad - i]);
    }
    padded.extend_from_slice(audio);
    for i in 0..pad {
        padded.push(audio[audio.len() - 2 - i]);
    }
    let filters = mel_filter_bank(cfg.mel_dim, MEL_N_FFT, cfg.sample_rate, MEL_FMIN, MEL_FMAX);
    let n_freqs = MEL_N_FFT / 2 + 1;
    let frames = if padded.len() >= MEL_N_FFT {
        1 + (padded.len() - MEL_N_FFT) / MEL_HOP
    } else {
        1
    };
    let hann: Vec<f32> = (0..MEL_WIN)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / MEL_WIN as f32).cos())
        .collect();
    let mut mel_out = vec![0f32; frames * cfg.mel_dim];
    for frame in 0..frames {
        let start = frame * MEL_HOP;
        let mut windowed = vec![0f32; MEL_N_FFT];
        for i in 0..MEL_WIN {
            if start + i < padded.len() {
                windowed[i] = padded[start + i] * hann[i];
            }
        }
        let fft_out = fft(&windowed);
        let mut mags = vec![0f32; n_freqs];
        for k in 0..n_freqs {
            let re = fft_out[2 * k];
            let im = fft_out[2 * k + 1];
            mags[k] = (re * re + im * im).sqrt();
        }
        for m in 0..cfg.mel_dim {
            let mut sum = 0.0f32;
            let filter = &filters[m * n_freqs..(m + 1) * n_freqs];
            for k in 0..n_freqs {
                sum += mags[k] * filter[k];
            }
            let val = if sum < 1e-5 { 1e-5 } else { sum };
            mel_out[frame * cfg.mel_dim + m] = val.ln();
        }
    }
    Ok((mel_out, frames))
}
