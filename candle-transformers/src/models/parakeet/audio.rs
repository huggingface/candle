use candle::{DType, Device, Result, Tensor};
use std::process::Command;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PreprocessArgs {
    pub sample_rate: usize,
    pub normalize: String,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub features: usize,
    pub n_fft: usize,
    pub dither: f64,
    #[serde(default)]
    pub pad_to: usize,
    #[serde(default)]
    pub pad_value: f64,
    #[serde(default = "default_preemph")]
    pub preemph: Option<f64>,
    #[serde(default = "default_mag_power")]
    pub mag_power: f64,
}

fn default_preemph() -> Option<f64> {
    Some(0.97)
}

fn default_mag_power() -> f64 {
    2.0
}

impl PreprocessArgs {
    pub fn win_length(&self) -> usize {
        (self.window_size * self.sample_rate as f64) as usize
    }

    pub fn hop_length(&self) -> usize {
        (self.window_stride * self.sample_rate as f64) as usize
    }
}

pub fn load_audio(path: &std::path::Path, sampling_rate: usize) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-i",
            path.to_string_lossy().as_ref(),
            "-threads",
            "0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            &sampling_rate.to_string(),
            "-",
        ])
        .output()
        .map_err(|e| candle::Error::Msg(format!("failed to run ffmpeg: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(candle::Error::Msg(format!("ffmpeg failed: {stderr}")));
    }

    let mut samples = Vec::with_capacity(output.stdout.len() / 2);
    for chunk in output.stdout.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample as f32 / 32768.0);
    }
    Ok(samples)
}

fn hz_to_mel(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if freq < min_log_hz {
        freq / f_sp
    } else {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fmin = 0.0;
    let fmax = sr as f64 / 2.0;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_points.push(mel_min + (mel_max - mel_min) * t);
    }

    let hz_points: Vec<f64> = mel_points.into_iter().map(mel_to_hz).collect();
    let bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft + 1) as f64 * hz / sr as f64).floor() as usize)
        .collect();

    let n_fft_bins = n_fft / 2 + 1;
    let mut filters = vec![0f32; n_mels * n_fft_bins];
    for m in 0..n_mels {
        let f_m_minus = bins[m];
        let f_m = bins[m + 1];
        let f_m_plus = bins[m + 2];

        if f_m_minus == f_m || f_m == f_m_plus {
            continue;
        }

        for k in f_m_minus..f_m {
            if k < n_fft_bins {
                filters[m * n_fft_bins + k] =
                    (k as f64 - f_m_minus as f64) as f32 / (f_m as f64 - f_m_minus as f64) as f32;
            }
        }
        for k in f_m..f_m_plus {
            if k < n_fft_bins {
                filters[m * n_fft_bins + k] =
                    (f_m_plus as f64 - k as f64) as f32 / (f_m_plus as f64 - f_m as f64) as f32;
            }
        }
    }

    // Slaney-style normalization
    for m in 0..n_mels {
        let f_m_minus = hz_points[m];
        let f_m_plus = hz_points[m + 2];
        let enorm = 2.0 / (f_m_plus - f_m_minus).max(1e-6);
        for k in 0..n_fft_bins {
            filters[m * n_fft_bins + k] *= enorm as f32;
        }
    }

    filters
}

fn fft(inp: &[f32]) -> Vec<f32> {
    let n = inp.len();
    let zero = 0f32;
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![zero; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &value) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(value);
        } else {
            odd.push(value);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = std::f32::consts::PI * 2.0;
    let n_t = n as f32;
    for k in 0..n / 2 {
        let k_t = k as f32;
        let theta = two_pi * k_t / n_t;
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
    let zero = 0f32;
    let n = inp.len();
    let two_pi = std::f32::consts::PI * 2.0;

    let mut out = Vec::with_capacity(2 * n);
    let n_t = n as f32;
    for k in 0..n {
        let k_t = k as f32;
        let mut re = zero;
        let mut im = zero;

        for (j, &value) in inp.iter().enumerate() {
            let j_t = j as f32;
            let angle = two_pi * k_t * j_t / n_t;
            re += value * angle.cos();
            im -= value * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

fn window_values(kind: &str, len: usize) -> Vec<f32> {
    match kind {
        "hann" | "hanning" => (0..len)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos())
            .collect(),
        "hamming" => (0..len)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos())
            .collect(),
        "blackman" => (0..len)
            .map(|i| {
                0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / len as f32).cos()
            })
            .collect(),
        "bartlett" => (0..len)
            .map(|i| {
                let v = (i as f32 - (len as f32 - 1.0) / 2.0).abs();
                1.0 - v / ((len as f32 - 1.0) / 2.0)
            })
            .collect(),
        _ => vec![1.0; len],
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() || pad == 0 {
        return samples.to_vec();
    }
    if samples.len() < 2 {
        let mut out = Vec::with_capacity(samples.len() + 2 * pad);
        out.extend(std::iter::repeat(samples[0]).take(pad));
        out.extend_from_slice(samples);
        out.extend(std::iter::repeat(samples[0]).take(pad));
        return out;
    }
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    let prefix = samples[1..=pad.min(samples.len() - 1)]
        .iter()
        .rev()
        .cloned();
    let suffix = samples[samples.len().saturating_sub(pad + 1)..samples.len() - 1]
        .iter()
        .rev()
        .cloned();
    out.extend(prefix);
    out.extend_from_slice(samples);
    out.extend(suffix);
    out
}

pub fn get_logmel(samples: &[f32], args: &PreprocessArgs, device: &Device) -> Result<Tensor> {
    let mut audio = samples.to_vec();
    if args.pad_to > 0 && audio.len() < args.pad_to {
        audio.resize(args.pad_to, args.pad_value as f32);
    }

    if let Some(preemph) = args.preemph {
        if audio.len() > 1 {
            let mut emphasized = Vec::with_capacity(audio.len());
            emphasized.push(audio[0]);
            for i in 1..audio.len() {
                emphasized.push(audio[i] - preemph as f32 * audio[i - 1]);
            }
            audio = emphasized;
        }
    }

    let win_length = args.win_length();
    let hop_length = args.hop_length();
    let n_fft = args.n_fft;

    let window = window_values(&args.window, win_length);
    let pad = n_fft / 2;
    let padded = reflect_pad(&audio, pad);

    let frame_count = if padded.len() < win_length {
        0
    } else {
        (padded.len() - win_length + hop_length) / hop_length
    };

    let n_fft_bins = n_fft / 2 + 1;
    let filters = mel_filterbank(args.sample_rate, n_fft, args.features);

    let mut mel = vec![0f32; args.features * frame_count];
    for frame in 0..frame_count {
        let start = frame * hop_length;
        let mut frame_buf = vec![0f32; n_fft];
        let slice = &padded[start..start + win_length];
        for (i, &v) in slice.iter().enumerate() {
            frame_buf[i] = v * window[i];
        }

        let fft_out = fft(&frame_buf);
        let mut mags = vec![0f32; n_fft_bins];
        for k in 0..n_fft_bins {
            let re = fft_out[2 * k];
            let im = fft_out[2 * k + 1];
            let mut mag = re.abs() + im.abs();
            if (args.mag_power - 1.0).abs() > f64::EPSILON {
                mag = mag.powf(args.mag_power as f32);
            }
            mags[k] = mag;
        }

        for mel_idx in 0..args.features {
            let mut sum = 0.0f32;
            let filter_offset = mel_idx * n_fft_bins;
            for k in 0..n_fft_bins {
                sum += filters[filter_offset + k] * mags[k];
            }
            mel[mel_idx * frame_count + frame] = (sum + 1e-5).ln();
        }
    }

    if args.normalize == "per_feature" {
        for mel_idx in 0..args.features {
            let offset = mel_idx * frame_count;
            let slice = &mel[offset..offset + frame_count];
            let mean = slice.iter().sum::<f32>() / frame_count.max(1) as f32;
            let var =
                slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / frame_count.max(1) as f32;
            let std = var.sqrt();
            for v in &mut mel[offset..offset + frame_count] {
                *v = (*v - mean) / (std + 1e-5);
            }
        }
    } else {
        let mean = mel.iter().sum::<f32>() / mel.len().max(1) as f32;
        let var = mel.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / mel.len().max(1) as f32;
        let std = var.sqrt();
        for v in &mut mel {
            *v = (*v - mean) / (std + 1e-5);
        }
    }

    // shape: (features, frames) -> (frames, features)
    let mut mel_t = vec![0f32; mel.len()];
    for mel_idx in 0..args.features {
        for frame in 0..frame_count {
            mel_t[frame * args.features + mel_idx] = mel[mel_idx * frame_count + frame];
        }
    }

    let mel_tensor = Tensor::from_vec(mel_t, (frame_count, args.features), device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?;
    Ok(mel_tensor)
}
