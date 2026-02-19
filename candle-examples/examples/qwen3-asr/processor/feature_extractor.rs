//! Audio feature extraction.
//!
//! The official processor uses `WhisperFeatureExtractor` (log-mel, padding, attention mask).
//! This module aims to match the HuggingFace implementation in:
//! `transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor`.

use anyhow::Result;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Features {
    pub input_features: Vec<Vec<f32>>,
    pub feature_attention_mask: Vec<u8>,
}

/// Extract Whisper-style log-mel features for a mono 16k waveform.
///
/// Returns:
/// - `input_features`: shape `(num_mel_bins=128, frames)`, matching HF layout.
/// - `feature_attention_mask`: shape `(frames,)`, 1 for real frames (no padding).
pub fn extract_features(wav_16k_mono: &[f32]) -> Result<Features> {
    // Defaults from `preprocessor_config.json` used by Qwen3-ASR.
    let cfg = WhisperFeatureExtractorConfig::default();
    let frames = wav_16k_mono.len() / cfg.hop_length;

    if frames == 0 {
        return Ok(Features {
            input_features: vec![vec![]; cfg.feature_size],
            feature_attention_mask: vec![],
        });
    }

    let wav_f64: Vec<f64> = wav_16k_mono.iter().map(|&x| f64::from(x)).collect();
    let padded = reflect_pad(&wav_f64, cfg.n_fft / 2, cfg.n_fft / 2);
    let window = hann_window_periodic(cfg.n_fft);
    let mel_filters_t = mel_filter_bank_slaney_t(
        1 + cfg.n_fft / 2,
        cfg.feature_size,
        0.0,
        cfg.max_frequency_hz,
        cfg.sampling_rate_hz,
    );

    let n_freq = 1 + cfg.n_fft / 2;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(cfg.n_fft);

    let mut buf: Vec<rustfft::num_complex::Complex<f64>> =
        vec![rustfft::num_complex::Complex::new(0.0, 0.0); cfg.n_fft];
    let mut mags: Vec<f64> = vec![0.0; n_freq];

    // (feature_size, frames)
    let mut log_spec: Vec<Vec<f32>> = (0..cfg.feature_size)
        .map(|_| Vec::with_capacity(frames))
        .collect();
    let mut max_log: f32 = f32::NEG_INFINITY;

    for frame_idx in 0..frames {
        let start = frame_idx * cfg.hop_length;

        for i in 0..cfg.n_fft {
            let x = padded.get(start + i).copied().unwrap_or_default();
            buf[i] = rustfft::num_complex::Complex::new(x * window[i], 0.0);
        }

        fft.process(&mut buf);

        for k in 0..n_freq {
            let c = buf[k];
            mags[k] = c.re.mul_add(c.re, c.im * c.im);
        }

        for (mel_row, filter_row) in log_spec.iter_mut().zip(mel_filters_t.iter()) {
            let mut sum = 0.0f64;
            for (w, mag) in filter_row.iter().zip(mags.iter()) {
                sum = (*w).mul_add(*mag, sum);
            }
            if sum < cfg.mel_floor {
                sum = cfg.mel_floor;
            }
            let log10 = sum.log10() as f32;
            if log10 > max_log {
                max_log = log10;
            }
            mel_row.push(log10);
        }
    }

    // Dynamic range compression, matching HF:
    //   log_spec = max(log_spec, max(log_spec) - 8)
    //   log_spec = (log_spec + 4) / 4
    let floor = max_log - 8.0f32;
    for mel in &mut log_spec {
        for v in mel {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0f32) / 4.0f32;
        }
    }

    Ok(Features {
        input_features: log_spec,
        feature_attention_mask: vec![1u8; frames],
    })
}

/// Incremental log-mel feature extraction for streaming transcription.
///
/// This avoids recomputing FFTs for the entire accumulated waveform each decode step by
/// caching per-frame log-mel values and updating only the affected tail frames.
pub struct StreamingFeatureExtractor {
    cfg: WhisperFeatureExtractorConfig,
    window: Vec<f64>,
    mel_filters_t: Vec<Vec<f64>>,
    fft: Arc<dyn rustfft::Fft<f64>>,
    buf: Vec<Complex<f64>>,
    mags: Vec<f64>,

    wav: Vec<f32>,

    raw_log_spec: Vec<Vec<f32>>,
    input_features: Vec<Vec<f32>>,
    feature_attention_mask: Vec<u8>,
    compressed_max_log: f32,
}

impl StreamingFeatureExtractor {
    pub fn new() -> Self {
        let cfg = WhisperFeatureExtractorConfig::default();
        let window = hann_window_periodic(cfg.n_fft);
        let mel_filters_t = mel_filter_bank_slaney_t(
            1 + cfg.n_fft / 2,
            cfg.feature_size,
            0.0,
            cfg.max_frequency_hz,
            cfg.sampling_rate_hz,
        );

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);

        let n_freq = 1 + cfg.n_fft / 2;
        let buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); cfg.n_fft];
        let mags: Vec<f64> = vec![0.0; n_freq];

        let raw_log_spec: Vec<Vec<f32>> = (0..cfg.feature_size).map(|_| vec![]).collect();
        let input_features: Vec<Vec<f32>> = (0..cfg.feature_size).map(|_| vec![]).collect();

        Self {
            cfg,
            window,
            mel_filters_t,
            fft,
            buf,
            mags,
            wav: vec![],
            raw_log_spec,
            input_features,
            feature_attention_mask: vec![],
            compressed_max_log: f32::NEG_INFINITY,
        }
    }

    pub fn append_samples(&mut self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.wav.extend_from_slice(samples);
        self.update()?;
        Ok(())
    }

    pub fn wav_len(&self) -> usize {
        self.wav.len()
    }

    pub fn input_features(&self) -> &[Vec<f32>] {
        self.input_features.as_slice()
    }

    pub fn feature_attention_mask(&self) -> &[u8] {
        self.feature_attention_mask.as_slice()
    }

    pub fn compressed_max_log(&self) -> f32 {
        self.compressed_max_log
    }

    fn update(&mut self) -> Result<()> {
        let frames_total = self.wav.len() / self.cfg.hop_length;
        let old_frames = self.feature_attention_mask.len();

        let recompute_from = if old_frames == 0 {
            0
        } else {
            old_frames.saturating_sub(1)
        };

        for row in &mut self.raw_log_spec {
            row.truncate(recompute_from);
        }
        for row in &mut self.input_features {
            row.truncate(recompute_from);
        }
        self.feature_attention_mask.truncate(recompute_from);

        if frames_total == 0 {
            self.compressed_max_log = f32::NEG_INFINITY;
            return Ok(());
        }

        let n_fft_half = self.cfg.n_fft / 2;
        for frame_idx in recompute_from..frames_total {
            self.push_frame_raw(frame_idx, n_fft_half)?;
        }
        self.feature_attention_mask.resize(frames_total, 1u8);

        let max_log = compute_max_log(self.raw_log_spec.as_slice());
        let floor = max_log - 8.0f32;

        let max_changed = max_log != self.compressed_max_log;
        self.compressed_max_log = max_log;

        if max_changed {
            for (out_row, raw_row) in self.input_features.iter_mut().zip(self.raw_log_spec.iter()) {
                out_row.clear();
                out_row.reserve(raw_row.len());
                for &v in raw_row {
                    out_row.push(compress_log_mel(v, floor));
                }
            }
            return Ok(());
        }

        for (out_row, raw_row) in self.input_features.iter_mut().zip(self.raw_log_spec.iter()) {
            out_row.reserve(raw_row.len().saturating_sub(out_row.len()));
            for &v in raw_row.iter().skip(recompute_from) {
                out_row.push(compress_log_mel(v, floor));
            }
        }

        Ok(())
    }

    fn push_frame_raw(&mut self, frame_idx: usize, n_fft_half: usize) -> Result<()> {
        if self.wav.is_empty() {
            anyhow::bail!("internal error: cannot extract features on empty audio");
        }

        let wav_len = self.wav.len();
        let max_wav_len = (isize::MAX as usize).saturating_sub(n_fft_half.saturating_sub(1));
        if wav_len > max_wav_len {
            anyhow::bail!("audio too long for reflection padding: wav_len={wav_len}");
        }

        let start = frame_idx
            .checked_mul(self.cfg.hop_length)
            .ok_or_else(|| anyhow::anyhow!("frame start overflow"))?;
        if start > wav_len {
            anyhow::bail!("internal error: frame start {start} exceeds wav_len {wav_len}");
        }

        let start_i =
            isize::try_from(start).map_err(|_| anyhow::anyhow!("frame start overflow"))?;
        let pad_left =
            isize::try_from(n_fft_half).map_err(|_| anyhow::anyhow!("pad_left overflow"))?;

        for i in 0..self.cfg.n_fft {
            let idx = start_i + i as isize - pad_left;
            let src = reflect_index(idx, wav_len);
            let x = f64::from(self.wav[src]);
            self.buf[i] = Complex::new(x * self.window[i], 0.0);
        }

        self.fft.process(&mut self.buf);

        let n_freq = 1 + self.cfg.n_fft / 2;
        for k in 0..n_freq {
            let c = self.buf[k];
            self.mags[k] = c.re.mul_add(c.re, c.im * c.im);
        }

        for (mel_row, filter_row) in self.raw_log_spec.iter_mut().zip(self.mel_filters_t.iter()) {
            let mut sum = 0.0f64;
            for (w, mag) in filter_row.iter().zip(self.mags.iter()) {
                sum = (*w).mul_add(*mag, sum);
            }
            if sum < self.cfg.mel_floor {
                sum = self.cfg.mel_floor;
            }
            mel_row.push(sum.log10() as f32);
        }

        Ok(())
    }
}

impl std::fmt::Debug for StreamingFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingFeatureExtractor")
            .field("wav_len", &self.wav.len())
            .field("frames", &self.feature_attention_mask.len())
            .field("compressed_max_log", &self.compressed_max_log)
            .finish()
    }
}

impl Default for StreamingFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_max_log(rows: &[Vec<f32>]) -> f32 {
    let mut max_log = f32::NEG_INFINITY;
    for row in rows {
        for &v in row {
            if v > max_log {
                max_log = v;
            }
        }
    }
    max_log
}

fn compress_log_mel(v: f32, floor: f32) -> f32 {
    let v = if v < floor { floor } else { v };
    (v + 4.0f32) / 4.0f32
}

#[derive(Debug, Clone, Copy)]
struct WhisperFeatureExtractorConfig {
    feature_size: usize,
    sampling_rate_hz: u32,
    hop_length: usize,
    n_fft: usize,
    max_frequency_hz: f64,
    mel_floor: f64,
}

impl Default for WhisperFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            feature_size: 128,
            sampling_rate_hz: 16_000,
            hop_length: 160,
            n_fft: 400,
            max_frequency_hz: 8_000.0,
            mel_floor: 1e-10,
        }
    }
}

fn hann_window_periodic(n: usize) -> Vec<f64> {
    let n_f64 = n as f64;
    let two_pi = std::f64::consts::PI * 2.0;
    (0..n)
        .map(|i| {
            let x = two_pi * (i as f64) / n_f64;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let mut j = i.rem_euclid(period);
    let len_i = len as isize;
    if j >= len_i {
        j = period - j;
    }
    j as usize
}

fn reflect_pad(signal: &[f64], pad_left: usize, pad_right: usize) -> Vec<f64> {
    if signal.is_empty() {
        return vec![];
    }

    let len_out = signal
        .len()
        .saturating_add(pad_left)
        .saturating_add(pad_right);
    let mut out = Vec::with_capacity(len_out);

    for k in 0..len_out {
        let idx = k as isize - pad_left as isize;
        let src = reflect_index(idx, signal.len());
        out.push(signal[src]);
    }

    out
}

fn hertz_to_mel_slaney(freq_hz: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4f64.ln();

    let mut mels = 3.0 * freq_hz / 200.0;
    if freq_hz >= min_log_hertz {
        mels = min_log_mel + (freq_hz / min_log_hertz).ln() * logstep;
    }
    mels
}

fn mel_to_hertz_slaney(mels: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4f64.ln() / 27.0;

    let mut freq = 200.0 * mels / 3.0;
    if mels >= min_log_mel {
        freq = min_log_hertz * (logstep * (mels - min_log_mel)).exp();
    }
    freq
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * (i as f64)).collect()
}

fn mel_filter_bank_slaney_t(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency_hz: f64,
    max_frequency_hz: f64,
    sampling_rate_hz: u32,
) -> Vec<Vec<f64>> {
    let mel_min = hertz_to_mel_slaney(min_frequency_hz);
    let mel_max = hertz_to_mel_slaney(max_frequency_hz);

    let mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2);
    let filter_freqs: Vec<f64> = mel_freqs.into_iter().map(mel_to_hertz_slaney).collect();

    let nyquist_hz = (sampling_rate_hz / 2) as f64;
    let fft_freqs = linspace(0.0, nyquist_hz, num_frequency_bins);

    let mut filters_t: Vec<Vec<f64>> = vec![vec![0.0; num_frequency_bins]; num_mel_filters];

    for mel_idx in 0..num_mel_filters {
        let left = filter_freqs[mel_idx];
        let center = filter_freqs[mel_idx + 1];
        let right = filter_freqs[mel_idx + 2];

        let denom_left = (center - left).max(f64::MIN_POSITIVE);
        let denom_right = (right - center).max(f64::MIN_POSITIVE);

        // Slaney-style area normalization.
        let enorm = 2.0 / (right - left).max(f64::MIN_POSITIVE);

        for (k, f) in fft_freqs.iter().copied().enumerate() {
            let down = (f - left) / denom_left;
            let up = (right - f) / denom_right;
            let w = down.min(up).max(0.0);
            filters_t[mel_idx][k] = w * enorm;
        }
    }

    filters_t
}

#[cfg(test)]
mod tests {
    use super::{extract_features, StreamingFeatureExtractor};

    #[test]
    fn test_extract_features_empty_audio() -> anyhow::Result<()> {
        let feats = extract_features(&[])?;
        if !feats.input_features.iter().all(|v| v.is_empty()) {
            anyhow::bail!("expected empty input_features");
        }
        if !feats.feature_attention_mask.is_empty() {
            anyhow::bail!("expected empty attention mask");
        }
        Ok(())
    }

    #[test]
    fn test_streaming_feature_extractor_matches_offline_incremental() -> anyhow::Result<()> {
        let wav: Vec<f32> = (0..2400).map(|i| ((i as f32) * 0.01).sin() * 0.3).collect();

        let base_len = 1600usize;
        let extra_len = 80usize;
        if wav.len() <= base_len + extra_len {
            anyhow::bail!("test waveform too short");
        }

        let offline0 = extract_features(&wav[..base_len])?;
        let offline1 = extract_features(&wav[..base_len + extra_len])?;
        let offline2 = extract_features(&wav)?;

        let mut s = StreamingFeatureExtractor::new();
        s.append_samples(&wav[..base_len])?;
        assert_features_close(s.input_features(), s.feature_attention_mask(), &offline0)?;

        s.append_samples(&wav[base_len..base_len + extra_len])?;
        assert_features_close(s.input_features(), s.feature_attention_mask(), &offline1)?;

        s.append_samples(&wav[base_len + extra_len..])?;
        assert_features_close(s.input_features(), s.feature_attention_mask(), &offline2)?;

        Ok(())
    }

    fn assert_features_close(
        got_feats: &[Vec<f32>],
        got_mask: &[u8],
        expected: &super::Features,
    ) -> anyhow::Result<()> {
        if got_mask != expected.feature_attention_mask.as_slice() {
            anyhow::bail!("feature_attention_mask mismatch");
        }
        if got_feats.len() != expected.input_features.len() {
            anyhow::bail!(
                "mel bins mismatch: got={} expected={}",
                got_feats.len(),
                expected.input_features.len()
            );
        }
        let eps = 1e-6f32;
        for (i, (got_row, exp_row)) in got_feats
            .iter()
            .zip(expected.input_features.iter())
            .enumerate()
        {
            if got_row.len() != exp_row.len() {
                anyhow::bail!(
                    "row length mismatch at mel={i}: got={} expected={}",
                    got_row.len(),
                    exp_row.len()
                );
            }
            for (j, (&a, &b)) in got_row.iter().zip(exp_row.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > eps {
                    anyhow::bail!("value mismatch at mel={i} frame={j}: diff={diff}");
                }
            }
        }
        Ok(())
    }
}
