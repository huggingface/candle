//! Audio Processing for CosyVoice3
//!
//! Implements mel spectrogram extraction and Kaldi-compatible Fbank features.

use candle::{DType, Device, Result, Tensor};
use std::f64::consts::PI;

/// Mel Spectrogram Extractor
///
/// Fully CPU-based implementation, output is automatically transferred to target device.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    #[allow(dead_code)]
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Vec<f32>,
    hann_window: Vec<f32>,
    target_device: Device,
}

impl MelSpectrogram {
    /// Create CosyVoice Speech Feat configuration
    pub fn new_cosyvoice_speech_feat(device: &Device) -> Result<Self> {
        Self::new(24000, 1920, 480, 80, device)
    }

    /// Create Whisper format configuration (for speech tokenizer)
    ///
    /// Uses librosa-compatible mel filters matching OpenAI Whisper's mel_filters.npz.
    /// This ensures speech tokens match the Python implementation exactly.
    pub fn new_whisper_format(device: &Device) -> Result<Self> {
        Self::new_whisper_with_librosa_filters(16000, 400, 160, 128, device)
    }

    /// Create Whisper format with librosa-compatible mel filters
    ///
    /// This matches OpenAI Whisper's mel spectrogram exactly by using
    /// Slaney-normalized mel filters computed the same way as librosa.
    fn new_whisper_with_librosa_filters(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        device: &Device,
    ) -> Result<Self> {
        let hann_window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let x = (2.0 * PI * i as f64) / n_fft as f64;
                (0.5 * (1.0 - x.cos())) as f32
            })
            .collect();

        // Use librosa-compatible mel filters (Slaney normalization)
        let mel_filters = Self::create_librosa_mel_filters(sample_rate, n_fft, n_mels);

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            hann_window,
            target_device: device.clone(),
        })
    }

    pub fn new(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        device: &Device,
    ) -> Result<Self> {
        // Pre-compute Hann window
        let hann_window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let x = (2.0 * PI * i as f64) / n_fft as f64;
                (0.5 * (1.0 - x.cos())) as f32
            })
            .collect();

        // Pre-compute Mel filter bank
        let mel_filters = Self::create_mel_filters(sample_rate, n_fft, n_mels);

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            hann_window,
            target_device: device.clone(),
        })
    }

    /// Create librosa-compatible mel filter bank with Slaney normalization
    ///
    /// This matches `librosa.filters.mel(sr, n_fft, n_mels)` which is used by OpenAI Whisper.
    /// Key differences from HTK-style:
    /// 1. Uses Slaney formula for mel scale (not HTK)
    /// 2. Applies area normalization (Slaney norm)
    fn create_librosa_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0f32; n_mels * n_freqs];

        // Slaney-style mel conversion (librosa default)
        // f_mel = 15 * log2(1 + f / 700) for f >= 1000
        // f_mel = 3 * f / 200 for f < 1000
        let f_min = 0.0f64;
        let f_max = sample_rate as f64 / 2.0;

        // Compute mel breakpoint
        let min_log_hz = 1000.0f64;
        let min_log_mel = 15.0f64; // 3 * 1000 / 200
        let logstep = 6.4f64.ln() / 27.0; // log(6.4) / 27 ≈ 0.0687

        let hz_to_mel_slaney = |hz: f64| -> f64 {
            if hz >= min_log_hz {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            } else {
                3.0 * hz / 200.0
            }
        };

        let mel_to_hz_slaney = |mel: f64| -> f64 {
            if mel >= min_log_mel {
                min_log_hz * ((mel - min_log_mel) * logstep).exp()
            } else {
                200.0 * mel / 3.0
            }
        };

        let mel_min = hz_to_mel_slaney(f_min);
        let mel_max = hz_to_mel_slaney(f_max);

        // Create n_mels + 2 equally spaced points in mel space
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

        // Convert to FFT bin indices (using floor like librosa)
        let fft_freqs: Vec<f64> = (0..n_freqs)
            .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
            .collect();

        // Build triangular filters
        for m in 0..n_mels {
            let left_hz = hz_points[m];
            let center_hz = hz_points[m + 1];
            let right_hz = hz_points[m + 2];

            // Compute filter weights for each frequency bin
            for k in 0..n_freqs {
                let freq = fft_freqs[k];

                let weight = if freq >= left_hz && freq < center_hz {
                    (freq - left_hz) / (center_hz - left_hz).max(1e-10)
                } else if freq >= center_hz && freq <= right_hz {
                    (right_hz - freq) / (right_hz - center_hz).max(1e-10)
                } else {
                    0.0
                };

                filters[m * n_freqs + k] = weight as f32;
            }

            // Apply Slaney normalization: normalize by bandwidth
            // enorm = 2.0 / (hz_points[m+2] - hz_points[m])
            let enorm = 2.0 / (right_hz - left_hz).max(1e-10);
            for k in 0..n_freqs {
                filters[m * n_freqs + k] *= enorm as f32;
            }
        }

        filters
    }

    fn create_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0f32; n_mels * n_freqs];

        // Hz to Mel conversion (HTK style)
        let hz_to_mel = |hz: f64| -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |mel: f64| -> f64 { 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0) };

        let mel_low = hz_to_mel(0.0);
        let mel_high = hz_to_mel(sample_rate as f64 / 2.0);

        // Evenly distributed points on Mel scale
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_mels + 1) as f64)
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft as f64 + 1.0) * hz / sample_rate as f64).floor() as usize)
            .collect();

        for m in 0..n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for k in left..center {
                if k < n_freqs {
                    filters[m * n_freqs + k] =
                        (k - left) as f32 / (center - left).max(1) as f32;
                }
            }
            for k in center..right {
                if k < n_freqs {
                    filters[m * n_freqs + k] =
                        (right - k) as f32 / (right - center).max(1) as f32;
                }
            }
        }

        filters
    }

    /// Extract Mel Spectrogram
    ///
    /// # Arguments
    /// * `audio` - [samples] or [1, samples] audio waveform
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward_internal(audio, false)
    }

    /// Extract Mel Spectrogram with CosyVoice/Matcha-TTS style padding
    ///
    /// This matches the Matcha-TTS mel_spectrogram function which adds
    /// reflect padding of (n_fft - hop_size) / 2 on each side before STFT.
    ///
    /// # Arguments
    /// * `audio` - [samples] or [1, samples] audio waveform
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram
    #[allow(clippy::needless_range_loop)]
    pub fn forward_cosyvoice(&self, audio: &Tensor) -> Result<Tensor> {
        self.forward_internal(audio, true)
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_internal(&self, audio: &Tensor, use_matcha_padding: bool) -> Result<Tensor> {
        // Ensure 1D input
        let audio = if audio.dims().len() == 2 {
            audio.squeeze(0)?
        } else {
            audio.clone()
        };

        // Convert to CPU f32 for processing
        let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1()?;

        // Apply Matcha-TTS style reflect padding if requested
        // pad_size = (n_fft - hop_length) / 2 on each side
        let padded_samples = if use_matcha_padding {
            let pad_size = (self.n_fft - self.hop_length) / 2;
            let padded_len = samples.len() + 2 * pad_size;
            let mut padded = vec![0.0f32; padded_len];

            // Reflect padding on left side
            for i in 0..pad_size {
                padded[pad_size - 1 - i] = samples[(i + 1).min(samples.len() - 1)];
            }

            // Copy original samples
            for (i, &s) in samples.iter().enumerate() {
                padded[pad_size + i] = s;
            }

            // Reflect padding on right side
            for i in 0..pad_size {
                let src_idx = samples.len().saturating_sub(2 + i);
                padded[pad_size + samples.len() + i] = samples[src_idx];
            }

            padded
        } else {
            samples.clone()
        };

        // Calculate number of frames
        let n_samples = padded_samples.len();
        let n_frames = if n_samples >= self.n_fft {
            (n_samples - self.n_fft) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((1, self.n_mels, 0), DType::F32, &self.target_device);
        }

        let n_freqs = self.n_fft / 2 + 1;
        let mut mel_output = vec![0.0f32; self.n_mels * n_frames];

        // Process frame by frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window + FFT
            let mut fft_input = vec![0.0f32; self.n_fft * 2]; // Complex representation
            for i in 0..self.n_fft {
                if start + i < n_samples {
                    fft_input[i * 2] = padded_samples[start + i] * self.hann_window[i];
                }
            }

            // Use Cooley-Tukey FFT
            let fft_out = Self::fft_complex(&fft_input);

            // Calculate power spectrum
            let mut power_spectrum = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = fft_out[k * 2];
                let im = fft_out[k * 2 + 1];
                power_spectrum[k] = re * re + im * im;
            }

            // Apply Mel filters
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += power_spectrum[k] * self.mel_filters[m * n_freqs + k];
                }
                // Log-mel
                mel_output[m * n_frames + frame_idx] = (sum.max(1e-10)).log10();
            }
        }

        // Transfer to target device
        let mel = Tensor::from_vec(mel_output, (self.n_mels, n_frames), &Device::Cpu)?;
        mel.unsqueeze(0)?.to_device(&self.target_device)
    }

    /// Extract Mel Spectrogram matching OpenAI Whisper's implementation
    ///
    /// This method matches `whisper.log_mel_spectrogram()` exactly:
    /// 1. Uses center=True padding (like torch.stft default)
    /// 2. Drops the last frame (magnitudes[..., :-1])
    /// 3. Uses librosa-compatible mel filters (Slaney normalization)
    ///
    /// # Arguments
    /// * `audio` - [samples] or [1, samples] audio waveform at 16kHz
    ///
    /// # Returns
    /// * `mel` - [1, n_mels, T] Mel spectrogram matching whisper output
    #[allow(clippy::needless_range_loop)]
    pub fn forward_whisper(&self, audio: &Tensor) -> Result<Tensor> {
        // Ensure 1D input
        let audio = if audio.dims().len() == 2 {
            audio.squeeze(0)?
        } else {
            audio.clone()
        };

        // Convert to CPU f32 for processing
        let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1()?;
        let n_samples = samples.len();

        // Apply center padding (like torch.stft with center=True)
        // Pad n_fft // 2 on each side
        let pad_len = self.n_fft / 2;
        let padded_len = n_samples + 2 * pad_len;
        let mut padded_samples = vec![0.0f32; padded_len];
        for i in 0..n_samples {
            padded_samples[pad_len + i] = samples[i];
        }

        // Calculate number of frames (torch.stft formula with center=True)
        // n_frames = 1 + n_samples // hop_length
        let n_frames_stft = 1 + n_samples / self.hop_length;

        // Whisper drops the last frame: magnitudes[..., :-1]
        let n_frames = if n_frames_stft > 0 {
            n_frames_stft - 1
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((1, self.n_mels, 0), DType::F32, &self.target_device);
        }

        let n_freqs = self.n_fft / 2 + 1;
        let mut mel_output = vec![0.0f32; self.n_mels * n_frames];

        // Process frame by frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window + FFT
            let mut fft_input = vec![0.0f32; self.n_fft * 2]; // Complex representation
            for i in 0..self.n_fft {
                if start + i < padded_len {
                    fft_input[i * 2] = padded_samples[start + i] * self.hann_window[i];
                }
            }

            // Use Cooley-Tukey FFT
            let fft_out = Self::fft_complex(&fft_input);

            // Calculate power spectrum (|STFT|^2)
            let mut power_spectrum = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = fft_out[k * 2];
                let im = fft_out[k * 2 + 1];
                power_spectrum[k] = re * re + im * im;
            }

            // Apply Mel filters
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += power_spectrum[k] * self.mel_filters[m * n_freqs + k];
                }
                // Log-mel (log10)
                mel_output[m * n_frames + frame_idx] = (sum.max(1e-10)).log10();
            }
        }

        // Transfer to target device
        let mel = Tensor::from_vec(mel_output, (self.n_mels, n_frames), &Device::Cpu)?;
        mel.unsqueeze(0)?.to_device(&self.target_device)
    }

    fn fft_complex(input: &[f32]) -> Vec<f32> {
        let n = input.len() / 2;
        if n == 1 {
            return input.to_vec();
        }
        if n % 2 == 1 {
            return Self::dft(input);
        }

        let mut even = Vec::with_capacity(n);
        let mut odd = Vec::with_capacity(n);
        for i in 0..n / 2 {
            even.push(input[i * 4]);
            even.push(input[i * 4 + 1]);
            odd.push(input[i * 4 + 2]);
            odd.push(input[i * 4 + 3]);
        }

        let even_fft = Self::fft_complex(&even);
        let odd_fft = Self::fft_complex(&odd);

        let mut out = vec![0.0f32; n * 2];
        let two_pi = 2.0 * PI;
        for k in 0..n / 2 {
            let angle = two_pi * k as f64 / n as f64;
            let cos_a = angle.cos() as f32;
            let sin_a = (-angle.sin()) as f32;

            let odd_re = odd_fft[k * 2];
            let odd_im = odd_fft[k * 2 + 1];
            let t_re = cos_a * odd_re - sin_a * odd_im;
            let t_im = cos_a * odd_im + sin_a * odd_re;

            out[k * 2] = even_fft[k * 2] + t_re;
            out[k * 2 + 1] = even_fft[k * 2 + 1] + t_im;
            out[(k + n / 2) * 2] = even_fft[k * 2] - t_re;
            out[(k + n / 2) * 2 + 1] = even_fft[k * 2 + 1] - t_im;
        }
        out
    }

    fn dft(input: &[f32]) -> Vec<f32> {
        let n = input.len() / 2;
        let mut out = vec![0.0f32; n * 2];
        let two_pi = 2.0 * PI;

        for k in 0..n {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for j in 0..n {
                let angle = two_pi * k as f64 * j as f64 / n as f64;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                re += input[j * 2] * cos_a + input[j * 2 + 1] * sin_a;
                im += -input[j * 2] * sin_a + input[j * 2 + 1] * cos_a;
            }
            out[k * 2] = re;
            out[k * 2 + 1] = im;
        }
        out
    }
}

/// Kaldi-compatible Fbank feature extractor
///
/// Equivalent to torchaudio.compliance.kaldi.fbank.
/// Used for CAMPPlus speaker embedding extraction.
#[derive(Debug, Clone)]
pub struct KaldiFbank {
    #[allow(dead_code)]
    sample_rate: usize,
    n_mels: usize,
    preemphasis: f64,
    window_size: usize,
    hop_length: usize,
    padded_window_size: usize,
    povey_window: Vec<f32>,
    mel_filters: Vec<f32>,
}

impl KaldiFbank {
    /// Create CosyVoice CAMPPlus configuration
    pub fn new_for_campplus() -> Self {
        Self::new(16000, 80, 25.0, 10.0, 0.97, 20.0, 0.0)
    }

    pub fn new(
        sample_rate: usize,
        n_mels: usize,
        frame_length_ms: f64,
        frame_shift_ms: f64,
        preemphasis: f64,
        low_freq: f64,
        high_freq: f64,
    ) -> Self {
        let window_size = (sample_rate as f64 * frame_length_ms * 0.001) as usize;
        let hop_length = (sample_rate as f64 * frame_shift_ms * 0.001) as usize;
        let padded_window_size = window_size.next_power_of_two();

        let actual_high_freq = if high_freq <= 0.0 {
            sample_rate as f64 / 2.0 + high_freq
        } else {
            high_freq
        };

        let povey_window = Self::create_povey_window(window_size);
        let mel_filters = Self::create_mel_filters(
            sample_rate,
            padded_window_size,
            n_mels,
            low_freq,
            actual_high_freq,
        );

        Self {
            sample_rate,
            n_mels,
            preemphasis,
            window_size,
            hop_length,
            padded_window_size,
            povey_window,
            mel_filters,
        }
    }

    fn create_povey_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f64 / size as f64).cos();
                hann.powf(0.85) as f32
            })
            .collect()
    }

    fn create_mel_filters(
        sample_rate: usize,
        padded_window_size: usize,
        n_mels: usize,
        low_freq: f64,
        high_freq: f64,
    ) -> Vec<f32> {
        // Kaldi/torchaudio uses n_fft/2 bins (excluding Nyquist)
        // The Nyquist bin is handled separately by padding with 0
        let n_fft_bins = padded_window_size / 2;
        // We store n_fft_bins + 1 to include the Nyquist bin (which will be 0)
        let mut filters = vec![0.0f32; n_mels * (n_fft_bins + 1)];

        // Kaldi/HTK mel scale
        let hz_to_mel = |hz: f64| -> f64 { 1127.0 * (1.0 + hz / 700.0).ln() };

        let mel_low = hz_to_mel(low_freq);
        let mel_high = hz_to_mel(high_freq);

        // Mel frequency delta between filter centers
        let mel_freq_delta = (mel_high - mel_low) / (n_mels + 1) as f64;

        // FFT bin width in Hz
        let fft_bin_width = sample_rate as f64 / padded_window_size as f64;

        // Build triangular mel filters (matching torchaudio/Kaldi exactly)
        // Filter m has:
        //   left_mel = mel_low + m * mel_freq_delta
        //   center_mel = mel_low + (m + 1) * mel_freq_delta
        //   right_mel = mel_low + (m + 2) * mel_freq_delta
        for m in 0..n_mels {
            let left_mel = mel_low + m as f64 * mel_freq_delta;
            let center_mel = mel_low + (m + 1) as f64 * mel_freq_delta;
            let right_mel = mel_low + (m + 2) as f64 * mel_freq_delta;

            for k in 0..n_fft_bins {
                // Convert bin to mel: mel(k * fft_bin_width)
                let k_mel = hz_to_mel(k as f64 * fft_bin_width);

                // Compute triangular filter weight in mel domain
                // up_slope = (mel - left_mel) / (center_mel - left_mel)
                // down_slope = (right_mel - mel) / (right_mel - center_mel)
                // weight = max(0, min(up_slope, down_slope))
                let up_slope = (k_mel - left_mel) / (center_mel - left_mel);
                let down_slope = (right_mel - k_mel) / (right_mel - center_mel);
                let weight = up_slope.min(down_slope).max(0.0);

                filters[m * (n_fft_bins + 1) + k] = weight as f32;
            }
            // Nyquist bin (k = n_fft_bins) is left as 0
        }

        filters
    }

    /// Extract Fbank features
    ///
    /// Matches torchaudio.compliance.kaldi.fbank exactly:
    /// Per-frame processing order: DC removal → pre-emphasis → window → FFT
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let samples: Vec<f32> = audio.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;

        // Calculate number of frames
        let n_frames = if samples.len() >= self.window_size {
            1 + (samples.len() - self.window_size) / self.hop_length
        } else {
            0
        };

        if n_frames == 0 {
            return Tensor::zeros((0, self.n_mels), DType::F32, audio.device());
        }

        // rfft output has n_fft/2 + 1 bins (DC to Nyquist)
        let n_fft_bins = self.padded_window_size / 2 + 1;
        let mut fbank = vec![0.0f32; n_frames * self.n_mels];
        let alpha = self.preemphasis as f32;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // 1. Extract raw frame
            let mut frame: Vec<f32> = (0..self.window_size)
                .map(|i| samples.get(start + i).copied().unwrap_or(0.0))
                .collect();

            // 2. DC offset removal (per-frame, before pre-emphasis)
            let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
            for x in frame.iter_mut() {
                *x -= mean;
            }

            // 3. Pre-emphasis (per-frame, replicate padding style like torchaudio)
            // frame[i] = frame[i] - alpha * frame[i-1], with frame[-1] = frame[0] (replicate)
            if alpha != 0.0 {
                let mut prev = frame[0]; // replicate padding: frame[-1] = frame[0]
                for i in 0..self.window_size {
                    let current = frame[i];
                    frame[i] = current - alpha * prev;
                    prev = current;
                }
            }

            // 4. Apply Povey window
            for (i, x) in frame.iter_mut().enumerate() {
                *x *= self.povey_window[i];
            }

            // 5. Zero padding
            frame.resize(self.padded_window_size, 0.0);

            // 6. FFT (returns n_fft_bins complex values as interleaved [re, im, re, im, ...])
            let fft_out = self.rfft(&frame);

            // 7. Power spectrum from rfft output (all n_fft_bins including Nyquist)
            let mut power = vec![0.0f32; n_fft_bins];
            for k in 0..n_fft_bins {
                let re = fft_out[k * 2];
                let im = fft_out[k * 2 + 1];
                power[k] = re * re + im * im;
            }

            // 8. Mel filter + log
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_fft_bins {
                    sum += power[k] * self.mel_filters[m * n_fft_bins + k];
                }
                fbank[frame_idx * self.n_mels + m] = sum.max(f32::EPSILON).ln();
            }
        }

        Tensor::from_vec(fbank, (n_frames, self.n_mels), audio.device())
    }

    fn rfft(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut complex_input = vec![0.0f32; n * 2];
        for (i, &x) in input.iter().enumerate() {
            complex_input[i * 2] = x;
        }

        let fft_out = MelSpectrogram::fft_complex(&complex_input);
        fft_out[..(n + 2)].to_vec()
    }
}

/// Convenience function: Extract Kaldi-style Fbank
pub fn kaldi_fbank(audio: &Tensor, n_mels: usize) -> Result<Tensor> {
    let extractor = KaldiFbank::new(16000, n_mels, 25.0, 10.0, 0.97, 20.0, 0.0);
    extractor.forward(audio)
}

/// High-quality audio resampling using windowed sinc interpolation
///
/// This implementation matches torchaudio.functional.resample with sinc_interp_hann method.
/// The algorithm uses bandlimited interpolation with a Hann-windowed sinc function.
///
/// # Algorithm
/// The sinc interpolation formula reconstructs the continuous signal:
///   x(t) = sum_i x[i] * sinc(π * orig_freq * (i/orig_freq - t))
///
/// We then sample at the new rate:
///   y[j] = x(j / new_freq)
///
/// A Hann window is applied to limit the filter width and reduce ringing artifacts.
///
/// # Parameters
/// - `lowpass_filter_width`: Controls sharpness (default: 6, matching torchaudio)
/// - `rolloff`: Anti-aliasing rolloff frequency (default: 0.99)
#[allow(clippy::needless_range_loop)]
pub fn resample(audio: &Tensor, from_sr: usize, to_sr: usize) -> Result<Tensor> {
    resample_sinc(audio, from_sr, to_sr, 6, 0.99)
}

/// Sinc interpolation resampling with configurable parameters
///
/// # Arguments
/// * `audio` - Input audio tensor [samples]
/// * `from_sr` - Original sample rate
/// * `to_sr` - Target sample rate
/// * `lowpass_filter_width` - Filter width in zero crossings (higher = sharper but slower)
/// * `rolloff` - Anti-aliasing rolloff (0.0-1.0, lower = more anti-aliasing)
#[allow(clippy::needless_range_loop)]
pub fn resample_sinc(
    audio: &Tensor,
    from_sr: usize,
    to_sr: usize,
    lowpass_filter_width: usize,
    rolloff: f64,
) -> Result<Tensor> {
    if from_sr == to_sr {
        return Ok(audio.clone());
    }

    let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_vec1()?;
    let device = audio.device();

    // Compute GCD to reduce the resampling ratio
    let gcd = gcd(from_sr, to_sr);
    let orig_freq = from_sr / gcd;
    let new_freq = to_sr / gcd;

    // Base frequency for anti-aliasing
    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;

    // Filter width (number of input samples to consider on each side)
    let width = ((lowpass_filter_width as f64 * orig_freq as f64 / base_freq).ceil()) as usize;

    // Build resampling kernels for each phase
    // We need new_freq different kernels (one for each output phase)
    let kernel_size = 2 * width + orig_freq;
    let mut kernels: Vec<Vec<f64>> = Vec::with_capacity(new_freq);

    for phase in 0..new_freq {
        let mut kernel = Vec::with_capacity(kernel_size);
        let phase_offset = -(phase as f64) / new_freq as f64;

        for k in 0..kernel_size {
            let idx = (k as i64 - width as i64) as f64 / orig_freq as f64;
            let t = (phase_offset + idx) * base_freq;

            // Clamp t for numerical stability
            let t_clamped = t.clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);

            // Hann window: cos²(π * t / (2 * width))
            let window = (t_clamped * PI / (2.0 * lowpass_filter_width as f64)).cos().powi(2);

            // Sinc function: sin(π * t) / (π * t), with sinc(0) = 1
            let t_pi = t_clamped * PI;
            let sinc = if t_pi.abs() < 1e-10 {
                1.0
            } else {
                t_pi.sin() / t_pi
            };

            // Combined kernel with scaling
            let scale = base_freq / orig_freq as f64;
            kernel.push(sinc * window * scale);
        }
        kernels.push(kernel);
    }

    // Pad input signal
    let padded_len = samples.len() + 2 * width + orig_freq;
    let mut padded = vec![0.0f64; padded_len];
    for (i, &s) in samples.iter().enumerate() {
        padded[width + i] = s as f64;
    }

    // Calculate output length
    let output_len = (samples.len() * new_freq + orig_freq - 1) / orig_freq;

    // Apply resampling via strided convolution
    let mut resampled = Vec::with_capacity(output_len);

    for out_idx in 0..output_len {
        // Determine which kernel to use (phase)
        let phase = out_idx % new_freq;
        let kernel = &kernels[phase];

        // Determine input position (stride by orig_freq in the rational resampling sense)
        let in_base = (out_idx / new_freq) * orig_freq;

        // Convolve
        let mut sum = 0.0f64;
        for (k, &w) in kernel.iter().enumerate() {
            let in_idx = in_base + k;
            if in_idx < padded_len {
                sum += padded[in_idx] * w;
            }
        }
        resampled.push(sum as f32);
    }

    // Trim to expected length
    let expected_len = (samples.len() as f64 * to_sr as f64 / from_sr as f64).round() as usize;
    resampled.truncate(expected_len);

    let len = resampled.len();
    Tensor::from_vec(resampled, len, device)
}

/// Greatest Common Divisor using Euclidean algorithm
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram() -> Result<()> {
        let device = Device::Cpu;

        let mel_extractor = MelSpectrogram::new_cosyvoice_speech_feat(&device)?;

        // Create 1 second test audio (24kHz)
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 24000.0).sin() as f32)
            .collect();
        let audio = Tensor::from_vec(samples, 24000, &device)?;

        let mel = mel_extractor.forward(&audio)?;

        // Check shape: [1, 80, T]
        assert_eq!(mel.dim(0)?, 1);
        assert_eq!(mel.dim(1)?, 80);
        assert!(mel.dim(2)? > 0);

        Ok(())
    }

    #[test]
    fn test_resample() -> Result<()> {
        let device = Device::Cpu;

        let audio = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4, &device)?;
        let resampled = resample(&audio, 2, 4)?;

        assert_eq!(resampled.dim(0)?, 8);
        Ok(())
    }
}

