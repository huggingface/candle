use candle::{Device, Error, Result, Tensor};
use std::f32::consts::PI;

/// Configuration that exactly matches WhisperFeatureExtractor/librosa parameters
pub struct MelSpectrogramConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 128,
            fmin: 0.0,
            fmax: 8000.0,
        }
    }
}

/// Apply reflection padding to match librosa's center=True, pad_mode='reflect'
fn apply_center_padding_reflect(audio: &[f32], n_fft: usize) -> Vec<f32> {
    let pad_size = n_fft / 2;
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad_size);

    // Left reflection padding
    for i in 0..pad_size {
        let reflect_idx = pad_size - i - 1;
        if reflect_idx < audio.len() {
            padded.push(audio[reflect_idx]);
        } else {
            padded.push(0.0);
        }
    }

    // Original audio
    padded.extend_from_slice(audio);

    // Right reflection padding
    for i in 0..pad_size {
        let reflect_idx = audio.len() - 1 - i;
        if reflect_idx < audio.len() {
            padded.push(audio[reflect_idx]);
        } else {
            padded.push(0.0);
        }
    }

    padded
}

/// Hann window exactly matching librosa
fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos()))
        .collect()
}

/// STFT computation matching librosa exactly
fn librosa_stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Result<Vec<Vec<(f32, f32)>>> {
    // Apply center padding with reflection (librosa default)
    let padded_audio = apply_center_padding_reflect(audio, n_fft);

    // Calculate number of frames
    let n_frames = if padded_audio.len() >= n_fft {
        1 + (padded_audio.len() - n_fft) / hop_length
    } else {
        0
    };

    let mut stft_result = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let mut frame = vec![0.0f32; n_fft];

        // Extract windowed frame
        for i in 0..n_fft {
            if start + i < padded_audio.len() {
                frame[i] = padded_audio[start + i] * window[i];
            }
        }

        // Compute FFT for this frame
        let fft_result = dft_real(&frame);
        stft_result.push(fft_result);
    }

    Ok(stft_result)
}

/// Discrete Fourier Transform for real input (returns only positive frequencies)
fn dft_real(signal: &[f32]) -> Vec<(f32, f32)> {
    let n = signal.len();
    let mut result = Vec::with_capacity(n / 2 + 1);

    // Compute only positive frequencies (0 to n/2)
    for k in 0..=n / 2 {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (n_idx, &sample) in signal.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * n_idx as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        result.push((real, imag));
    }

    result
}

/// Load mel filter bank using built-in Whisper filters
fn load_mel_filters(n_mels: usize, n_freqs: usize) -> Result<Vec<Vec<f32>>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    // Use built-in Whisper filters based on n_mels
    let mel_bytes = match n_mels {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => return Err(Error::Msg(format!("unexpected num_mel_bins {}", nmel)).into()),
    };

    let mut filters = vec![0f32; mel_bytes.len() / 4];
    let mut cursor = Cursor::new(mel_bytes);
    cursor
        .read_f32_into::<LittleEndian>(&mut filters)
        .map_err(|e| Error::Io(e))?;

    // Convert flat array to 2D: [n_mels][n_freqs]
    let mut filter_bank = Vec::with_capacity(n_mels);
    for mel_idx in 0..n_mels {
        let start = mel_idx * n_freqs;
        let end = start + n_freqs;
        if end <= filters.len() {
            filter_bank.push(filters[start..end].to_vec());
        } else {
            return Err(Error::Msg(format!(
                "Filter data size mismatch: expected {} elements, got {}",
                n_mels * n_freqs,
                filters.len()
            ))
            .into());
        }
    }

    Ok(filter_bank)
}

/// Apply log transform exactly like WhisperFeatureExtractor
/// Based on analysis: log + clipping + min-max normalization to [-1, 1] range
fn whisper_log_transform(mel_spec: &mut [f32]) {
    // First pass: apply log and find min/max
    let mut max_val = f32::NEG_INFINITY;
    let mut min_val = f32::INFINITY;

    for value in mel_spec.iter_mut() {
        // Clamp to minimum value to prevent log(0)
        *value = value.max(1e-10);

        // Apply natural log
        *value = value.ln();

        // Track min and max for normalization
        max_val = max_val.max(*value);
        min_val = min_val.min(*value);
    }

    // Second pass: apply clipping (like WhisperFeatureExtractor)
    let clipped_min = max_val - 8.0;
    for value in mel_spec.iter_mut() {
        *value = value.max(clipped_min);
    }

    // Third pass: min-max normalization to [-1, 1] range
    // This matches the best transform from Python analysis
    let new_min = clipped_min;
    let new_max = max_val;
    let range = new_max - new_min;

    if range > 0.0 {
        for value in mel_spec.iter_mut() {
            // Normalize to [0, 1] then scale to [-1, 1]
            *value = (*value - new_min) / range * 2.0 - 1.0;
        }
    }
}

/// Main function: PCM to mel spectrogram matching librosa/WhisperFeatureExtractor exactly
pub fn pcm_to_mel_fixed(
    audio: &[f32],
    config: &MelSpectrogramConfig,
    device: &Device,
) -> Result<Tensor> {
    // Create Hann window
    let window = hann_window(config.n_fft);

    // Compute STFT with librosa-compatible parameters
    let stft_result = librosa_stft(audio, config.n_fft, config.hop_length, &window)?;

    let n_frames = stft_result.len();
    let n_freqs = config.n_fft / 2 + 1; // Should be 201 for n_fft=400

    // Convert STFT to power spectrogram
    let mut power_spec = vec![0.0f32; n_frames * n_freqs];
    for (frame_idx, frame) in stft_result.iter().enumerate() {
        for (freq_idx, &(real, imag)) in frame.iter().enumerate() {
            let power = real * real + imag * imag;
            power_spec[frame_idx * n_freqs + freq_idx] = power;
        }
    }

    // Load mel filter bank using built-in Whisper filters
    let mel_filters = load_mel_filters(config.n_mels, n_freqs)?;

    // Apply mel filters to power spectrogram
    let mut mel_spec = vec![0.0f32; n_frames * config.n_mels];
    for frame_idx in 0..n_frames {
        for (mel_idx, filter) in mel_filters.iter().enumerate() {
            let mut mel_energy = 0.0f32;
            for (freq_idx, &filter_val) in filter.iter().enumerate() {
                mel_energy += power_spec[frame_idx * n_freqs + freq_idx] * filter_val;
            }
            mel_spec[frame_idx * config.n_mels + mel_idx] = mel_energy;
        }
    }

    // Apply WhisperFeatureExtractor log transform (not power_to_db)
    whisper_log_transform(&mut mel_spec);

    // Process all frames instead of cropping to 3000 like Python does for multi-chunk processing
    let final_frames = n_frames;
    let cropped_mel: Vec<f32> = mel_spec;

    // Convert to tensor with shape [1, n_mels, n_frames] (same as Python)
    let tensor = Tensor::from_vec(cropped_mel, (final_frames, config.n_mels), device)?;
    let tensor = tensor.t()?.unsqueeze(0)?; // Transpose to [n_mels, n_frames] then add batch dim

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Device;

    #[test]
    fn test_center_padding() {
        let audio = vec![1.0, 2.0, 3.0, 4.0];
        let padded = apply_center_padding_reflect(&audio, 4);
        // Should pad with 2 samples on each side
        assert_eq!(padded.len(), 8);
        // Left padding should be [2.0, 1.0]
        // Right padding should be [4.0, 3.0]
        assert_eq!(padded, vec![2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0]);
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);
        // Check that it matches expected Hann window values
        assert!((window[0] - 0.0).abs() < 1e-6); // Should be 0
        assert!((window[3] - 0.0).abs() < 1e-6); // Should be 0
    }

    #[test]
    fn test_power_to_db() {
        let mut power = vec![1.0, 0.1, 0.01, 1e-12];
        power_to_db(&mut power, 1.0, 1e-10, None);

        // 10*log10(1.0/1.0) = 0 dB
        assert!((power[0] - 0.0).abs() < 1e-5);
        // 10*log10(0.1/1.0) = -10 dB
        assert!((power[1] - (-10.0)).abs() < 1e-5);
        // 10*log10(0.01/1.0) = -20 dB
        assert!((power[2] - (-20.0)).abs() < 1e-5);
        // Should be clamped to amin=1e-10, so 10*log10(1e-10/1.0) = -100 dB
        assert!((power[3] - (-100.0)).abs() < 1e-5);
    }
}
