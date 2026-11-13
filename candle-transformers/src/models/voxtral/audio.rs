use candle::{Device, Error, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::models::whisper::audio::{log_mel_spectrogram_, Float};

/// Legacy whisper.cpp-style frontend. For checkpoint-compatible Voxtral input,
/// use `extract_features`, which follows WhisperFeatureExtractor's centered STFT.
pub fn pcm_to_mel<T: Float>(samples: &[T], filters: &[T]) -> Vec<T> {
    log_mel_spectrogram_(
        samples,
        filters,
        super::N_FFT,
        super::HOP_LENGTH,
        super::N_MELS,
        false,
    )
}

/// WhisperFeatureExtractor followed by VoxtralProcessor's 30-second chunking.
/// PCM is padded to the next 480000 samples before a centered, reflect-padded
/// 400-point STFT. This does not add whisper.cpp's extra 15-second tail or fold
/// negative-frequency power into the positive-frequency bins.
pub fn extract_features(audio: &[f32], filters: &[f32], device: &Device) -> Result<Tensor, Error> {
    const CHUNK_SAMPLES: usize = 480000;
    const CHUNK_FRAMES: usize = 3000;
    const BINS: usize = super::N_FFT / 2 + 1;
    if audio.is_empty()
        || audio.iter().any(|v| !v.is_finite())
        || filters.len() != super::N_MELS * BINS
        || filters.iter().any(|v| !v.is_finite())
    {
        candle::bail!("invalid Voxtral PCM or mel filters");
    }
    let chunks = audio.len().div_ceil(CHUNK_SAMPLES);
    let padded_len = chunks
        .checked_mul(CHUNK_SAMPLES)
        .ok_or_else(|| Error::Msg("Voxtral padded audio size overflow".into()))?;
    let frames = chunks * CHUNK_FRAMES;
    let mut samples = audio.to_vec();
    samples.resize(padded_len, 0.);
    let window: Vec<f32> = (0..super::N_FFT)
        .map(|i| {
            (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / super::N_FFT as f64).cos()) as f32
        })
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(super::N_FFT);
    let mut input = vec![Complex32::new(0., 0.); super::N_FFT];
    let mut scratch = vec![Complex32::new(0., 0.); fft.get_inplace_scratch_len()];
    let mut powers = vec![0f32; frames * BINS];
    for frame in 0..frames {
        let start = (frame * super::HOP_LENGTH) as isize - (super::N_FFT / 2) as isize;
        for (idx, value) in input.iter_mut().enumerate() {
            let pos = start + idx as isize;
            let pos = if pos < 0 {
                -pos
            } else if pos >= padded_len as isize {
                2 * padded_len as isize - 2 - pos
            } else {
                pos
            } as usize;
            *value = Complex32::new(samples[pos] * window[idx], 0.);
        }
        fft.process_with_scratch(&mut input, &mut scratch);
        for (power, value) in powers[frame * BINS..(frame + 1) * BINS]
            .iter_mut()
            .zip(&input)
        {
            *power = value.norm_sqr();
        }
    }
    // Batched native GEMM avoids a scalar dot product for every mel/frame.
    // The frontend runs on CPU; only the final features move to the model.
    let powers = Tensor::from_vec(powers, (frames, BINS), &Device::Cpu)?.t()?;
    let filters = Tensor::from_slice(filters, (super::N_MELS, BINS), &Device::Cpu)?;
    let mel = (filters
        .matmul(&powers)?
        .clamp(1e-10f32, f32::INFINITY)?
        .log()?
        / std::f64::consts::LN_10)?;
    let max_log = mel.max_all()?.to_scalar::<f32>()?;
    let mel = ((mel.clamp(max_log - 8., f32::INFINITY)? + 4.)? / 4.)?;
    // Reshape (mel, chunks, time) then transpose to the encoder's
    // (chunks, mel, time), preserving the order of contiguous audio chunks.
    mel.reshape((super::N_MELS, chunks, CHUNK_FRAMES))?
        .transpose(0, 1)?
        .contiguous()?
        .to_device(device)
}
