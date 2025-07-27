use candle::{DType, Device, Error, Tensor};

use crate::models::whisper::audio::{log_mel_spectrogram_, Float};

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

/// Process audio using exact WhisperFeatureExtractor algorithm then apply VoxtralProcessor chunking
pub fn extract_features(audio: &[f32], filters: &[f32], device: &Device) -> Result<Tensor, Error> {
    const N_MELS: usize = super::N_MELS;

    // Use the exact WhisperFeatureExtractor algorithm
    // Use the whisper implementation from the parent module
    let mel_vec = pcm_to_mel(audio, filters);

    // The whisper implementation returns Vec<f32> in shape (n_mel * n_len)
    // We need to reshape it to match the expected tensor format
    let n_mel = super::N_MELS;
    let n_len = mel_vec.len() / n_mel;

    // Create tensor with shape (n_mel, n_len) then add batch dimension
    let mel_tensor = Tensor::from_vec(mel_vec, (n_mel, n_len), device)?;
    let mel_tensor = mel_tensor.unsqueeze(0)?; // Add batch dimension -> (1, n_mel, n_len)

    // Convert tensor back to Vec<f32> for compatibility with existing code
    let mel = mel_tensor.flatten_all()?.to_vec1::<f32>()?;
    let mel_len = mel.len();

    // Apply VoxtralProcessor chunking exactly like Python
    let total_frames = mel_len / N_MELS;
    let max_source_positions = 3000; // From VoxtralProcessor defaults

    // Python approach: reshape (feature_size, total_frames) -> (feature_size, -1, max_source_positions)
    // First, create mel tensor with shape (N_MELS, total_frames)
    let mel_tensor = Tensor::from_vec(mel, (N_MELS, total_frames), device)
        .map_err(|e| Error::Msg(format!("Failed to create mel tensor: {}", e).into()))?;

    // Calculate number of chunks (equivalent to Python's -1 dimension in reshape)
    let num_chunks = (total_frames + max_source_positions - 1) / max_source_positions;

    // Pad the mel tensor to be divisible by max_source_positions
    let padded_frames = num_chunks * max_source_positions;
    let padding_needed = padded_frames - total_frames;

    let mel_padded = if padding_needed > 0 {
        let padding = Tensor::zeros((N_MELS, padding_needed), DType::F32, device)?;
        Tensor::cat(&[&mel_tensor, &padding], 1)?
    } else {
        mel_tensor
    };

    // Reshape to (N_MELS, num_chunks, max_source_positions)
    let reshaped = mel_padded.reshape((N_MELS, num_chunks, max_source_positions))?;

    // Transpose to (num_chunks, N_MELS, max_source_positions) - matching Python's transpose(0,1)
    let audio_features = reshaped.transpose(0, 1)?;

    Ok(audio_features)
}
