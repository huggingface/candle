//! CosyVoice3 Frontend ONNX Models
//!
//! Provides speech tokenization and speaker embedding extraction using ONNX models.
//!
//! # Components
//! - `CosyVoice3Frontend`: Main frontend for extracting prompt features from audio
//! - Speech tokenizer (speech_tokenizer_v3.onnx)
//! - Speaker embedding extractor (campplus.onnx)
//!
//! # Usage
//! ```ignore
//! use candle_transformers::models::cosyvoice::frontend::CosyVoice3Frontend;
//!
//! let frontend = CosyVoice3Frontend::load("model_dir", &device)?;
//! let (tokens, mel, embedding) = frontend.extract_prompt_features(&audio, sample_rate)?;
//! ```
//!
//! # Feature Flag
//! This module requires the `onnx` feature to be enabled.

#[cfg(feature = "onnx")]
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::path::Path;

/// CosyVoice3 Frontend - Extract prompt features from audio
///
/// This struct provides methods to extract:
/// - Speech tokens using speech_tokenizer_v3.onnx
/// - Speaker embeddings using campplus.onnx
/// - Mel spectrograms for the flow decoder
#[cfg(feature = "onnx")]
pub struct CosyVoice3Frontend {
    /// Speech tokenizer ONNX model
    speech_tokenizer: candle_onnx::onnx::ModelProto,
    /// CAMPPlus speaker embedding ONNX model
    campplus: candle_onnx::onnx::ModelProto,
    /// Whisper-style mel extractor (for speech tokenizer input)
    whisper_mel: super::audio::MelSpectrogram,
    /// Kaldi fbank extractor (for campplus input)
    kaldi_fbank: super::audio::KaldiFbank,
    /// CosyVoice speech feat extractor (for flow decoder)
    speech_feat_mel: super::audio::MelSpectrogram,
    /// Target device
    device: Device,
}

#[cfg(feature = "onnx")]
impl CosyVoice3Frontend {
    /// Load Frontend from model directory
    ///
    /// # Arguments
    /// * `model_dir` - Path to directory containing ONNX models
    /// * `device` - Target device for computations
    ///
    /// # Required Files
    /// - `speech_tokenizer_v3.onnx` - Speech tokenizer model
    /// - `campplus.onnx` - Speaker embedding model
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load ONNX models
        let speech_tokenizer_path = model_dir.join("speech_tokenizer_v3.onnx");
        let campplus_path = model_dir.join("campplus.onnx");

        let speech_tokenizer = candle_onnx::read_file(&speech_tokenizer_path).map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to load speech_tokenizer_v3.onnx from {:?}: {}",
                speech_tokenizer_path, e
            ))
        })?;

        let campplus = candle_onnx::read_file(&campplus_path).map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to load campplus.onnx from {:?}: {}",
                campplus_path, e
            ))
        })?;

        // Create audio processors
        let whisper_mel = super::audio::MelSpectrogram::new_whisper_format(device)?;
        let kaldi_fbank = super::audio::KaldiFbank::new_for_campplus();
        let speech_feat_mel = super::audio::MelSpectrogram::new_cosyvoice_speech_feat(device)?;

        Ok(Self {
            speech_tokenizer,
            campplus,
            whisper_mel,
            kaldi_fbank,
            speech_feat_mel,
            device: device.clone(),
        })
    }

    /// Extract speech tokens from audio
    ///
    /// # Arguments
    /// * `audio_16k` - 16kHz sampled audio waveform [samples]
    ///
    /// # Returns
    /// * `speech_tokens` - Speech token sequence [1, T]
    pub fn extract_speech_tokens(&self, audio_16k: &Tensor) -> Result<Tensor> {
        // 1. Extract whisper-style mel spectrogram using whisper-compatible method
        //    This matches whisper.log_mel_spectrogram() exactly:
        //    - Uses center padding (like torch.stft)
        //    - Drops the last frame (magnitudes[..., :-1])
        //    - Uses librosa-compatible mel filters
        //    Input: [samples] @ 16kHz
        //    Output: [1, 128, T] (n_mels=128)
        let mel = self.whisper_mel.forward_whisper(audio_16k)?;

        // Apply whisper-style log normalization
        let mel = Self::whisper_log_normalize(&mel)?;

        // 2. Prepare ONNX inputs
        // Speech tokenizer expects:
        //   feats: [1, 128, T] float32
        //   feats_length: [1] int32
        let mel_len = mel.dim(2)? as i32;
        let mel_len_tensor = Tensor::from_slice(&[mel_len], (1,), &Device::Cpu)?;

        // Get input names from model
        let graph = self
            .speech_tokenizer
            .graph
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("No graph in speech_tokenizer model".into()))?;

        let input_names: Vec<&str> = graph.input.iter().map(|i| i.name.as_str()).collect();

        let mut inputs = HashMap::new();
        // First input is mel spectrogram (feats)
        if !input_names.is_empty() {
            inputs.insert(
                input_names[0].to_string(),
                mel.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            );
        }
        // Second input is mel length (feats_length)
        if input_names.len() > 1 {
            inputs.insert(
                input_names[1].to_string(),
                mel_len_tensor.to_dtype(DType::I64)?,
            );
        }

        // 3. Run speech_tokenizer ONNX model
        let outputs = candle_onnx::simple_eval(&self.speech_tokenizer, inputs)?;

        // 4. Extract output tokens (first output)
        let output_names: Vec<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();
        let output_name = output_names
            .first()
            .ok_or_else(|| candle::Error::Msg("No output in speech_tokenizer model".into()))?;

        let tokens = outputs
            .get(*output_name)
            .ok_or_else(|| {
                candle::Error::Msg(format!("Missing output '{}' from speech_tokenizer", output_name))
            })?
            .clone();

        // Ensure shape is [1, T]
        let tokens = if tokens.rank() == 1 {
            tokens.unsqueeze(0)?
        } else {
            tokens
        };

        tokens.to_device(&self.device)
    }

    /// Extract speaker embedding from audio
    ///
    /// # Arguments
    /// * `audio_16k` - 16kHz sampled audio waveform [samples]
    ///
    /// # Returns
    /// * `embedding` - Speaker embedding [1, 192]
    pub fn extract_speaker_embedding(&self, audio_16k: &Tensor) -> Result<Tensor> {
        // 1. Extract Kaldi fbank features
        //    Input: [samples] @ 16kHz
        //    Output: [T, 80] (num_mel_bins=80)
        let fbank = self.kaldi_fbank.forward(audio_16k)?;

        // 2. Mean normalization (subtract mean along time axis)
        //    Note: candle-onnx now supports variable-length input after fixing
        //    the AveragePool ceil_mode issue. No frame limit needed.
        let mean = fbank.mean(0)?;
        let fbank_normalized = fbank.broadcast_sub(&mean)?;

        // 3. Add batch dimension: [T, 80] -> [1, T, 80]
        let fbank_batched = fbank_normalized.unsqueeze(0)?;

        // 4. Prepare ONNX inputs
        let graph = self
            .campplus
            .graph
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("No graph in campplus model".into()))?;

        let input_names: Vec<&str> = graph.input.iter().map(|i| i.name.as_str()).collect();

        let mut inputs = HashMap::new();
        if !input_names.is_empty() {
            inputs.insert(
                input_names[0].to_string(),
                fbank_batched.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            );
        }

        // 5. Run campplus ONNX model
        let outputs = candle_onnx::simple_eval(&self.campplus, inputs)?;

        // 6. Extract output embedding (first output)
        let output_names: Vec<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();
        let output_name = output_names
            .first()
            .ok_or_else(|| candle::Error::Msg("No output in campplus model".into()))?;

        let embedding = outputs
            .get(*output_name)
            .ok_or_else(|| {
                candle::Error::Msg(format!("Missing output '{}' from campplus", output_name))
            })?
            .clone();

        // Ensure shape is [1, 192]
        let embedding = if embedding.rank() == 1 {
            embedding.unsqueeze(0)?
        } else {
            embedding
        };

        embedding.to_device(&self.device)
    }

    /// Extract speech feat (mel spectrogram for flow decoder)
    ///
    /// Uses Matcha-TTS style reflect padding to match Python implementation.
    ///
    /// # Arguments
    /// * `audio_24k` - 24kHz sampled audio waveform [samples]
    ///
    /// # Returns
    /// * `speech_feat` - Mel spectrogram [1, T, 80]
    pub fn extract_speech_feat(&self, audio_24k: &Tensor) -> Result<Tensor> {
        // 1. Extract mel spectrogram with Matcha-TTS style padding
        //    Matcha adds (n_fft - hop_size) / 2 reflect padding on each side
        //    This ensures frame count matches Python exactly
        //    Input: [samples] @ 24kHz
        //    Output: [1, 80, T]
        let mel = self.speech_feat_mel.forward_cosyvoice(audio_24k)?;

        // 2. Transpose to [1, T, 80] format
        let mel = mel.transpose(1, 2)?;

        mel.to_device(&self.device)
    }

    /// Extract complete prompt features from audio
    ///
    /// # Arguments
    /// * `audio` - Raw audio waveform [samples]
    /// * `sample_rate` - Audio sample rate
    ///
    /// # Returns
    /// * `(speech_tokens, speech_feat, speaker_embedding)`
    ///   - speech_tokens: [1, T_tokens] i64
    ///   - speech_feat: [1, T_mel, 80] f32
    ///   - speaker_embedding: [1, 192] f32
    pub fn extract_prompt_features(
        &self,
        audio: &Tensor,
        sample_rate: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // 1. Resample to 16kHz (for speech tokenizer and campplus)
        let audio_16k = if sample_rate != 16000 {
            super::audio::resample(audio, sample_rate, 16000)?
        } else {
            audio.clone()
        };

        // 2. Resample to 24kHz (for speech feat)
        let audio_24k = if sample_rate != 24000 {
            super::audio::resample(audio, sample_rate, 24000)?
        } else {
            audio.clone()
        };

        // 3. Extract features
        let speech_tokens = self.extract_speech_tokens(&audio_16k)?;
        let speaker_embedding = self.extract_speaker_embedding(&audio_16k)?;
        let speech_feat = self.extract_speech_feat(&audio_24k)?;

        // 4. Align speech_feat and speech_tokens (token_mel_ratio = 2)
        //    Ensure speech_feat.len() = speech_tokens.len() * 2
        let token_len = speech_tokens.dim(1)?;
        let feat_len = speech_feat.dim(1)?;
        let aligned_len = token_len.min(feat_len / 2);

        let speech_tokens = speech_tokens.narrow(1, 0, aligned_len)?;
        let speech_feat = speech_feat.narrow(1, 0, aligned_len * 2)?;

        Ok((speech_tokens, speech_feat, speaker_embedding))
    }

    /// Whisper-style log normalization
    ///
    /// Applies the same normalization as whisper.log_mel_spectrogram:
    /// - log10 with clamp
    /// - max - 8.0 floor
    /// - (x + 4.0) / 4.0 scaling
    fn whisper_log_normalize(mel: &Tensor) -> Result<Tensor> {
        // mel is already in log10 scale from MelSpectrogram::forward
        // Apply whisper normalization: max(log_spec, log_spec.max() - 8.0)
        // Then scale: (log_spec + 4.0) / 4.0

        // Get max value across all dimensions
        let max_val = mel.flatten_all()?.max(0)?;

        // Floor at max - 8.0
        let floor = (max_val - 8.0)?;
        let log_spec = mel.broadcast_maximum(&floor)?;

        // Scale: (x + 4.0) / 4.0
        let normalized = ((log_spec + 4.0)? / 4.0)?;

        Ok(normalized)
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use candle::{Device, Result, Tensor};

    #[test]
    fn test_whisper_log_normalize() -> Result<()> {
        let device = Device::Cpu;

        // Create test mel spectrogram
        let mel = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], (1, 1, 5), &device)?;

        let normalized = CosyVoice3Frontend::whisper_log_normalize(&mel)?;

        // Check shape is preserved
        assert_eq!(normalized.dims(), mel.dims());

        // Check values are in reasonable range
        let values: Vec<f32> = normalized.flatten_all()?.to_vec1()?;
        for v in values {
            assert!(v >= 0.0 && v <= 2.0, "Normalized value {} out of range", v);
        }

        Ok(())
    }
}
