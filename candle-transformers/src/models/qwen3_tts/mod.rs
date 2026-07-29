//! Qwen3-TTS: autoregressive text-to-speech with 12 Hz Mimi codec
//!
//! This module implements the full Qwen3-TTS pipeline from
//! [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B).
//!
//! ## Architecture overview
//!
//! | Component | Role |
//! |-----------|------|
//! | [`TalkerModel`] | Generates semantic codec tokens autoregressively from text |
//! | [`CodePredictor`] | Generates 15 acoustic tokens per frame (residual VQ groups 2-16) |
//! | [`Decoder12Hz`] | Decodes 16-codebook codec frames to 24 kHz audio |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use candle_transformers::models::qwen3_tts::{
//!     TalkerModel, TalkerConfig, CodePredictor, CodePredictorConfig, Decoder12HzConfig,
//! };
//! ```

pub mod codec;
pub mod config;
pub mod kv_cache;
pub mod talker;
pub mod code_predictor;

pub use codec::{Decoder12Hz, Decoder12HzConfig};
pub use config::{ModelType, ParsedModelConfig, Qwen3TTSConfig};
pub use kv_cache::{AnyKVCache, KVCache};
pub use talker::{Language, Speaker, TalkerConfig, TalkerModel};
pub use code_predictor::{CodePredictor, CodePredictorConfig};

use candle::{DType, Device, Result, Tensor};

/// Create a causal attention mask.
///
/// Returns a `[1, 1, seq_len, offset + seq_len]` tensor where position `(i, j)`
/// is `0.0` if `j <= offset + i` (allowed) and `NEG_INFINITY` otherwise.
pub fn create_causal_mask(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
    let total_len = offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j <= offset + i {
                    0.0_f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::new(mask.as_slice(), device)?.reshape((1, 1, seq_len, total_len))
}

/// Convert a slice of codec frames `[Vec<u32>; num_frames]` into a
/// `[1, 16, num_frames]` i64 tensor suitable for the decoder.
pub fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Tensor::zeros((1, 16, 0), DType::I64, device);
    }
    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }
    Tensor::from_vec(data, (1, 16, num_frames), device)
}

/// Return the recommended compute dtype for the given device.
pub fn compute_dtype_for_device(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}
