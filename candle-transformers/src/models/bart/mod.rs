//! BART/MBart decoder implementation.
//!
//! BART is a denoising autoencoder for pretraining sequence-to-sequence models.
//! This implementation focuses on the decoder component, suitable for use with
//! HuggingFace VisionEncoderDecoderModel checkpoints including:
//! - Donut (document understanding)
//! - TrOCR (OCR)
//! - ViT-BART and similar vision-language models
//!
//! Key characteristics:
//! - MBart-style decoder with cross-attention to encoder
//! - Learned positional embeddings with offset
//! - KV-cache for efficient incremental decoding
//! - Layer normalization after embeddings (MBart)
//!
//! References:
//! - [BART Paper](https://arxiv.org/abs/1910.13461)
//! - [VisionEncoderDecoderModel](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)

pub mod attention;
pub mod beam_search;
pub mod causal_lm;
pub mod config;
pub mod decode;
pub mod encode;
pub mod generation;
pub mod model;

// Re-export commonly used types for convenience
pub use beam_search::{
    batched_beam_search, beam_search, BatchedBeamSearchConfig, BatchedKVCache, BeamSearchConfig,
    EarlyStoppingMode,
};
pub use causal_lm::BartForCausalLM;
pub use config::{BartConfig, BartWeightPrefix, LayerNormOrder};
pub use decode::{BartDecoder, BartDecoderLayer};
pub use encode::{BartEncoder, BartEncoderLayer};
pub use generation::BartForConditionalGeneration;
