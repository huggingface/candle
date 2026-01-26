pub mod audio;
pub mod model;
pub mod voxtral_llama;

pub use audio::extract_features;
pub use model::{
    VoxtralCache, VoxtralConfig, VoxtralEncoder, VoxtralEncoderConfig,
    VoxtralForConditionalGeneration, VoxtralGenerationConfig, VoxtralMultiModalProjector,
};
pub use voxtral_llama::{VoxtralLlama, VoxtralLlamaCache, VoxtralLlamaConfig};

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
