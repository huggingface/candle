use candle::DType;
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    Absolute,
    Alibi,
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/persimmon/configuration_persimmon.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub qk_layernorm: bool,
    pub partial_rotary_factor: f64,
}

impl Config {
    pub fn base_8b() -> Self {
        // https://huggingface.co/adept/persimmon-8b-base/blob/main/config.json
        Self {
            hidden_act: candle_nn::Activation::Relu,
            hidden_size: 4096,
            initializer_range: 0.02,
            intermediate_size: 16384,
            layer_norm_eps: 1e-05,
            max_position_embeddings: 16384,
            num_attention_heads: 64,
            num_hidden_layers: 36,
            num_key_value_heads: 64,
            qk_layernorm: true,
            rms_norm_eps: 1e-06,
            rope_theta: 25000.0,
            tie_word_embeddings: false,
            use_cache: true,
            vocab_size: 262144,
            partial_rotary_factor: 0.5,
        }
    }
}
