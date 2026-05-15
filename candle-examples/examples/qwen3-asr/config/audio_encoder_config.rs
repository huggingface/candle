//! Audio encoder config (mirrors `Qwen3ASRAudioEncoderConfig`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEncoderConfig {
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,

    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,

    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,

    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,

    #[serde(default = "default_d_model")]
    pub d_model: usize,

    #[serde(default)]
    pub dropout: f64,

    #[serde(default)]
    pub attention_dropout: f64,

    #[serde(default = "default_activation_function")]
    pub activation_function: String,

    #[serde(default)]
    pub activation_dropout: f64,

    #[serde(default)]
    pub scale_embedding: bool,

    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,

    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,

    #[serde(default = "default_n_window")]
    pub n_window: usize,

    #[serde(default = "default_output_dim")]
    pub output_dim: usize,

    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,

    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,

    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: usize,
}

fn default_num_mel_bins() -> usize {
    128
}

fn default_encoder_layers() -> usize {
    32
}

fn default_encoder_attention_heads() -> usize {
    20
}

fn default_encoder_ffn_dim() -> usize {
    5120
}

fn default_d_model() -> usize {
    1280
}

fn default_activation_function() -> String {
    "gelu".to_string()
}

fn default_initializer_range() -> f64 {
    0.02
}

fn default_max_source_positions() -> usize {
    1500
}

fn default_n_window() -> usize {
    100
}

fn default_output_dim() -> usize {
    3584
}

fn default_n_window_infer() -> usize {
    400
}

fn default_conv_chunksize() -> usize {
    500
}

fn default_downsample_hidden_size() -> usize {
    480
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            num_mel_bins: default_num_mel_bins(),
            encoder_layers: default_encoder_layers(),
            encoder_attention_heads: default_encoder_attention_heads(),
            encoder_ffn_dim: default_encoder_ffn_dim(),
            d_model: default_d_model(),
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: default_activation_function(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: default_initializer_range(),
            max_source_positions: default_max_source_positions(),
            n_window: default_n_window(),
            output_dim: default_output_dim(),
            n_window_infer: default_n_window_infer(),
            conv_chunksize: default_conv_chunksize(),
            downsample_hidden_size: default_downsample_hidden_size(),
        }
    }
}
