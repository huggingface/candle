//! Configuration for Mistral3 Vision-Language Model

use crate::models::mistral::Config as MistralConfig;
use crate::models::pixtral::vision_model::Config as PixtralVisionConfig;
use serde::Deserialize;

fn default_image_token_index() -> usize {
    10
}

fn default_projector_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::Gelu
}

fn default_vision_feature_layer() -> i32 {
    -1
}

fn default_spatial_merge_size() -> usize {
    2
}

/// Configuration for Mistral3 multimodal model.
///
/// This configuration combines:
/// - `vision_config`: Pixtral vision encoder configuration
/// - `text_config`: Mistral language model configuration
/// - Multimodal-specific parameters for the projector and patch merger
#[derive(Debug, Clone, Deserialize)]
pub struct Mistral3Config {
    /// Vision encoder configuration, reuses pixtral::vision_model::Config.
    /// Extra fields in JSON (like attention_dropout) are automatically ignored by serde.
    pub vision_config: PixtralVisionConfig,

    /// Language model configuration, reuses mistral::Config.
    /// Extra fields in JSON (like attention_dropout, use_cache) are automatically ignored by serde.
    pub text_config: MistralConfig,

    /// Token ID for image placeholder in the input sequence.
    /// Default: 10
    #[serde(default = "default_image_token_index")]
    pub image_token_index: usize,

    /// Activation function for the multimodal projector MLP.
    /// Default: GELU
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: candle_nn::Activation,

    /// Which vision encoder layer to use for features.
    /// Negative values index from the end (-1 = last layer).
    /// Default: -1
    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: i32,

    /// Whether to use bias in the multimodal projector linear layers.
    /// Default: false
    #[serde(default)]
    pub multimodal_projector_bias: bool,

    /// Spatial merge size for PatchMerger.
    /// Merges spatial_merge_size² patches into one, reducing image tokens by this factor squared.
    /// Default: 2 (reduces tokens by 4x)
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
}

impl Mistral3Config {
    /// Get the actual layer index for vision_feature_layer.
    /// Handles negative indexing (e.g., -1 means last layer).
    pub fn get_vision_feature_layer_index(&self, num_layers: usize) -> usize {
        if self.vision_feature_layer < 0 {
            (num_layers as i32 + self.vision_feature_layer) as usize
        } else {
            self.vision_feature_layer as usize
        }
    }
}
