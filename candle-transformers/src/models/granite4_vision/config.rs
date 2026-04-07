//! Configuration for Granite 4.0 3B Vision model.
//!
//! Architecture: SigLIP2 vision encoder → Window Q-Former projectors → Deepstack injection
//! into GraniteMoeHybrid language model.
//!
//! References:
//! - [Model Card](https://huggingface.co/ibm-granite/granite-4.0-3b-vision)

use crate::models::granitemoehybrid::GraniteMoeHybridConfig;
use crate::models::siglip;

fn default_spatial_stride() -> usize {
    2
}

fn default_spatial_vision_layer() -> i32 {
    -1
}

fn default_projector_dropout() -> f32 {
    0.1
}

#[derive(serde::Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FeatureSelectStrategy {
    Full,
    Default,
}

fn default_vision_feature_select_strategy() -> FeatureSelectStrategy {
    FeatureSelectStrategy::Full
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub vision_config: siglip::VisionConfig,
    pub text_config: GraniteMoeHybridConfig,

    /// List of [vision_layer_idx, llm_layer_idx] pairs for deepstack injection.
    /// Vision layer indices are negative (from end of vision encoder).
    pub deepstack_layer_map: Vec<[i32; 2]>,

    /// Downsample rate as a fraction string, e.g. "4/8".
    /// Numerator = query_side, denominator = window_side for the Q-Former.
    pub downsample_rate: String,

    /// LLM layers where spatial features are injected.
    #[serde(default)]
    pub spatial_target_layers: Vec<usize>,

    #[serde(default = "default_spatial_stride")]
    pub spatial_stride: usize,

    #[serde(default = "default_spatial_vision_layer")]
    pub spatial_vision_layer: i32,

    #[serde(default)]
    pub use_spatial_sampling: bool,

    #[serde(default)]
    pub use_image_newline_parameter: bool,

    #[serde(default = "default_projector_dropout")]
    pub projector_dropout: f32,

    #[serde(default = "default_vision_feature_select_strategy")]
    pub vision_feature_select_strategy: FeatureSelectStrategy,

    #[serde(default)]
    pub image_token_index: u32,

    #[serde(default)]
    pub image_grid_pinpoints: Vec<[usize; 2]>,

    #[serde(default)]
    pub image_seq_length: usize,
}

impl Config {
    /// Parse downsample_rate "N/M" into (query_side, window_side).
    pub fn query_and_window_side(&self) -> (usize, usize) {
        let parts: Vec<&str> = self.downsample_rate.split('/').collect();
        let query_side: usize = parts[0].parse().expect("invalid downsample_rate numerator");
        let window_side: usize = parts[1].parse().expect("invalid downsample_rate denominator");
        (query_side, window_side)
    }

    /// Number of patches per side of a single tile.
    pub fn patches_per_side(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    /// Downsampled patches per side after area interpolation.
    pub fn downsampled_patches_per_side(&self) -> usize {
        let (q, w) = self.query_and_window_side();
        self.patches_per_side() * q / w
    }

    /// Resolve a negative vision layer index to a positive index into hidden_states vec.
    /// hidden_states has len = num_hidden_layers + 1 (embedding + N layer outputs).
    pub fn resolve_vision_layer(&self, idx: i32) -> usize {
        let total = self.vision_config.num_hidden_layers + 1;
        if idx < 0 {
            (total as i32 + idx) as usize
        } else {
            idx as usize
        }
    }
}
