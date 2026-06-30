//! Granite 4.0 3B Vision model config. https://huggingface.co/ibm-granite/granite-4.0-3b-vision

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

    pub deepstack_layer_map: Vec<[i32; 2]>,

    pub downsample_rate: String,

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
    pub fn query_and_window_side(&self) -> (usize, usize) {
        let parts: Vec<&str> = self.downsample_rate.split('/').collect();
        let query_side: usize = parts[0].parse().expect("invalid downsample_rate numerator");
        let window_side: usize = parts[1]
            .parse()
            .expect("invalid downsample_rate denominator");
        (query_side, window_side)
    }

    pub fn patches_per_side(&self) -> usize {
        self.vision_config.image_size / self.vision_config.patch_size
    }

    pub fn downsampled_patches_per_side(&self) -> usize {
        let (q, w) = self.query_and_window_side();
        self.patches_per_side() * q / w
    }

    pub fn resolve_vision_layer(&self, idx: i32) -> usize {
        let total = self.vision_config.num_hidden_layers + 1;
        if idx < 0 {
            (total as i32 + idx) as usize
        } else {
            idx as usize
        }
    }
}
