use candle_nn::Activation;

use crate::serde_default_fn;

serde_default_fn!(Activation, default_vision_hidden_act, Activation::Gelu);
serde_default_fn!(usize, default_in_channels, 3);
serde_default_fn!(usize, default_depth, 32);
serde_default_fn!(usize, default_hidden_size, 3584);
serde_default_fn!(usize, default_out_hidden_size, 3584);
serde_default_fn!(usize, default_intermediate_size, 3420);
serde_default_fn!(usize, default_num_heads, 16);
serde_default_fn!(usize, default_patch_size, 14);
serde_default_fn!(usize, default_spatial_merge_size, 2);
serde_default_fn!(usize, default_temporal_patch_size, 2);
serde_default_fn!(usize, default_num_position_embeddings, 576);
serde_default_fn!(Vec<usize>, default_deepstack_visual_indexes, Vec::new());

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_chans: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_deepstack_visual_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}
