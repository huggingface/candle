use crate::models::lfm2::Lfm2Config as TextConfig;

use candle_nn::Activation;

use crate::serde_default_fn;

serde_default_fn!(usize, default_vision_hidden_size, 768);
serde_default_fn!(usize, default_vision_intermediate_size, 3072);
serde_default_fn!(usize, default_vision_num_hidden_layers, 12);
serde_default_fn!(usize, default_vision_num_attention_heads, 12);
serde_default_fn!(usize, default_vision_num_channels, 3);
serde_default_fn!(usize, default_vision_patch_size, 16);
serde_default_fn!(usize, default_vision_num_patches, 256);
serde_default_fn!(
    Activation,
    default_vision_hidden_act,
    Activation::GeluPytorchTanh
);
serde_default_fn!(f64, default_vision_layer_norm_eps, 1e-6);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Siglip2VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,

    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,

    #[serde(default = "default_vision_num_patches")]
    pub num_patches: usize,

    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

impl Siglip2VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Side of the learned square position grid (16 for 256 positions).
    pub fn position_embedding_size(&self) -> usize {
        self.num_patches.isqrt()
    }
}

serde_default_fn!(u32, default_image_token, 396);
serde_default_fn!(Activation, default_projector_hidden_act, Activation::Gelu);

serde_default_fn!(usize, default_projector_hidden_size, 2560);
serde_default_fn!(bool, default_projector_bias, true);
serde_default_fn!(bool, default_projector_use_layernorm, true);
serde_default_fn!(usize, default_downsample_factor, 2);
serde_default_fn!(bool, default_tie_word_embeddings, true);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2VlConfig {
    pub model_type: Option<String>,
    pub vision_config: Siglip2VisionConfig,
    pub text_config: TextConfig,

    #[serde(default = "default_image_token")]
    pub image_token_id: u32,

    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: Activation,

    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,

    #[serde(default = "default_projector_bias")]
    pub projector_bias: bool,

    #[serde(default = "default_projector_use_layernorm")]
    pub projector_use_layernorm: bool,

    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}
