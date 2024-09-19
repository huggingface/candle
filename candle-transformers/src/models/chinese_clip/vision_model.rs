use super::Activation;

#[derive(Clone, Debug)]
pub struct ChineseClipVisionConfig {
    // hidden_size=768,
    //     intermediate_size=3072,
    //     projection_dim=512,
    //     num_hidden_layers=12,
    //     num_attention_heads=12,
    //     num_channels=3,
    //     image_size=224,
    //     patch_size=32,
    //     hidden_act="quick_gelu",
    //     layer_norm_eps=1e-5,
    //     attention_dropout=0.0,
    //     initializer_range=0.02,
    //     initializer_factor=1.0,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f32,
    pub attention_dropout: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
}

impl Default for ChineseClipVisionConfig {
    fn default() -> Self {
        ChineseClipVisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            hidden_act: Activation::QuickGelu,
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}

impl ChineseClipVisionConfig {
    /// referer: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json
    pub fn clip_vit_base_patch16() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
            hidden_act: Activation::QuickGelu,
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}
