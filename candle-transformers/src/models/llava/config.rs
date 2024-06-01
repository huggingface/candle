use crate::models::llama::Config;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLaVAConfig {
    pub _name_or_path: String,
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub freeze_mm_mlp_adapter: bool,
    pub freeze_mm_vision_resampler: bool,
    pub hidden_act: String,
    pub hidden_size: usize,
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
    pub image_crop_resolution: usize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_split_resolution: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mm_hidden_size: usize,
    #[serde(default = "default_mm_patch_merge_type")]
    pub mm_patch_merge_type: String,
    pub mm_projector_lr: Option<f32>,
    pub mm_projector_type: String,
    pub mm_resampler_type: Option<String>,
    pub mm_use_im_patch_token: bool,
    pub mm_use_im_start_end: bool,
    pub mm_vision_select_feature: String,
    pub mm_vision_select_layer: isize,
    pub mm_vision_tower: String,
    pub mm_vision_tower_lr: f32,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<f32>,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub tokenizer_model_max_length: Option<usize>,
    pub tokenizer_padding_side: String,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub tune_mm_mlp_adapter: bool,
    pub tune_mm_vision_resampler: bool,
    pub unfreeze_mm_vision_tower: bool,
    pub use_cache: bool,
    pub use_mm_proj: bool,
    pub vocab_size: usize,
}

fn default_mm_patch_merge_type() -> String {
    "flat".to_string()
}

fn default_image_aspect_ratio() -> String {
    "square".to_string()
}
impl LLaVAConfig {
    pub fn to_llama_config(&self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            use_flash_attn: false,
        }
    }
}
