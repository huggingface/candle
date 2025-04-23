use crate::models::{gemma3, siglip};

#[derive(Debug, Clone, serde::Deserialize)]
pub enum Gemma3Config {
    #[serde(untagged)]
    WithVision {
        text_config: gemma3::Config,
        vision_config: siglip::VisionConfig,
        image_token_index: usize,
        mm_tokens_per_image: usize,
    },
    #[serde(untagged)]
    Text(gemma3::Config),
}
