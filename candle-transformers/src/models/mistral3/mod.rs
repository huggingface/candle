//! Mistral3 Vision-Language Model
//!
//! Mistral3 is a multimodal model combining Pixtral vision encoder with Mistral language model.
//!
//! - 💻 [HuggingFace](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
//! - 📝 [Transformers Reference](https://github.com/huggingface/transformers/tree/main/src/transformers/models/mistral3)
//!
//! # Architecture
//!
//! The model consists of:
//! - Vision Tower: Pixtral vision encoder (reused from `crate::models::pixtral::vision_model`)
//! - Multi-Modal Projector: RMSNorm + PatchMerger + MLP
//! - Language Model: Mistral (reused from `crate::models::mistral`)
//!
//! # Example
//!
//! ```rust,ignore
//! use candle_transformers::models::mistral3::{Mistral3Config, Mistral3ForConditionalGeneration};
//!
//! let config: Mistral3Config = serde_json::from_str(&config_str)?;
//! let model = Mistral3ForConditionalGeneration::new(&config, vb)?;
//! let logits = model.forward(&input_ids, Some(&pixel_values), Some(&image_sizes), 0)?;
//! ```

mod config;
mod model;
mod patch_merger;
mod projector;

pub use config::Mistral3Config;
pub use model::{
    find_image_token_positions, replace_image_tokens, Mistral3ForConditionalGeneration,
    Mistral3Model,
};
pub use patch_merger::PatchMerger;
pub use projector::MultiModalProjector;
