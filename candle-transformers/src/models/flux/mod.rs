//! Flux  Model
//!
//! Flux is a series of text-to-image generation models based on diffusion transformers.
//!
//! - [GH Link](https://github.com/black-forest-labs/flux)
//! - Transformers Python [reference implementation](https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py)
//!
use candle::{Result, Tensor};

pub trait WithForward {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub mod autoencoder;
pub mod model;
pub mod quantized_model;
pub mod sampling;
