//! Flux  Model
//!
//! Flux is a 12B rectified flow transformer capable of generating images from text descriptions.
//!
//! - 🤗 [Hugging Face Model](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
//! - 💻 [GitHub Repository](https://github.com/black-forest-labs/flux)
//! - 📝 [Blog Post](https://blackforestlabs.ai/announcing-black-forest-labs/)
//!
//! # Usage
//!
//! ```bash
//! cargo run --features cuda \
//!     --example flux -r -- \
//!     --height 1024 --width 1024 \
//!     --prompt "a rusty robot walking on a beach holding a small torch, \
//!               the robot has the word \"rust\" written on it, high quality, 4k"
//! ```
//!
//! <div align=center>
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/flux/assets/flux-robot.jpg" alt="" width=320>
//! </div>
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
        kontext_image_ids: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub mod autoencoder;
pub mod model;
pub mod quantized_model;
pub mod sampling;
