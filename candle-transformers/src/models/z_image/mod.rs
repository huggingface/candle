//! Z-Image Model
//!
//! Z-Image is a text-to-image generation model from Alibaba using Flow Matching.
//!
//! - 🤗 [Hugging Face Model](https://huggingface.co/Z-a-o/Z-Image-Turbo)
//!
//! # Example
//!
//! ```bash
//! cargo run --features metal --example z_image --release -- \
//!     --prompt "A beautiful landscape" --height 1024 --width 1024
//! ```
//!
//! # Architecture
//!
//! - Transformer: ~24B parameters, 30 main layers + 2 noise_refiner + 2 context_refiner
//! - Text Encoder: Qwen3 (hidden_size=2560, 36 layers)
//! - VAE: AutoencoderKL (diffusers format)
//! - Scheduler: FlowMatchEulerDiscreteScheduler (shift=3.0)

pub mod preprocess;
pub mod sampling;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

// Re-export main types
pub use preprocess::{prepare_inputs, PreparedInputs};
pub use sampling::{get_noise, get_schedule, postprocess_image};
pub use scheduler::{calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig};
pub use text_encoder::{TextEncoderConfig, ZImageTextEncoder};
pub use transformer::{Config, ZImageTransformer2DModel};
pub use vae::{AutoEncoderKL, VaeConfig};
