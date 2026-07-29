//! Audio codec sub-modules for Qwen3-TTS.

mod causal_conv;
mod causal_trans_conv;
mod convnext_block;
mod decoder_12hz;
mod decoder_block;
mod snake_beta;

pub use causal_conv::CausalConv1d;
pub use causal_trans_conv::CausalTransConv1d;
pub use convnext_block::ConvNeXtBlock;
pub use decoder_12hz::{Decoder12Hz, Decoder12HzConfig};
pub use decoder_block::{DecoderBlock, ResidualUnit};
pub use snake_beta::SnakeBeta;
