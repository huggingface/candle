//! mimi model
//!
//! [Mimi](https://huggingface.co/kyutai/mimi) is a state of the art audio
//! compression model using an encoder/decoder architecture with residual vector
//! quantization. The candle implementation supports streaming meaning that it's
//! possible to encode or decode a stream of audio tokens on the flight to provide
//! low latency interaction with an audio model.
//!
//! - 🤗 [HuggingFace Model Card](https://huggingface.co/kyutai/mimi)
//! - 💻 [GitHub](https://github.com/kyutai-labs/moshi)
//!
//!
//! # Example
//! ```bash
//! # Generating some audio tokens from an audio files.
//! wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
//! cargo run --example mimi \
//!   --features mimi --release -- \
//!   audio-to-code bria.mp3 bria.safetensors
//!
//! # And decoding the audio tokens back into a sound file.
//! cargo run --example mimi
//!   --features mimi --release -- \
//!   code-to-audio bria.safetensors bria.wav
//!

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
pub use candle;
pub use candle_nn;

pub mod conv;
pub mod encodec;
pub mod quantization;
pub mod seanet;
pub mod transformer;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

pub use encodec::{load, Config, Encodec as Model};
