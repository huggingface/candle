//! NV-Embed-v2
//!
//! NV-Embed-v2 is a text embedding model that combines a Mistral decoder with a latent attention mechanism to produce high-quality text embeddings.
//!
//! - [HuggingFace Model Card](https://huggingface.co/nvidia/NV-Embed-v2)
//!
//! # Query-Passage Retrieval Example
//! ```bash
//! cargo run --example nvembed_v2 --release
//! ```
//!
//! # Sentence Embedding Example
//! ```bash
//! cargo run --example nvembed_v2 --release -- --prompt "Here is a test sentence"
//! ```

// Copyright (c) NVIDIA CORPORATION, all rights reserved.
// This source code is licensed under the CC-BY-NC-4.0 license.
// See https://spdx.org/licenses/CC-BY-NC-4.0 for details.

pub mod decoder;
pub mod model;
