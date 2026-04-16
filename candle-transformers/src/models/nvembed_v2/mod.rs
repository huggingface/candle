//! NV-Embed-v2
//!
//! NV-Embed-v2 is a text embedding model that combines a Mistral decoder with a latent attention mechanism to produce high-quality text embeddings.
//!
//! This implementation is based on the [paper](https://arxiv.org/pdf/2405.17428) and [weights](https://huggingface.co/nvidia/NV-Embed-v2)
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

pub mod embedding;
pub mod model;
