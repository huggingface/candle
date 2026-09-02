//! SmolLM model family implementations.
//!
//! The SmolLM family consists of efficient language models developed by HuggingFace:
//! - **SmolLM2** (135M, 360M, 1.7B): Uses standard Llama architecture (see `models::llama`)
//! - **SmolLM3** (3B): Introduces hybrid RoPE/NoPE architecture (implemented here)
//!
//! # SmolLM3 Architecture
//!
//! SmolLM3-3B introduces NoPE (No Positional Encoding) as a key innovation:
//! - 3:1 RoPE/NoPE ratio: every 4th layer skips positional encoding
//! - Grouped Query Attention: 32 attention heads, 8 KV heads (4 groups)
//! - High RoPE theta: 5,000,000 (vs typical 10,000-500,000)
//! - Extended context: 64k-128k tokens
//!
//! # Module Structure
//!
//! - [`smollm3`]: Full precision model implementation (safetensors)
//! - [`quantized_smollm3`]: Quantized model implementation (GGUF)
//!
//! # Example Usage
//!
//! ```ignore
//! use candle_transformers::models::smol::smollm3::{Config, ModelForCausalLM};
//! use candle_transformers::models::smol::quantized_smollm3::QuantizedModelForCausalLM;
//! use candle::{Device, Tensor};
//! use candle_nn::VarBuilder;
//!
//! # fn main() -> anyhow::Result<()> {
//! let device = Device::Cpu;
//!
//! // Load full precision model
//! let vb = VarBuilder::zeros(candle::DType::F32, &device);
//! let config = Config::default();
//! let model = ModelForCausalLM::new(&config, vb)?;
//!
//! // Or load quantized model
//! // let model = QuantizedModelForCausalLM::from_gguf(path, &device)?;
//!
//! // Run inference
//! let input = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;
//! let logits = model.forward(&input, 0)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Thinking Mode
//!
//! SmolLM3 supports explicit reasoning via thinking tags in chat templates:
//! - Thinking enabled: `<|im_start|>assistant\n<think>\n` (model generates reasoning)
//! - Thinking disabled: `<|im_start|>assistant\n<think>\n\n</think>\n` (skip to answer)
//!
//! # Performance Considerations
//!
//! | Format | Size  | Inference Speed | Quality |
//! |--------|-------|-----------------|---------|
//! | Q4_K_M | 1.9GB | Fastest         | Good    |
//! | Q8_0   | 3.3GB | Fast            | Better  |
//! | F16    | 6.2GB | Medium          | Best    |
//! | F32    | 12GB  | Slow            | Best    |
//!
//! # References
//!
//! - [SmolLM3 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
//! - [NoPE Paper](https://arxiv.org/abs/2410.01926)

pub mod quantized_smollm3;
pub mod smollm3;
