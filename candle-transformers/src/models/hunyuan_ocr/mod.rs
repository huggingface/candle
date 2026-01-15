//! HunyuanOCR Vision-Language Model for OCR.
//!
//! HunyuanOCR combines a Vision Transformer encoder with a Transformer decoder
//! for document OCR tasks including text recognition, table parsing, and more.
//!
//! Key features:
//! - Dynamic resolution support for variable-sized document images
//! - xDRoPE (Extended Dynamic Rotary Position Embedding) for position encoding
//! - Flash Attention / SDPA support for efficient inference
//!
//! References:
//! - [HuggingFace Model](https://huggingface.co/tencent/HunyuanOCR)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;

pub mod config;
mod text;
mod vision;

pub use config::Config;
use text::TextModel;
use vision::VisionModel;

/// HunyuanOCR Model for vision-language OCR tasks.
pub struct HunyuanOCRModel {
    vision: VisionModel,
    text: TextModel,
    cache: text::Cache,
    image_token_id: u32,
    xdrope_x_dim: usize,
    dtype: DType,
    device: Device,
}

impl HunyuanOCRModel {
    /// Create a new HunyuanOCR model.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `use_flash_attn` - Whether to use Flash Attention (CUDA) when available
    /// * `vb` - Variable builder for loading weights
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        // Vision model at "vit" prefix
        let vision = VisionModel::new(&cfg.vision_config, vb.pp("vit"))?;
        // Text model at root level
        let text = TextModel::new(&cfg.text_config, use_flash_attn, vb.clone())?;

        // Initialize cache for KV caching
        let cache = text::Cache::new(text.num_layers());

        // Get xDRoPE x_dim from config (dynamically, not hardcoded)
        let xdrope_x_dim = cfg
            .text_config
            .rope_scaling
            .as_ref()
            .and_then(|s| s.xdrope_section.as_ref())
            .map(|s| s.len())
            .unwrap_or(4); // Default to 4 if not specified

        Ok(Self {
            vision,
            text,
            cache,
            image_token_id: cfg.image_token_id,
            xdrope_x_dim,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Encode image to vision features.
    pub fn encode_image(
        &mut self,
        pixel_values: &Tensor,
        grid_thw: &[(i64, i64, i64)],
    ) -> Result<Tensor> {
        self.vision.forward(pixel_values, grid_thw)
    }

    /// Forward pass for vision-language generation.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        grid_thw: Option<&[(i64, i64, i64)]>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let hidden_dim = input_embeds.dim(2)?;

        // Inject vision embeddings if provided
        if let (Some(pixel_values), Some(grid_thw)) = (pixel_values, grid_thw) {
            let image_embeds = self.encode_image(pixel_values, grid_thw)?;
            let image_embeds = image_embeds.to_dtype(self.dtype)?;

            let input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
            let mut image_offset = 0usize;
            let num_image_tokens = image_embeds.dim(1)?;

            for batch in 0..batch_size {
                for pos in 0..seq_len {
                    let idx = batch * seq_len + pos;
                    if input_ids_vec[idx] == self.image_token_id && image_offset < num_image_tokens
                    {
                        let img_emb = image_embeds.i((.., image_offset, ..))?.unsqueeze(1)?;
                        input_embeds = input_embeds.slice_assign(
                            &[batch..batch + 1, pos..pos + 1, 0..hidden_dim],
                            &img_emb,
                        )?;
                        image_offset += 1;
                    }
                }
            }
        }

        // Generate position IDs
        let position_ids = self.generate_position_ids(seq_len, seqlen_offset)?;

        // Create causal mask for prefill
        let attention_mask = if seqlen_offset == 0 && seq_len > 1 {
            Some(self.create_causal_mask(batch_size, seq_len)?)
        } else {
            None
        };

        self.text.forward(
            input_embeds,
            attention_mask.as_ref(),
            &position_ids,
            seqlen_offset,
            &mut self.cache,
        )
    }

    /// Generate text using greedy decoding.
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &[(i64, i64, i64)],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();
        let mut generated_tokens = Vec::new();

        // Prefill
        let logits = self.forward(input_ids, Some(pixel_values), Some(grid_thw), 0)?;
        let next_token = logits
            .i((.., logits.dim(1)? - 1, ..))?
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        generated_tokens.push(next_token);
        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        // Decode
        let mut seqlen_offset = input_ids.dim(1)?;
        let mut current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        for _ in 1..max_new_tokens {
            let logits = self.forward(&current_ids, None, None, seqlen_offset)?;
            let next_token = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];

            generated_tokens.push(next_token);
            if next_token == eos_token_id {
                break;
            }

            seqlen_offset += 1;
            current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        Ok(generated_tokens)
    }

    /// Clear KV cache.
    pub fn clear_kv_cache(&mut self) {
        self.cache.clear();
    }

    /// Generate position IDs for xDRoPE.
    ///
    /// For xDRoPE, position_ids should be 3D: [batch, x_dim, seq_len]
    /// where x_dim = len(xdrope_section), dynamically obtained from config.
    fn generate_position_ids(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        // x_dim is dynamically obtained from config during initialization
        let x_dim = self.xdrope_x_dim;

        if seqlen_offset == 0 {
            // Prefill: generate sequential position IDs
            let ids: Vec<i64> = (0..seq_len as i64).collect();
            let position_ids_1d = Tensor::from_vec(ids, (seq_len,), &self.device)?;

            // Expand to 3D: [1, x_dim, seq_len]
            let mut dims = Vec::new();
            for _ in 0..x_dim {
                dims.push(position_ids_1d.clone());
            }
            let position_ids = Tensor::stack(&dims, 0)?;
            position_ids.unsqueeze(0)
        } else {
            // Decode: single position
            let ids = vec![seqlen_offset as i64; x_dim];
            let position_ids = Tensor::from_vec(ids, (x_dim, 1), &self.device)?;
            position_ids.unsqueeze(0)
        }
    }

    fn create_causal_mask(&self, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
        mask.expand((batch_size, 1, seq_len, seq_len))?
            .to_dtype(self.dtype)
    }
}
