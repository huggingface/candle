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
    spatial_merge_size: usize,
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
        // Text model at "model" prefix (matches weight file structure)
        let text = TextModel::new(&cfg.text_config, use_flash_attn, vb.pp("model"))?;

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
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
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

        // Prefill: logits shape [batch, seq_len, vocab_size]
        let logits = self.forward(input_ids, Some(pixel_values), Some(grid_thw), 0)?;
        let seq_len = logits.dim(1)?;
        // Get last position logits: [vocab_size]
        let next_token_logits = logits.i((0, seq_len - 1))?;
        let next_token = next_token_logits
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;

        generated_tokens.push(next_token);
        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        // Decode
        let mut seqlen_offset = input_ids.dim(1)?;
        let mut current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        for _ in 1..max_new_tokens {
            // Decode: logits shape [batch=1, seq_len=1, vocab_size]
            let logits = self.forward(&current_ids, None, None, seqlen_offset)?;
            // Get logits for the single token: [vocab_size]
            let next_token_logits = logits.i((0, 0))?;
            let next_token = next_token_logits
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;

            generated_tokens.push(next_token);
            if next_token == eos_token_id {
                break;
            }

            seqlen_offset += 1;
            current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        Ok(generated_tokens)
    }

    /// Generate text using greedy decoding with proper xDRoPE position IDs.
    ///
    /// This method generates proper 4D xDRoPE position IDs for image tokens,
    /// where each image token has row/column position information.
    pub fn generate_with_xdrope<F>(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &[(i64, i64, i64)],
        input_ids_vec: &[u32],
        max_new_tokens: usize,
        is_eos: F,
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        self.clear_kv_cache();
        let mut generated_tokens = Vec::new();

        // Generate xDRoPE position IDs for prefill
        let position_ids = self.generate_xdrope_position_ids(
            input_ids_vec,
            grid_thw,
            self.spatial_merge_size,
        )?;

        // Prefill with xDRoPE position IDs
        let logits = self.forward_with_position_ids(
            input_ids,
            Some(pixel_values),
            Some(grid_thw),
            &position_ids,
            0,
        )?;

        let seq_len = logits.dim(1)?;
        let next_token_logits = logits.i((0, seq_len - 1))?;
        let next_token = next_token_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        if is_eos(next_token) {
            return Ok(generated_tokens);
        }
        generated_tokens.push(next_token);

        // Decode phase
        let mut seqlen_offset = input_ids.dim(1)?;
        let mut current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        for _ in 1..max_new_tokens {
            // Generate decode position IDs (simple sequential for decode)
            let decode_position_ids = self.generate_position_ids(1, seqlen_offset)?;

            let logits = self.forward_with_position_ids(
                &current_ids,
                None,
                None,
                &decode_position_ids,
                seqlen_offset,
            )?;

            let next_token_logits = logits.i((0, 0))?;
            let next_token = next_token_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

            if is_eos(next_token) {
                break;
            }
            generated_tokens.push(next_token);

            seqlen_offset += 1;
            current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        Ok(generated_tokens)
    }

    /// Forward pass with explicit position IDs.
    fn forward_with_position_ids(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        grid_thw: Option<&[(i64, i64, i64)]>,
        position_ids: &Tensor,
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

        // Create causal mask for prefill
        let attention_mask = if seqlen_offset == 0 && seq_len > 1 {
            Some(self.create_causal_mask(batch_size, seq_len)?)
        } else {
            None
        };

        self.text.forward(
            input_embeds,
            attention_mask.as_ref(),
            position_ids,
            seqlen_offset,
            &mut self.cache,
        )
    }

    /// Generate xDRoPE position IDs for the input sequence.
    ///
    /// For xDRoPE, position_ids have 4 dimensions:
    /// - Dimension 0: text sequential position [0, 1, 2, ..., seq_len-1]
    /// - Dimension 1: width/column position (for image patches)
    /// - Dimension 2: height/row position (for image patches)
    /// - Dimension 3: time/frame position (0 for images)
    fn generate_xdrope_position_ids(
        &self,
        input_ids: &[u32],
        grid_thw: &[(i64, i64, i64)],
        spatial_merge_size: usize,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();

        // Initialize 4 dimensions of position_ids
        // Default: all use sequential positions [0, 1, 2, ..., seq_len-1]
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let mut position_ids_w: Vec<i64> = (0..seq_len as i64).collect();
        let mut position_ids_h: Vec<i64> = (0..seq_len as i64).collect();
        let mut position_ids_t: Vec<i64> = (0..seq_len as i64).collect();

        // Find all image_token positions
        let image_token_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.image_token_id)
            .map(|(idx, _)| idx)
            .collect();

        if !image_token_positions.is_empty() && !grid_thw.is_empty() {
            // Calculate token count per image (excluding IM_START/IM_END)
            let mut image_token_counts: Vec<usize> = Vec::new();
            for (_t, h, w) in grid_thw {
                let grid_h = (*h / spatial_merge_size as i64) as usize;
                let grid_w = (*w / spatial_merge_size as i64) as usize;
                // Tokens per image: grid_h * (grid_w + 1)
                // +1 is for newline token at end of each row
                let tokens_per_image = grid_h * (grid_w + 1);
                image_token_counts.push(tokens_per_image);
            }

            // Set correct position_ids for each image
            let mut image_token_offset = 0;
            for (img_idx, (_t, _h, w)) in grid_thw.iter().enumerate() {
                let grid_w = (*w / spatial_merge_size as i64) as usize;
                let tokens_per_image = image_token_counts[img_idx];

                let start_idx = image_token_offset;
                let end_idx = start_idx + tokens_per_image;

                if start_idx < image_token_positions.len() && end_idx <= image_token_positions.len()
                {
                    for (local_idx, &global_pos) in
                        image_token_positions[start_idx..end_idx].iter().enumerate()
                    {
                        // Calculate 2D grid coordinates
                        // Row: local_idx / (grid_w + 1)
                        // Col: local_idx % (grid_w + 1)
                        let row = (local_idx / (grid_w + 1)) as i64;
                        let col = (local_idx % (grid_w + 1)) as i64;

                        // Dimension 1: column position
                        position_ids_w[global_pos] = col;
                        // Dimension 2: row position
                        position_ids_h[global_pos] = row;
                        // Dimension 3: time position (0 for single image)
                        position_ids_t[global_pos] = 0;
                        // Dimension 0: keep sequential position unchanged
                    }
                }

                image_token_offset += tokens_per_image;
            }
        }

        // Build 4D position_ids tensor: [1, 4, seq_len]
        let pos_ids_tensor = Tensor::from_vec(position_ids, (seq_len,), &self.device)?;
        let pos_w_tensor = Tensor::from_vec(position_ids_w, (seq_len,), &self.device)?;
        let pos_h_tensor = Tensor::from_vec(position_ids_h, (seq_len,), &self.device)?;
        let pos_t_tensor = Tensor::from_vec(position_ids_t, (seq_len,), &self.device)?;

        let stacked =
            Tensor::stack(&[pos_ids_tensor, pos_w_tensor, pos_h_tensor, pos_t_tensor], 0)?;
        stacked.unsqueeze(0)
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
