//! PaddleOCR-VL Vision-Language Model for OCR.
//!
//! PaddleOCR-VL is a state-of-the-art vision-language model for document parsing,
//! combining a NaViT-style visual encoder with the ERNIE-4.5-0.3B language model.
//!
//! Key features:
//! - Dynamic resolution support for variable-sized document images
//! - 2D rotary position embeddings for vision, 1D for text
//! - Grouped Query Attention (GQA) for efficient inference
//! - Supports 109 languages for multilingual OCR
//! - Recognizes text, tables, formulas, and charts
//!
//! Architecture:
//! - Vision Encoder: NaViT-style with 27 layers, 1152 hidden dim, 16 heads
//! - Projector: 2x2 spatial merge + 2-layer MLP (1152*4 → 1024)
//! - Text Decoder: ERNIE-4.5-0.3B with 18 layers, GQA (16 query, 2 KV heads)
//!
//! References:
//! - [Paper](https://arxiv.org/abs/2510.14528)
//! - [HuggingFace Model](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;

pub mod config;
mod text;
mod vision;

pub use config::{Config, TextConfig, VisionConfig};
use text::TextModel;
pub use text::{
    compute_mrope_position_ids, compute_mrope_position_ids_multi, compute_mrope_position_ids_video,
    ImageGrid, VideoGrid,
};
use vision::VisionModel;

/// Type alias for debug generation output: generated tokens and per-step tensor exports.
pub type GenerateDebugOutput = (Vec<u32>, Vec<std::collections::HashMap<String, Tensor>>);

/// PaddleOCR-VL Model for vision-language OCR tasks.
///
/// This model combines a NaViT-style vision encoder with an ERNIE-4.5 text decoder
/// for document parsing tasks including OCR, table recognition, formula recognition,
/// and chart recognition.
pub struct PaddleOCRVLModel {
    vision: VisionModel,
    text: TextModel,
    image_token_id: u32,
    video_token_id: u32,
    dtype: DType,
    device: Device,
    /// Tracks the M-RoPE position delta for incremental decoding.
    /// After prefill with M-RoPE, incremental positions need adjustment.
    mrope_position_delta: i64,
}

impl PaddleOCRVLModel {
    /// Create a new PaddleOCR-VL model.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_cfg: TextConfig = cfg.into();
        // Vision model is at "visual.vision_model"
        let vision = VisionModel::new(
            &cfg.vision_config,
            cfg.hidden_size,
            vb.pp("visual").pp("vision_model"),
            vb.pp("mlp_AR"), // Projector is separate at "mlp_AR"
        )?;
        // Language model is at "model" (not "language_model.model")
        let text = TextModel::new(&text_cfg, vb.clone())?;

        Ok(Self {
            vision,
            text,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            mrope_position_delta: 0,
        })
    }

    /// Encode image to vision features.
    ///
    /// # Arguments
    /// * `pixel_values` - Image tensor of shape (batch, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3) with [temporal, height, width]
    ///
    /// # Returns
    /// Vision features projected to text model dimension
    pub fn encode_image(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        self.vision.forward(pixel_values, grid_thw)
    }

    /// Encode image with debug output.
    pub fn encode_image_debug(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        self.vision.forward_with_debug(pixel_values, grid_thw, true)
    }

    /// Encode image and export intermediate tensors for comparison with PyTorch.
    ///
    /// Returns vision features and a HashMap of checkpoint tensors.
    pub fn encode_image_with_export(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        self.vision.forward_with_export(pixel_values, grid_thw)
    }

    /// Encode multiple images, returning separate embeddings for each.
    ///
    /// # Arguments
    /// * `pixel_values` - Batched image tensor of shape (num_images, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3) with [temporal, height, width]
    ///
    /// # Returns
    /// Vector of vision feature tensors, one per image
    pub fn encode_images_multi(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<Vec<Tensor>> {
        self.vision.forward_multi(pixel_values, grid_thw)
    }

    /// Encode multiple images of different sizes separately.
    ///
    /// This method handles images with different resolutions by processing
    /// each image individually through the vision encoder.
    ///
    /// # Arguments
    /// * `pixel_values_list` - Vector of image tensors, each of shape (1, channels, height, width)
    /// * `grid_thw_list` - Vector of grid tensors, each of shape (1, 3)
    ///
    /// # Returns
    /// Vector of vision feature tensors, one per image
    pub fn encode_images_separate(
        &self,
        pixel_values_list: &[Tensor],
        grid_thw_list: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let mut embeddings = Vec::with_capacity(pixel_values_list.len());

        for (pixel_values, grid_thw) in pixel_values_list.iter().zip(grid_thw_list.iter()) {
            let emb = self.vision.forward(pixel_values, grid_thw)?;
            embeddings.push(emb);
        }

        Ok(embeddings)
    }

    /// Forward pass for vision-language generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `pixel_values` - Optional image tensor
    /// * `grid_thw` - Optional grid dimensions for images
    /// * `seqlen_offset` - Sequence length offset for KV cache
    ///
    /// # Returns
    /// Logits for next token prediction
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let hidden_dim = self.text.hidden_size;

        // Track grid dimensions for M-RoPE position computation
        let mut merged_grid_h = 0usize;
        let mut merged_grid_w = 0usize;

        // If we have images, encode them and inject into embeddings
        if let (Some(pixel_values), Some(grid_thw)) = (pixel_values, grid_thw) {
            // Encode images
            let image_embeds = self.encode_image(pixel_values, grid_thw)?;
            let image_embeds = image_embeds.to_dtype(self.dtype)?;

            // Get grid dimensions for M-RoPE (after 2x2 merge)
            let grid_thw_vec: Vec<u32> = grid_thw.flatten_all()?.to_vec1()?;
            if grid_thw_vec.len() >= 3 {
                let spatial_merge_size = 2; // 2x2 merge
                merged_grid_h = (grid_thw_vec[1] as usize) / spatial_merge_size;
                merged_grid_w = (grid_thw_vec[2] as usize) / spatial_merge_size;
            }

            // Find image token positions and replace with image embeddings
            let input_ids_flat = input_ids.flatten_all()?;
            let input_ids_vec = input_ids_flat.to_vec1::<u32>()?;

            let mut image_offset = 0usize;
            let num_image_tokens = image_embeds.dim(0)?;

            for batch in 0..batch_size {
                for pos in 0..seq_len {
                    let idx = batch * seq_len + pos;
                    if input_ids_vec[idx] == self.image_token_id && image_offset < num_image_tokens
                    {
                        // Replace this token's embedding with image embedding
                        let img_emb = image_embeds.i(image_offset)?.unsqueeze(0)?.unsqueeze(0)?;
                        input_embeds = input_embeds.slice_assign(
                            &[batch..batch + 1, pos..pos + 1, 0..hidden_dim],
                            &img_emb,
                        )?;
                        image_offset += 1;
                    }
                }
            }

            // Use M-RoPE with 3D position IDs for prefill with vision tokens
            let position_ids = compute_mrope_position_ids(
                input_ids,
                self.image_token_id,
                merged_grid_h,
                merged_grid_w,
                &self.device,
            )?;

            // Compute mrope_position_delta for incremental decoding
            // delta = max_position - seq_len + 1, so that position seq_len becomes max_position + 1
            let position_ids_vec: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
            let max_pos = position_ids_vec.iter().copied().max().unwrap_or(0);
            self.mrope_position_delta = max_pos + 1 - seq_len as i64;

            return self
                .text
                .forward_embeds_with_mrope(input_embeds, &position_ids);
        }

        // Forward through text model with M-RoPE (for incremental decoding)
        //
        // CRITICAL: We must use M-RoPE during generation, NOT 1D RoPE!
        //
        // Reason: M-RoPE and 1D RoPE produce DIFFERENT rotations even for the same position
        // because M-RoPE splits head_dim by mrope_section [32,48,48] and applies different
        // dimension's cos/sin to each section, while 1D RoPE just uses first 64 dims duplicated.
        //
        // For text tokens, all 3 position dimensions have the same value, but we still need
        // to use M-RoPE to maintain consistency with prefill.
        //
        // Position calculation: seqlen_offset + mrope_position_delta
        // This gives the correct sequential position after accounting for the difference
        // between sequence index and M-RoPE position caused by 2D vision token positions.
        let pos = seqlen_offset as i64 + self.mrope_position_delta;
        let (batch_size, seq_len, _) = input_embeds.dims3()?;

        // Create position_ids [3, batch, seq_len] with all dimensions = pos
        // For text tokens in generation, all 3 dimensions (temporal, height, width) are identical
        let positions: Vec<i64> = vec![pos; batch_size * seq_len];
        let pos_tensor = Tensor::from_vec(positions, (batch_size, seq_len), &self.device)?;
        let position_ids = Tensor::stack(&[&pos_tensor, &pos_tensor, &pos_tensor], 0)?;

        self.text
            .forward_embeds_with_mrope(input_embeds, &position_ids)
    }

    /// Forward pass for multi-image vision-language generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len) containing multiple image placeholder ranges
    /// * `pixel_values` - Batched image tensor of shape (num_images, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3) with [temporal, height, width]
    /// * `seqlen_offset` - Sequence length offset for KV cache (0 for prefill)
    ///
    /// # Returns
    /// Logits for next token prediction
    pub fn forward_multi_image(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        _seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let hidden_dim = self.text.hidden_size;

        // Encode all images, getting separate embeddings for each
        let image_embeds_list = self.encode_images_multi(pixel_values, grid_thw)?;
        let image_embeds_list: Vec<Tensor> = image_embeds_list
            .into_iter()
            .map(|t| t.to_dtype(self.dtype))
            .collect::<Result<Vec<_>>>()?;

        // Build image grids for M-RoPE position computation
        let grid_thw_vec: Vec<Vec<u32>> = grid_thw.to_vec2()?;
        let spatial_merge_size = 2; // 2x2 merge
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| ImageGrid {
                grid_h: (g[1] as usize) / spatial_merge_size,
                grid_w: (g[2] as usize) / spatial_merge_size,
            })
            .collect();

        // Find image token ranges and inject embeddings
        let input_ids_flat = input_ids.flatten_all()?;
        let input_ids_vec = input_ids_flat.to_vec1::<u32>()?;

        // Find all image token ranges
        let mut image_ranges: Vec<(usize, usize)> = Vec::new();
        let mut in_image = false;
        let mut image_start = 0usize;

        for (pos, &token_id) in input_ids_vec.iter().enumerate() {
            if token_id == self.image_token_id {
                if !in_image {
                    in_image = true;
                    image_start = pos;
                }
            } else if in_image {
                image_ranges.push((image_start, pos));
                in_image = false;
            }
        }
        if in_image {
            image_ranges.push((image_start, input_ids_vec.len()));
        }

        // Verify we have the right number of image ranges
        if image_ranges.len() != image_embeds_list.len() {
            return Err(candle::Error::Msg(format!(
                "Found {} image ranges but have {} encoded images",
                image_ranges.len(),
                image_embeds_list.len()
            )));
        }

        // Inject each image's embeddings at the correct positions
        for batch in 0..batch_size {
            for (img_idx, ((start, end), embeddings)) in image_ranges
                .iter()
                .zip(image_embeds_list.iter())
                .enumerate()
            {
                let num_tokens = end - start;
                let num_embeddings = embeddings.dim(0)?;

                if num_tokens != num_embeddings {
                    return Err(candle::Error::Msg(format!(
                        "Image {} has {} placeholder tokens but {} embeddings",
                        img_idx, num_tokens, num_embeddings
                    )));
                }

                // Replace each placeholder token with the corresponding embedding
                for (offset, pos) in (*start..*end).enumerate() {
                    let img_emb = embeddings.i(offset)?.unsqueeze(0)?.unsqueeze(0)?;
                    input_embeds = input_embeds
                        .slice_assign(&[batch..batch + 1, pos..pos + 1, 0..hidden_dim], &img_emb)?;
                }
            }
        }

        // Compute M-RoPE position IDs for multi-image input
        let position_ids = compute_mrope_position_ids_multi(
            input_ids,
            self.image_token_id,
            &image_grids,
            &self.device,
        )?;

        // Compute mrope_position_delta for incremental decoding
        let position_ids_vec: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let max_pos = position_ids_vec.iter().copied().max().unwrap_or(0);
        self.mrope_position_delta = max_pos + 1 - seq_len as i64;

        self.text
            .forward_embeds_with_mrope(input_embeds, &position_ids)
    }

    /// Forward pass for multi-image with variable resolutions.
    ///
    /// This method handles images of different sizes by processing each
    /// image separately through the vision encoder.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs containing multiple image placeholder ranges
    /// * `pixel_values_list` - Vector of image tensors, each (1, C, H, W)
    /// * `grid_thw_list` - Vector of grid tensors, each (1, 3)
    /// * `_seqlen_offset` - Unused, kept for API consistency
    pub fn forward_multi_image_separate(
        &mut self,
        input_ids: &Tensor,
        pixel_values_list: &[Tensor],
        grid_thw_list: &[Tensor],
        _seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let hidden_dim = self.text.hidden_size;

        // Encode each image separately
        let image_embeds_list = self.encode_images_separate(pixel_values_list, grid_thw_list)?;
        let image_embeds_list: Vec<Tensor> = image_embeds_list
            .into_iter()
            .map(|t| t.to_dtype(self.dtype))
            .collect::<Result<Vec<_>>>()?;

        // Build image grids for M-RoPE position computation
        let spatial_merge_size = 2; // 2x2 merge
        let mut image_grids: Vec<ImageGrid> = Vec::with_capacity(grid_thw_list.len());
        for grid_thw in grid_thw_list {
            let grid_vec: Vec<Vec<u32>> = grid_thw.to_vec2()?;
            let g = &grid_vec[0];
            image_grids.push(ImageGrid {
                grid_h: (g[1] as usize) / spatial_merge_size,
                grid_w: (g[2] as usize) / spatial_merge_size,
            });
        }

        // Find image token ranges and inject embeddings
        let input_ids_flat = input_ids.flatten_all()?;
        let input_ids_vec = input_ids_flat.to_vec1::<u32>()?;

        // Find all image token ranges
        let mut image_ranges: Vec<(usize, usize)> = Vec::new();
        let mut in_image = false;
        let mut image_start = 0usize;

        for (pos, &token_id) in input_ids_vec.iter().enumerate() {
            if token_id == self.image_token_id {
                if !in_image {
                    in_image = true;
                    image_start = pos;
                }
            } else if in_image {
                image_ranges.push((image_start, pos));
                in_image = false;
            }
        }
        if in_image {
            image_ranges.push((image_start, input_ids_vec.len()));
        }

        // Verify we have the right number of image ranges
        if image_ranges.len() != image_embeds_list.len() {
            return Err(candle::Error::Msg(format!(
                "Found {} image ranges but have {} encoded images",
                image_ranges.len(),
                image_embeds_list.len()
            )));
        }

        // Inject each image's embeddings at the correct positions
        for batch in 0..batch_size {
            for (img_idx, ((start, end), embeddings)) in image_ranges
                .iter()
                .zip(image_embeds_list.iter())
                .enumerate()
            {
                let num_tokens = end - start;
                let num_embeddings = embeddings.dim(0)?;

                if num_tokens != num_embeddings {
                    return Err(candle::Error::Msg(format!(
                        "Image {} has {} placeholder tokens but {} embeddings",
                        img_idx, num_tokens, num_embeddings
                    )));
                }

                // Replace each placeholder token with the corresponding embedding
                for (offset, pos) in (*start..*end).enumerate() {
                    let img_emb = embeddings.i(offset)?.unsqueeze(0)?.unsqueeze(0)?;
                    input_embeds = input_embeds
                        .slice_assign(&[batch..batch + 1, pos..pos + 1, 0..hidden_dim], &img_emb)?;
                }
            }
        }

        // Compute M-RoPE position IDs for multi-image input
        let position_ids = compute_mrope_position_ids_multi(
            input_ids,
            self.image_token_id,
            &image_grids,
            &self.device,
        )?;

        // Compute mrope_position_delta for incremental decoding
        let position_ids_vec: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let max_pos = position_ids_vec.iter().copied().max().unwrap_or(0);
        self.mrope_position_delta = max_pos + 1 - seq_len as i64;

        self.text
            .forward_embeds_with_mrope(input_embeds, &position_ids)
    }

    /// Generate text from image using greedy decoding.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs (including image placeholders)
    /// * `pixel_values` - Image tensor
    /// * `grid_thw` - Grid dimensions for images
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();

        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();

        // First forward pass with image
        let logits = self.forward(&current_ids, Some(pixel_values), Some(grid_thw), 0)?;
        let next_token = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        generated_tokens.push(next_token);

        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        let mut seqlen_offset = current_ids.dim(1)?;
        current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        // Subsequent forward passes (text only, using KV cache)
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

    /// Generate text from multiple images using greedy decoding.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs (including multiple image placeholder ranges)
    /// * `pixel_values` - Batched image tensor of shape (num_images, channels, height, width)
    /// * `grid_thw` - Grid dimensions tensor of shape (num_images, 3)
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate_multi_image(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();

        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();

        // First forward pass with all images
        let logits = self.forward_multi_image(&current_ids, pixel_values, grid_thw, 0)?;
        let next_token = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        generated_tokens.push(next_token);

        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        let mut seqlen_offset = current_ids.dim(1)?;
        current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        // Subsequent forward passes (text only, using KV cache)
        // Uses same incremental decoding as single-image generation
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

    /// Generate text from multiple images of different sizes using greedy decoding.
    ///
    /// This method handles images with different resolutions by processing
    /// each image separately through the vision encoder.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs (including multiple image placeholder ranges)
    /// * `pixel_values_list` - Vector of image tensors, each (1, C, H, W)
    /// * `grid_thw_list` - Vector of grid tensors, each (1, 3)
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate_multi_image_separate(
        &mut self,
        input_ids: &Tensor,
        pixel_values_list: &[Tensor],
        grid_thw_list: &[Tensor],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();

        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();

        // First forward pass with all images (processed separately)
        let logits =
            self.forward_multi_image_separate(&current_ids, pixel_values_list, grid_thw_list, 0)?;
        let next_token = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        generated_tokens.push(next_token);

        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        let mut seqlen_offset = current_ids.dim(1)?;
        current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        // Subsequent forward passes (text only, using KV cache)
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

    /// Forward pass for video input.
    ///
    /// This method processes video frames with temporal position encoding,
    /// where each frame gets sequential temporal positions (t=0, 1, 2, ...)
    /// unlike images which all use t=0.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs containing video placeholder tokens
    /// * `pixel_values_video` - Stacked video frames (num_frames * C * H * W flattened)
    /// * `video_grid_thw` - Grid dimensions (1, 3) = [temporal, height, width]
    /// * `fps` - Frames per second used to extract video frames
    /// * `seqlen_offset` - Sequence length offset for KV cache
    ///
    /// # Returns
    /// Logits for next token prediction
    pub fn forward_video(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        fps: f32,
        _seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let hidden_dim = self.text.hidden_size;

        // Encode video frames through vision encoder
        // The vision encoder treats video frames similarly to batched images
        let video_embeds = self.vision.forward(pixel_values_video, video_grid_thw)?;
        let video_embeds = video_embeds.to_dtype(self.dtype)?;

        // Build video grid for M-RoPE position computation
        let grid_thw_vec: Vec<Vec<u32>> = video_grid_thw.to_vec2()?;
        let g = &grid_thw_vec[0];
        let spatial_merge_size = 2; // 2x2 merge
        let video_grid = VideoGrid {
            grid_t: g[0] as usize,
            grid_h: (g[1] as usize) / spatial_merge_size,
            grid_w: (g[2] as usize) / spatial_merge_size,
        };
        // Find video token range and inject embeddings
        let input_ids_flat = input_ids.flatten_all()?;
        let input_ids_vec = input_ids_flat.to_vec1::<u32>()?;

        let mut video_start = None;
        let mut video_end = None;
        let mut in_video = false;

        for (pos, &token_id) in input_ids_vec.iter().enumerate() {
            if token_id == self.video_token_id {
                if !in_video {
                    in_video = true;
                    video_start = Some(pos);
                }
            } else if in_video {
                video_end = Some(pos);
                break;
            }
        }
        if in_video && video_end.is_none() {
            video_end = Some(input_ids_vec.len());
        }

        // Inject video embeddings
        if let (Some(start), Some(end)) = (video_start, video_end) {
            let num_tokens = end - start;
            let num_embeddings = video_embeds.dim(0)?;

            if num_tokens != num_embeddings {
                return Err(candle::Error::Msg(format!(
                    "Video has {} placeholder tokens but {} embeddings",
                    num_tokens, num_embeddings
                )));
            }

            for batch in 0..batch_size {
                for (offset, pos) in (start..end).enumerate() {
                    let emb = video_embeds.i(offset)?.unsqueeze(0)?.unsqueeze(0)?;
                    input_embeds = input_embeds
                        .slice_assign(&[batch..batch + 1, pos..pos + 1, 0..hidden_dim], &emb)?;
                }
            }
        }

        // Compute temporal scaling parameters for M-RoPE
        // HuggingFace Qwen2-VL uses simple sequential temporal indices (0, 1, 2, ...)
        // second_per_grid_t * tokens_per_second = 1.0 gives sequential frame indices
        // Python shows second_per_grid_ts = 0.5 with tokens_per_second = 2 -> 0.5 * 2 = 1.0
        let second_per_grid_t = 0.5f32; // Match Python processor output
        let tokens_per_second = 2usize;
        let _ = fps; // fps is used to determine how frames are sampled, not for position encoding

        // Compute M-RoPE position IDs with temporal encoding
        let position_ids = compute_mrope_position_ids_video(
            input_ids,
            self.video_token_id,
            &video_grid,
            second_per_grid_t,
            tokens_per_second,
            &self.device,
        )?;

        // Compute mrope_position_delta for incremental decoding
        let position_ids_vec: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let max_pos = position_ids_vec.iter().copied().max().unwrap_or(0);
        self.mrope_position_delta = max_pos + 1 - seq_len as i64;

        self.text
            .forward_embeds_with_mrope(input_embeds, &position_ids)
    }

    /// Generate text from video using greedy decoding.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs (including video placeholder tokens)
    /// * `pixel_values_video` - Stacked video frames
    /// * `video_grid_thw` - Grid dimensions (1, 3) = [temporal, height, width]
    /// * `fps` - Frames per second used to extract video frames
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Generated token IDs
    pub fn generate_video(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        fps: f32,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.clear_kv_cache();

        let repetition_penalty = 1.1f32;
        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();

        // Helper function to apply repetition penalty
        fn apply_repetition_penalty(
            logits: &Tensor,
            generated: &[u32],
            penalty: f32,
        ) -> Result<Tensor> {
            if generated.is_empty() || penalty == 1.0 {
                return Ok(logits.clone());
            }
            let device = logits.device();
            let original_shape = logits.dims().to_vec();
            let logits_flat = logits.flatten_all()?;
            let mut logits_vec: Vec<f32> = logits_flat.to_vec1()?;
            for &token in generated {
                let idx = token as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= penalty;
                    } else {
                        logits_vec[idx] *= penalty;
                    }
                }
            }
            Tensor::from_vec(logits_vec, original_shape, device)
        }

        // First forward pass with video
        let logits =
            self.forward_video(&current_ids, pixel_values_video, video_grid_thw, fps, 0)?;
        let logits = apply_repetition_penalty(&logits, &generated_tokens, repetition_penalty)?;
        let next_token = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        generated_tokens.push(next_token);

        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        let mut seqlen_offset = current_ids.dim(1)?;
        current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        // Subsequent forward passes (text only, using KV cache)
        for _ in 1..max_new_tokens {
            let logits = self.forward(&current_ids, None, None, seqlen_offset)?;
            let logits = apply_repetition_penalty(&logits, &generated_tokens, repetition_penalty)?;
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

    /// Clear all KV caches and reset M-RoPE position delta.
    pub fn clear_kv_cache(&mut self) {
        self.text.clear_kv_cache();
        self.mrope_position_delta = 0;
    }

    /// Forward pass with tensor export for decoder comparison.
    ///
    /// This method captures intermediate tensors at key checkpoints for
    /// comparison with PyTorch implementation.
    ///
    /// # Returns
    /// Tuple of (logits, HashMap of checkpoint tensors)
    pub fn forward_with_decoder_export(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        use std::collections::HashMap;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Step 1: Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        tensors.insert(
            "input_embeds_before_merge".to_string(),
            input_embeds.clone(),
        );
        let hidden_dim = self.text.hidden_size;

        // Step 2: Encode images
        let image_embeds = self.encode_image(pixel_values, grid_thw)?;
        let image_embeds = image_embeds.to_dtype(self.dtype)?;
        tensors.insert("vision_embeds".to_string(), image_embeds.clone());

        // Get grid dimensions for M-RoPE
        let grid_thw_vec: Vec<u32> = grid_thw.flatten_all()?.to_vec1()?;
        let spatial_merge_size = 2;
        let merged_grid_h = (grid_thw_vec[1] as usize) / spatial_merge_size;
        let merged_grid_w = (grid_thw_vec[2] as usize) / spatial_merge_size;

        // Step 3: Merge vision embeddings into text embeddings
        let input_ids_flat = input_ids.flatten_all()?;
        let input_ids_vec = input_ids_flat.to_vec1::<u32>()?;
        let mut image_offset = 0usize;
        let num_image_tokens = image_embeds.dim(0)?;

        for batch in 0..batch_size {
            for pos in 0..seq_len {
                let idx = batch * seq_len + pos;
                if input_ids_vec[idx] == self.image_token_id && image_offset < num_image_tokens {
                    let img_emb = image_embeds.i(image_offset)?.unsqueeze(0)?.unsqueeze(0)?;
                    input_embeds = input_embeds
                        .slice_assign(&[batch..batch + 1, pos..pos + 1, 0..hidden_dim], &img_emb)?;
                    image_offset += 1;
                }
            }
        }
        tensors.insert(
            "inputs_embeds_after_merge".to_string(),
            input_embeds.clone(),
        );

        // Step 4: Compute M-RoPE position IDs
        let position_ids = compute_mrope_position_ids(
            input_ids,
            self.image_token_id,
            merged_grid_h,
            merged_grid_w,
            &self.device,
        )?;
        tensors.insert("position_ids".to_string(), position_ids.clone());

        // Compute rope_deltas (max_pos - seq_len + 1)
        let position_ids_vec: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let max_pos = position_ids_vec.iter().copied().max().unwrap_or(0);
        let rope_delta = max_pos + 1 - seq_len as i64;

        // CRITICAL: Set mrope_position_delta for incremental decoding
        self.mrope_position_delta = rope_delta;

        tensors.insert(
            "rope_deltas".to_string(),
            Tensor::new(&[rope_delta], &self.device)?,
        );

        // Step 5: Forward through text model with export
        let (logits, decoder_tensors) = self
            .text
            .forward_embeds_with_mrope_export(input_embeds, &position_ids)?;

        // Merge decoder tensors
        for (k, v) in decoder_tensors {
            tensors.insert(k, v);
        }

        // Store last token logits
        let last_token_logits = logits.i((.., seq_len - 1, ..))?;
        tensors.insert("last_token_logits".to_string(), last_token_logits);

        Ok((logits, tensors))
    }

    /// Generate with debug tensor export at each step.
    ///
    /// Returns generated tokens and a vector of tensor maps for each step.
    pub fn generate_debug(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        max_steps: usize,
        eos_token_id: u32,
    ) -> Result<GenerateDebugOutput> {
        use std::collections::HashMap;

        self.clear_kv_cache();

        let mut generated_tokens = Vec::new();
        let mut all_tensors: Vec<HashMap<String, Tensor>> = Vec::new();

        // Step 0: Prefill with image
        let (logits, prefill_tensors) =
            self.forward_with_decoder_export(input_ids, pixel_values, grid_thw)?;

        let next_token = logits
            .i((.., logits.dim(1)? - 1, ..))?
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        let mut step_tensors = prefill_tensors;
        step_tensors.insert("step".to_string(), Tensor::new(&[0i64], &self.device)?);
        step_tensors.insert(
            "predicted_token".to_string(),
            Tensor::new(&[next_token as i64], &self.device)?,
        );
        step_tensors.insert(
            "mrope_delta".to_string(),
            Tensor::new(&[self.mrope_position_delta], &self.device)?,
        );
        all_tensors.push(step_tensors);

        generated_tokens.push(next_token);

        if next_token == eos_token_id || max_steps <= 1 {
            return Ok((generated_tokens, all_tensors));
        }

        // Generation steps
        let mut seqlen_offset = input_ids.dim(1)?;
        let mut current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

        for step in 1..max_steps {
            // Compute position for M-RoPE
            let pos = seqlen_offset as i64 + self.mrope_position_delta;
            let (batch_size, seq_len, _) = {
                let embeds = self.text.embed_tokens(&current_ids)?;
                embeds.dims3()?
            };

            // Create position_ids [3, batch, seq_len]
            let positions: Vec<i64> = vec![pos; batch_size * seq_len];
            let pos_tensor = Tensor::from_vec(positions, (batch_size, seq_len), &self.device)?;
            let position_ids = Tensor::stack(&[&pos_tensor, &pos_tensor, &pos_tensor], 0)?;

            // Get embeddings
            let input_embeds = self.text.embed_tokens(&current_ids)?;

            // Forward with export
            let (logits, decoder_tensors) = self
                .text
                .forward_embeds_with_mrope_export(input_embeds, &position_ids)?;

            let next_token = logits
                .i((.., logits.dim(1)? - 1, ..))?
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];

            let mut step_tensors: HashMap<String, Tensor> = decoder_tensors;
            step_tensors.insert(
                "step".to_string(),
                Tensor::new(&[step as i64], &self.device)?,
            );
            step_tensors.insert(
                "seqlen_offset".to_string(),
                Tensor::new(&[seqlen_offset as i64], &self.device)?,
            );
            step_tensors.insert(
                "mrope_position".to_string(),
                Tensor::new(&[pos], &self.device)?,
            );
            step_tensors.insert("position_ids".to_string(), position_ids);
            step_tensors.insert(
                "predicted_token".to_string(),
                Tensor::new(&[next_token as i64], &self.device)?,
            );
            all_tensors.push(step_tensors);

            generated_tokens.push(next_token);

            if next_token == eos_token_id {
                break;
            }

            seqlen_offset += 1;
            current_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        Ok((generated_tokens, all_tensors))
    }
}
