//! Mistral3 Model Implementation
//!
//! Main model components for Mistral3 vision-language model.

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::mistral::Model as MistralModel;
use crate::models::pixtral::vision_model::Model as PixtralVisionModel;

use super::config::Mistral3Config;
use super::projector::MultiModalProjector;

/// Find positions of image tokens in input sequences.
///
/// # Arguments
/// * `input_ids` - Input token IDs: (batch_size, seq_len)
/// * `image_token_id` - The token ID used as image placeholder
///
/// # Returns
/// Vector of (batch_idx, seq_idx) positions where image tokens are found
pub fn find_image_token_positions(
    input_ids: &Tensor,
    image_token_id: usize,
) -> Result<Vec<(usize, usize)>> {
    // Handle both i64 and u32 token types
    let input_ids = if input_ids.dtype() == DType::U32 {
        input_ids.to_dtype(DType::I64)?
    } else {
        input_ids.clone()
    };

    let input_ids = input_ids.to_vec2::<i64>()?;
    let mut positions = Vec::new();

    for (batch_idx, sequence) in input_ids.iter().enumerate() {
        for (seq_idx, &token_id) in sequence.iter().enumerate() {
            if token_id as usize == image_token_id {
                positions.push((batch_idx, seq_idx));
            }
        }
    }

    Ok(positions)
}

/// Replace image token positions in embeddings with image features.
///
/// This is the Candle equivalent of PyTorch's `masked_scatter` operation.
///
/// # Arguments
/// * `inputs_embeds` - Text embeddings: (batch_size, seq_len, hidden_size)
/// * `image_embeds` - Image features: (num_image_tokens, hidden_size)
/// * `image_positions` - Positions to replace: [(batch_idx, seq_idx), ...]
/// * `device` - Device for tensor operations
///
/// # Returns
/// Updated embeddings with image features inserted at specified positions
pub fn replace_image_tokens(
    inputs_embeds: &Tensor,
    image_embeds: &Tensor,
    image_positions: &[(usize, usize)],
    device: &Device,
) -> Result<Tensor> {
    if image_positions.is_empty() {
        return Ok(inputs_embeds.clone());
    }

    let (batch_size, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let num_image_tokens = image_positions.len();

    // Validate image embeddings count
    let (total_image_embeds, _) = image_embeds.dims2()?;

    if total_image_embeds < num_image_tokens {
        candle::bail!(
            "Not enough image embeddings: need {}, got {}",
            num_image_tokens,
            total_image_embeds
        );
    }

    let image_embeds = if num_image_tokens == total_image_embeds {
        image_embeds.clone()
    } else {
        image_embeds.i(0..num_image_tokens)?
    };

    // Replace tokens position by position
    let mut result = inputs_embeds.clone();

    for (idx, &(batch_idx, seq_idx)) in image_positions.iter().enumerate() {
        if batch_idx >= batch_size || seq_idx >= seq_len {
            candle::bail!(
                "Invalid image position: ({}, {}) for tensor shape ({}, {}, {})",
                batch_idx,
                seq_idx,
                batch_size,
                seq_len,
                hidden_size
            );
        }

        // Get the image embedding for this position
        let image_embed = image_embeds.i(idx)?;

        // Create position mask
        let mut position_mask = vec![0f32; batch_size * seq_len];
        position_mask[batch_idx * seq_len + seq_idx] = 1.0;
        let position_mask = Tensor::new(position_mask.as_slice(), device)?
            .reshape((batch_size, seq_len, 1))?
            .to_dtype(inputs_embeds.dtype())?;

        // Broadcast image embedding to full tensor shape
        let image_embed_broadcast = image_embed
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, seq_len, hidden_size))?;

        // Update result: mask=0 keeps original, mask=1 uses image embedding
        let inverse_mask = (1.0 - &position_mask)?;
        let masked_result = result.broadcast_mul(&inverse_mask)?;
        let masked_image = image_embed_broadcast.broadcast_mul(&position_mask)?;
        result = (masked_result + masked_image)?;
    }

    Ok(result)
}

/// Mistral3Model - the core multimodal model without lm_head.
///
/// Combines:
/// - Vision Tower (Pixtral)
/// - Multi-Modal Projector
/// - Language Model (Mistral, without lm_head)
#[derive(Debug, Clone)]
pub struct Mistral3Model {
    vision_tower: PixtralVisionModel,
    multi_modal_projector: MultiModalProjector,
    language_model: MistralModel,
    image_token_index: usize,
    spatial_merge_size: usize,
    patch_size: usize,
}

impl Mistral3Model {
    /// Create a new Mistral3Model.
    ///
    /// VarBuilder paths correspond to safetensors key prefixes:
    /// - `vision_tower.*` → vb.pp("vision_tower")
    /// - `multi_modal_projector.*` → vb.pp("multi_modal_projector")
    /// - `language_model.*` → vb.pp("language_model")
    ///
    /// All components use the dtype specified in VarBuilder (e.g., BF16).
    /// This matches HuggingFace Transformers behavior where all components
    /// share the same dtype, with only specific operations (RMSNorm, Softmax, RoPE)
    /// temporarily using F32 for numerical stability.
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        // Vision Tower (Pixtral) - uses VarBuilder's dtype
        let vision_tower = PixtralVisionModel::new(&cfg.vision_config, vb.pp("vision_tower"))?;
        // let vision_tower = PixtralVisionModel::new(
        //     &cfg.vision_config,
        //     vb.pp("vision_tower").to_dtype(DType::F32),
        // )?;

        // Multi-Modal Projector - uses VarBuilder's dtype
        let multi_modal_projector =
            MultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;
        // let multi_modal_projector = MultiModalProjector::new(
        //     cfg,
        //     vb.pp("multi_modal_projector").to_dtype(DType::F32),
        // )?;
        // Language Model (Mistral)
        // Note: mistral::Model::new internally adds "model" prefix
        let language_model = MistralModel::new(&cfg.text_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
            language_model,
            image_token_index: cfg.image_token_index,
            spatial_merge_size: cfg.spatial_merge_size,
            patch_size: cfg.vision_config.patch_size,
        })
    }

    /// Get image features from pixel values.
    ///
    /// # Arguments
    /// * `pixel_values` - Preprocessed images: (batch, channels, height, width)
    /// * `image_sizes` - Original pixel sizes: [(height, width), ...]
    ///
    /// # Returns
    /// Vector of feature tensors, one per image
    ///
    /// # Note
    /// Uses `forward_with_hidden_states` to get output with batch dimension preserved,
    /// matching PyTorch Transformers behavior. The last hidden state is then processed
    /// through the multi-modal projector.
    pub fn get_image_features(
        &self,
        pixel_values: &Tensor,
        image_sizes: &[(usize, usize)],
    ) -> Result<Vec<Tensor>> {
        // 1. Vision tower forward with hidden states support
        // Returns (batch, patches, hidden_size) with batch dimension preserved
        let vision_output = self.vision_tower.forward_with_hidden_states(pixel_values, true)?;

        // Get the last hidden state: (batch, patches, hidden_size)
        // For Mistral3, we use hidden_states[-1] which equals last_hidden_state
        let image_outputs = vision_output
            .hidden_states
            .as_ref()
            .and_then(|hs| hs.last().cloned())
            .unwrap_or(vision_output.last_hidden_state);

        // Squeeze batch dimension for projector: (batch, patches, hidden_size) -> (patches, hidden_size)
        let image_outputs = if image_outputs.dims().len() == 3 && image_outputs.dim(0)? == 1 {
            image_outputs.squeeze(0)?
        } else {
            image_outputs
        };

        // 2. Multi-modal projector
        let image_features = self
            .multi_modal_projector
            .forward(&image_outputs, image_sizes)?;

        // 3. Split features by image
        let downsample_ratio = self.patch_size * self.spatial_merge_size;
        let split_sizes: Vec<usize> = image_sizes
            .iter()
            .map(|(h, w)| (h / downsample_ratio) * (w / downsample_ratio))
            .collect();

        let mut features = Vec::new();
        let mut offset = 0;
        for size in split_sizes {
            features.push(image_features.narrow(0, offset, size)?);
            offset += size;
        }

        Ok(features)
    }

    /// Forward pass returning hidden states (not logits).
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs: (batch_size, seq_len)
    /// * `pixel_values` - Optional preprocessed images
    /// * `image_sizes` - Optional image sizes
    /// * `seqlen_offset` - Sequence length offset for KV cache
    ///
    /// # Returns
    /// Hidden states: (batch_size, seq_len, hidden_size)
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_sizes: Option<&[(usize, usize)]>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // 1. Get text embeddings
        let mut inputs_embeds = self.language_model.embed_tokens().forward(input_ids)?;

        // 2. Process images if provided
        if let (Some(pixels), Some(sizes)) = (pixel_values, image_sizes) {
            let image_features = self.get_image_features(pixels, sizes)?;
            let image_features = Tensor::cat(&image_features, 0)?
                .to_dtype(inputs_embeds.dtype())?
                .to_device(inputs_embeds.device())?;

            // 3. Find and replace image tokens
            let image_positions = find_image_token_positions(input_ids, self.image_token_index)?;
            inputs_embeds = replace_image_tokens(
                &inputs_embeds,
                &image_features,
                &image_positions,
                input_ids.device(),
            )?;
        }

        // 4. Forward through language model (returns hidden states, not logits)
        self.language_model
            .forward_embeds_hidden(&inputs_embeds, None, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    /// Returns a reference to the underlying language model.
    /// This is useful for accessing the lm_head without loading it twice.
    pub fn language_model(&self) -> &MistralModel {
        &self.language_model
    }
}

/// Mistral3ForConditionalGeneration - complete model with lm_head.
///
/// This is the main entry point for Mistral3 inference.
/// Note: The lm_head is shared with the underlying MistralModel to avoid
/// loading the same weights twice (saves ~1.34 GB for 24B model).
#[derive(Debug, Clone)]
pub struct Mistral3ForConditionalGeneration {
    model: Mistral3Model,
    dtype: DType,
    device: Device,
}

impl Mistral3ForConditionalGeneration {
    /// Create a new Mistral3ForConditionalGeneration.
    ///
    /// Weight paths in safetensors:
    /// - `vision_tower.*`, `multi_modal_projector.*`, `language_model.*` at root level
    /// - `lm_head` is at `language_model.lm_head.weight` (loaded by MistralModel)
    ///
    /// Note: The lm_head is reused from the underlying MistralModel to avoid
    /// duplicate memory allocation.
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let model = Mistral3Model::new(cfg, vb.clone())?;

        Ok(Self {
            model,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Forward pass returning full sequence logits.
    ///
    /// # Returns
    /// Logits: (batch_size, seq_len, vocab_size)
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_sizes: Option<&[(usize, usize)]>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let hidden_states = self
            .model
            .forward(input_ids, pixel_values, image_sizes, seqlen_offset)?;
        // Reuse lm_head from the underlying MistralModel
        self.model.language_model().lm_head().forward(&hidden_states)
    }

    /// Forward pass optimized for generation - returns only last token logits.
    ///
    /// # Returns
    /// Logits for last token: (batch_size, 1, vocab_size)
    pub fn forward_generate(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_sizes: Option<&[(usize, usize)]>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let hidden_states = self
            .model
            .forward(input_ids, pixel_values, image_sizes, seqlen_offset)?;

        // Only take last token's hidden states
        let seq_len = hidden_states.dim(1)?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;

        // Reuse lm_head from the underlying MistralModel
        self.model.language_model().lm_head().forward(&last_hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    /// Encode images to features (for preprocessing).
    pub fn encode_image(
        &self,
        pixel_values: &Tensor,
        image_sizes: &[(usize, usize)],
    ) -> Result<Vec<Tensor>> {
        self.model.get_image_features(pixel_values, image_sizes)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
