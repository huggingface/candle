//! Granite 4.0 3B Vision model for document understanding.
//!
//! Architecture: SigLIP2 vision encoder → Window Q-Former projectors → Deepstack/Spatial
//! injection into GraniteMoeHybrid language model.
//!
//! Key mechanism: vision features are extracted from multiple SigLIP layers and additively
//! injected at specific LLM layers (Deepstack), plus spatial offset features from the
//! deepest vision layer are injected at additional LLM layers.
//!
//! References:
//! - [Model Card](https://huggingface.co/ibm-granite/granite-4.0-3b-vision)

pub mod config;
pub mod downsampling;

use crate::models::granitemoehybrid::{GraniteMoeHybrid, GraniteMoeHybridCache};
use crate::models::siglip;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use config::{Config, FeatureSelectStrategy};
use downsampling::WindowQFormerDownsampler;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: build an injection tensor with vision features at image positions
// ---------------------------------------------------------------------------

fn build_injection_tensor(
    seq_len: usize,
    hidden_size: usize,
    input_ids: &[u32],
    features: &Tensor,
    image_token_id: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut segments: Vec<Tensor> = Vec::new();
    let mut feat_idx = 0;
    let mut pos = 0;

    while pos < seq_len {
        if input_ids[pos] == image_token_id {
            let start = pos;
            while pos < seq_len && input_ids[pos] == image_token_id {
                pos += 1;
            }
            let count = pos - start;
            let available = count.min(features.dim(0)? - feat_idx);
            if available > 0 {
                segments.push(features.narrow(0, feat_idx, available)?);
                feat_idx += available;
            }
            let remaining = count - available;
            if remaining > 0 {
                segments.push(Tensor::zeros((remaining, hidden_size), dtype, device)?);
            }
        } else {
            let start = pos;
            while pos < seq_len && input_ids[pos] != image_token_id {
                pos += 1;
            }
            let count = pos - start;
            segments.push(Tensor::zeros((count, hidden_size), dtype, device)?);
        }
    }

    Tensor::cat(&segments, 0)?.unsqueeze(0)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Model {
    vision_tower: siglip::VisionModel,
    layerwise_projectors: Vec<WindowQFormerDownsampler>,
    spatial_projectors: Vec<WindowQFormerDownsampler>,
    language_model: GraniteMoeHybrid,
    image_newline: Option<Tensor>,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Vision tower (SigLIP2)
        let vision_tower = siglip::VisionModel::new(
            &cfg.vision_config,
            false,
            vb.pp("vision_tower.vision_model"),
        )?;

        // Deepstack layerwise projectors
        let mut layerwise_projectors = Vec::with_capacity(cfg.deepstack_layer_map.len());
        for idx in 0..cfg.deepstack_layer_map.len() {
            let proj = WindowQFormerDownsampler::new(
                cfg,
                None,
                vb.pp(format!("model.layerwise_projectors.{idx}")),
            )?;
            layerwise_projectors.push(proj);
        }

        // Spatial projectors
        let mut spatial_projectors = Vec::new();
        if cfg.use_spatial_sampling {
            for idx in 0..cfg.spatial_target_layers.len() {
                let proj = WindowQFormerDownsampler::new(
                    cfg,
                    Some(idx),
                    vb.pp(format!("model.spatial_projectors.{idx}")),
                )?;
                spatial_projectors.push(proj);
            }
        }

        // Language model (GraniteMoeHybrid)
        let text_internal_cfg = cfg.text_config.clone().into_config(false);
        let language_model =
            GraniteMoeHybrid::load_inner(vb.pp("model.language_model"), &text_internal_cfg)?;

        // Image newline parameter
        let image_newline = if cfg.use_image_newline_parameter {
            Some(vb.get(cfg.text_config.hidden_size, "model.image_newline")?)
        } else {
            None
        };

        Ok(Self {
            vision_tower,
            layerwise_projectors,
            spatial_projectors,
            language_model,
            image_newline,
            config: cfg.clone(),
        })
    }

    /// Encode images and produce per-layer injection tensors for the LLM.
    /// Returns: Vec<(llm_layer_idx, features)> where features is (num_image_patches, hidden).
    fn get_image_features(&self, pixel_values: &Tensor) -> Result<Vec<(usize, Tensor)>> {
        // Run SigLIP with hidden state extraction
        let hidden_states = self.vision_tower.forward_with_hidden_states(pixel_values)?;

        let strip_cls =
            self.config.vision_feature_select_strategy == FeatureSelectStrategy::Default;
        let mut all_features = Vec::new();

        // Deepstack: extract from multiple vision layers, project via Q-Former
        for (idx, &[vision_layer, llm_layer]) in self.config.deepstack_layer_map.iter().enumerate()
        {
            let vis_idx = self.config.resolve_vision_layer(vision_layer);
            let mut selected = hidden_states[vis_idx].clone();

            if strip_cls {
                selected = selected.narrow(1, 1, selected.dim(1)? - 1)?;
            }

            let projected = self.layerwise_projectors[idx].forward(&selected)?;
            all_features.push((llm_layer as usize, projected));
        }

        // Spatial: extract 4 offset groups from deepest vision layer
        if self.config.use_spatial_sampling {
            let spatial_idx = self
                .config
                .resolve_vision_layer(self.config.spatial_vision_layer);
            let mut spatial_feature = hidden_states[spatial_idx].clone();

            if strip_cls {
                spatial_feature = spatial_feature.narrow(1, 1, spatial_feature.dim(1)? - 1)?;
            }

            for (group_idx, &llm_layer) in self.config.spatial_target_layers.iter().enumerate() {
                let projected = self.spatial_projectors[group_idx].forward(&spatial_feature)?;
                all_features.push((llm_layer, projected));
            }
        }

        Ok(all_features)
    }

    /// Pack projected features for a single image (handles multi-tile with unpadding).
    /// pixel_values: (num_tiles, C, H, W) for one image.
    /// image_size: (height, width) of original image.
    /// Returns packed features (total_tokens, llm_hidden).
    fn pack_image_features(&self, features: &Tensor, image_size: (usize, usize)) -> Result<Tensor> {
        let num_tiles = features.dim(0)?;

        if num_tiles > 1 {
            // First tile is the global thumbnail
            let base_features = features.get(0)?;
            let tile_features = features.narrow(0, 1, num_tiles - 1)?;

            let (q_side, w_side) = self.config.query_and_window_side();
            let patches_per_side = self.config.patches_per_side();
            let ds_patches = patches_per_side * q_side / w_side;

            // Figure out the tile grid dimensions
            let (grid_h, grid_w) = self.get_anyres_grid_shape(image_size);

            // Rearrange tiles into spatial grid
            // (grid_h, grid_w, ds_patches, ds_patches, hidden) → (hidden, grid_h*ds, grid_w*ds)
            let hidden = tile_features.dim(2)?;
            let arranged = tile_features
                .reshape((grid_h, grid_w, ds_patches, ds_patches, hidden))?
                .permute((4, 0, 2, 1, 3))?
                .contiguous()?;
            // (hidden, grid_h * ds_patches, grid_w * ds_patches)
            let full_h = grid_h * ds_patches;
            let full_w = grid_w * ds_patches;
            let arranged = arranged.reshape((hidden, full_h, full_w))?;

            // Unpad to actual aspect ratio
            let unpadded = self.unpad_image(&arranged, image_size)?;
            let (_, up_h, _up_w) = unpadded.dims3()?;

            // Add image newline tokens
            let unpadded = if let Some(ref newline) = self.image_newline {
                let nl = newline
                    .unsqueeze(1)?
                    .unsqueeze(2)?
                    .expand((hidden, up_h, 1))?;
                Tensor::cat(&[&unpadded, &nl], 2)?
            } else {
                unpadded
            };

            // Flatten and transpose: (hidden, H, W+1) → (H*(W+1), hidden)
            let (_, fh, fw) = unpadded.dims3()?;
            let flat = unpadded.reshape((hidden, fh * fw))?.t()?;

            // Prepend base features
            Tensor::cat(&[&base_features, &flat], 0)
        } else {
            // Single tile
            let features = features.get(0)?;
            if let Some(ref newline) = self.image_newline {
                let nl = newline.unsqueeze(0)?;
                Tensor::cat(&[&features, &nl], 0)
            } else {
                Ok(features)
            }
        }
    }

    fn get_anyres_grid_shape(&self, image_size: (usize, usize)) -> (usize, usize) {
        let (best_h, best_w) =
            select_best_resolution(image_size, &self.config.image_grid_pinpoints);
        let tile = self.config.vision_config.image_size;
        (best_h / tile, best_w / tile)
    }

    fn unpad_image(&self, tensor: &Tensor, original_size: (usize, usize)) -> Result<Tensor> {
        let (original_h, original_w) = original_size;
        let (_c, current_h, current_w) = tensor.dims3()?;

        let orig_aspect = original_w as f64 / original_h as f64;
        let curr_aspect = current_w as f64 / current_h as f64;

        if orig_aspect > curr_aspect {
            let scale = current_w as f64 / original_w as f64;
            let new_h = (original_h as f64 * scale) as usize;
            let padding = (current_h - new_h) / 2;
            tensor.narrow(1, padding, new_h)
        } else {
            let scale = current_h as f64 / original_h as f64;
            let new_w = (original_w as f64 * scale) as usize;
            let padding = (current_w - new_w) / 2;
            tensor.narrow(2, padding, new_w)
        }
    }

    /// Initial forward pass with vision injection.
    /// pixel_values: (num_tiles, C, H, W) for a single image.
    /// image_size: (height, width) of the original image.
    pub fn setup(
        &self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_size: (usize, usize),
        cache: &mut GraniteMoeHybridCache,
    ) -> Result<Tensor> {
        let device = input_ids.device();
        let dtype = pixel_values.dtype();
        let (_b, seq_len) = input_ids.dims2()?;
        let hidden_size = self.config.text_config.hidden_size;

        // 1. Get per-layer vision features
        let layer_features = self.get_image_features(pixel_values)?;

        // 2. Pack features for each injection point
        let mut deepstack_injections: Vec<(usize, Tensor)> = Vec::new();
        for (llm_layer, features) in &layer_features {
            let packed = self.pack_image_features(features, image_size)?;
            deepstack_injections.push((*llm_layer, packed));
        }

        // 3. Get text embeddings and zero out image positions
        let input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let text_embeds = self
            .language_model
            .word_token_embedding
            .forward(input_ids)?;

        let mask_vals: Vec<f32> = input_ids_vec
            .iter()
            .map(|&id| {
                if id == self.config.image_token_index {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, seq_len, 1), device)?.to_dtype(dtype)?;
        let text_embeds = text_embeds.to_dtype(dtype)?.broadcast_mul(&mask)?;

        // 4. Scale by embedding_multiplier
        let mut hidden = if (self.language_model.embedding_scale - 1.0).abs() < f32::EPSILON {
            text_embeds
        } else {
            text_embeds.affine(self.language_model.embedding_scale as f64, 0.)?
        };

        // 5. Build injection tensors for each target layer
        let injection_map: HashMap<usize, Tensor> = deepstack_injections
            .iter()
            .map(|(layer, features)| {
                let inj = build_injection_tensor(
                    seq_len,
                    hidden_size,
                    &input_ids_vec,
                    features,
                    self.config.image_token_index,
                    device,
                    dtype,
                )?;
                Ok((*layer, inj))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        // 6. Run through LLM layers with injection
        for (block_idx, block) in self.language_model.blocks.iter().enumerate() {
            if let Some(injection) = injection_map.get(&block_idx) {
                hidden = (hidden + injection)?;
            }
            hidden = block.forward(&hidden, 0, block_idx, cache)?;
        }

        // 7. Final norm + logits
        let hidden = self.language_model.ln_f.forward(&hidden)?;
        let hidden = hidden.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = hidden.matmul(&self.language_model.word_token_embedding.embeddings().t()?)?;
        let logits = logits.to_dtype(DType::F32)?;
        if (self.language_model.logits_scale - 1.0).abs() < f32::EPSILON {
            Ok(logits)
        } else {
            Ok(logits.affine(self.language_model.logits_scale as f64, 0.)?)
        }
    }

    /// Subsequent autoregressive forward pass (no vision injection).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut GraniteMoeHybridCache,
    ) -> Result<Tensor> {
        self.language_model.forward(input_ids, index_pos, cache)
    }
}

/// Select the best resolution from grid pinpoints for the given image size.
pub fn select_best_resolution(
    image_size: (usize, usize),
    pinpoints: &[[usize; 2]],
) -> (usize, usize) {
    let (orig_h, orig_w) = image_size;
    let mut best = (pinpoints[0][0], pinpoints[0][1]);
    let mut max_effective = 0usize;
    let mut min_waste = usize::MAX;

    for &[ph, pw] in pinpoints {
        let scale = f64::min(ph as f64 / orig_h as f64, pw as f64 / orig_w as f64);
        let downscaled_h = (orig_h as f64 * scale) as usize;
        let downscaled_w = (orig_w as f64 * scale) as usize;
        let effective = downscaled_h.min(ph) * downscaled_w.min(pw);
        let waste = (ph * pw).saturating_sub(effective);

        if effective > max_effective || (effective == max_effective && waste < min_waste) {
            max_effective = effective;
            min_waste = waste;
            best = (ph, pw);
        }
    }

    best
}
