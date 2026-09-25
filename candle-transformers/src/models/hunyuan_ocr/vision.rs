//! Vision encoder for HunyuanOCR.
//!
//! Contains:
//! - PatchEmbed: Patch embedding with position interpolation
//! - VisionAttention: Standard multi-head attention
//! - VisionMlp: Two-layer MLP with GELU
//! - VisionBlock: Transformer block
//! - PatchMerger: Spatial merging with special tokens
//! - VisionModel: Complete vision encoder

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Embedding, Linear, VarBuilder};

use super::config::VisionConfig;

// ============================================================================
// Precise LayerNorm (F32 computation for numerical stability)
// ============================================================================

/// Precise LayerNorm that computes in F32 for numerical stability.
///
/// Unlike candle_nn::LayerNorm, this implementation:
/// 1. Performs all computations in F32
/// 2. Converts back to original dtype at the end
pub struct PreciseLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl PreciseLayerNorm {
    pub fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dtype = input.dtype();
        let input_device = input.device().clone();

        // Convert to F32 for computation
        let x = input.to_dtype(DType::F32)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let bias_f32 = self.bias.to_dtype(DType::F32)?;

        let hidden_size = x.dim(D::Minus1)?;

        // Compute mean
        let mean = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;

        // Center
        let centered = x.broadcast_sub(&mean)?;

        // Compute variance
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;

        // Normalize
        let denom = (var + self.eps)?.sqrt()?;
        let normed = centered.broadcast_div(&denom)?;

        // Scale and shift
        let scaled = normed.broadcast_mul(&weight_f32)?;
        let shifted = scaled.broadcast_add(&bias_f32)?;

        // Convert back to original dtype
        shifted.to_dtype(input_dtype)?.to_device(&input_device)
    }
}

// ============================================================================
// RMSNorm for PatchMerger
// ============================================================================

/// RMSNorm layer for PatchMerger.
pub struct RmsNormVision {
    weight: Tensor,
    eps: f64,
}

impl RmsNormVision {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;

        // RMS computation
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Apply weight
        let result = x_normed.broadcast_mul(&weight_f32)?;

        result.to_dtype(original_dtype)
    }
}

// ============================================================================
// Patch Embedding
// ============================================================================

/// Patch embedding with position embedding interpolation.
pub struct PatchEmbed {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    patch_pos_embed: Option<Tensor>,
    config: VisionConfig,
}

impl PatchEmbed {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;

        let max_num_patches = (cfg.max_image_size / cfg.patch_size).pow(2);
        let position_embedding = candle_nn::embedding(
            max_num_patches + 1,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            patch_pos_embed: None,
            config: cfg.clone(),
        })
    }

    /// Initialize position embedding cache.
    fn init_patch_pos_embed(&mut self, device: &Device, dtype: DType) -> Result<()> {
        if self.patch_pos_embed.is_some() {
            return Ok(());
        }

        // Get position embedding weights [max_num_patches+1, hidden_size]
        let pos_embed_weight = self.position_embedding.embeddings();

        // Skip first token (CLS token), use [1:, :]
        let pos_embed = pos_embed_weight.narrow(0, 1, pos_embed_weight.dim(0)? - 1)?;

        // Calculate edge length (square grid)
        let max_num_patches = (self.config.max_image_size / self.config.patch_size).pow(2);
        let edge = (max_num_patches as f64).sqrt() as usize;

        // Reshape to 2D grid: [1, edge, edge, hidden_size]
        let pos_embed = pos_embed.reshape((1, edge, edge, self.config.hidden_size))?;

        // Transpose to [1, hidden_size, edge, edge]
        let pos_embed = pos_embed.permute((0, 3, 1, 2))?;

        // Convert to target device and dtype
        let pos_embed = pos_embed.to_device(device)?.to_dtype(dtype)?;

        self.patch_pos_embed = Some(pos_embed);
        Ok(())
    }

    /// Interpolate position embedding to target size.
    fn interpolate_pos_embed(
        &self,
        h: i64,
        w: i64,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let patch_pos_embed = self
            .patch_pos_embed
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("Position embedding not initialized".to_string()))?;

        // Use scale_factor for interpolation (matching PyTorch behavior)
        // Add 0.1 to avoid floating point errors, same as original transformers impl
        let h_float = h as f64 + 0.1;
        let w_float = w as f64 + 0.1;

        let max_num_patches = (self.config.max_image_size / self.config.patch_size).pow(2);
        let position_edge = (max_num_patches as f64).sqrt();

        let scale_h = h_float / position_edge;
        let scale_w = w_float / position_edge;

        // Use candle's native upsample_bilinear2d_with_scale method
        let pos_embed = patch_pos_embed.upsample_bilinear2d_with_scale(scale_h, scale_w, false)?;

        // Reshape to sequence format: [1, hidden_size, h, w] -> [1, h*w, hidden_size]
        let pos_embed = pos_embed
            .reshape((self.config.hidden_size, (h * w) as usize))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        pos_embed.to_dtype(dtype)?.to_device(device)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `pixel_values` - Input patches: [num_patches, 3, patch_size, patch_size]
    /// * `grid_thw` - Grid info for each image: [(t, h, w), ...]
    ///
    /// # Returns
    /// * embeddings: [1, total_patches, hidden_size]
    pub fn forward(
        &mut self,
        pixel_values: &Tensor,
        grid_thw: &[(i64, i64, i64)],
    ) -> Result<Tensor> {
        let model_device = self.patch_embedding.weight().device().clone();
        let dtype = pixel_values.dtype();

        // Move input to model device if needed
        let pixel_values = if pixel_values.device().same_device(&model_device) {
            pixel_values.clone()
        } else {
            pixel_values.to_device(&model_device)?
        };

        // Step 1: Generate patch embeddings via Conv2d
        let patch_embeds = self.patch_embedding.forward(&pixel_values)?;
        // patch_embeds: [num_patches, hidden_size, 1, 1]

        // Step 2: Remove spatial dims and add batch dim
        let patch_embeds = patch_embeds
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?
            .unsqueeze(0)?;
        // patch_embeds: [1, num_patches, hidden_size]

        // Step 3: Initialize position embedding cache
        self.init_patch_pos_embed(&model_device, dtype)?;

        // Step 4: Interpolate position embedding for each image
        let mut patch_pos_embed_list = Vec::new();
        for (_t, h, w) in grid_thw {
            let pos_embed = self.interpolate_pos_embed(*h, *w, &model_device, dtype)?;
            patch_pos_embed_list.push(pos_embed);
        }

        // Step 5: Concatenate all position embeddings
        let patch_pos_embed = if patch_pos_embed_list.len() == 1 {
            patch_pos_embed_list[0].clone()
        } else {
            Tensor::cat(&patch_pos_embed_list, 1)?
        };

        // Step 6: Add position embedding
        patch_embeds.broadcast_add(&patch_pos_embed)
    }
}

// ============================================================================
// Vision Attention
// ============================================================================

/// Vision attention (standard multi-head attention without KV cache).
struct VisionAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head format
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Reshape to 3D for Metal compatibility
        let q_3d = q.reshape((batch_size * self.num_heads, seq_len, self.head_dim))?;
        let k_3d = k.reshape((batch_size * self.num_heads, seq_len, self.head_dim))?;
        let v_3d = v.reshape((batch_size * self.num_heads, seq_len, self.head_dim))?;

        // Compute attention scores
        let attn_weights = q_3d.matmul(&k_3d.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v_3d)?;

        // Reshape back
        let attn_output =
            attn_output.reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}

// ============================================================================
// Vision MLP
// ============================================================================

/// Vision MLP with GELU activation.
struct VisionMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl VisionMlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let dense_h_to_4h = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("dense_h_to_4h"),
        )?;
        let dense_4h_to_h = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("dense_4h_to_h"),
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_h_to_4h.forward(x)?;
        // Use gelu_erf() to match PyTorch's default GELU (erf exact version)
        let x = x.gelu_erf()?;
        self.dense_4h_to_h.forward(&x)
    }
}

// ============================================================================
// Vision Block
// ============================================================================

/// Vision transformer block with pre-norm architecture.
struct VisionBlock {
    input_layernorm: PreciseLayerNorm,
    self_attn: VisionAttention,
    post_attention_layernorm: PreciseLayerNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            PreciseLayerNorm::load(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = PreciseLayerNorm::load(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let mlp = VisionMlp::new(cfg, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

// ============================================================================
// Patch Merger
// ============================================================================

/// Patch merger for spatial merging and projection to text space.
struct PatchMerger {
    before_rms: RmsNormVision,
    proj_conv1: Conv2d,
    proj_conv2: Conv2d,
    mlp: Linear,
    after_rms: RmsNormVision,
    image_newline: Tensor,
    image_begin: Tensor,
    image_end: Tensor,
    config: VisionConfig,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_channels = cfg.hidden_size;
        let mid_channels = in_channels * 2;
        let proj_out_channels = in_channels * 4;
        let out_channels = cfg.out_hidden_size;
        let spatial_merge_size = cfg.spatial_merge_size;

        let before_rms = RmsNormVision::new(in_channels, cfg.rms_norm_eps, vb.pp("before_rms"))?;
        let after_rms = RmsNormVision::new(out_channels, cfg.rms_norm_eps, vb.pp("after_rms"))?;

        // First Conv2d: in_channels -> mid_channels
        let proj_conv1 = conv2d(
            in_channels,
            mid_channels,
            spatial_merge_size,
            Conv2dConfig {
                stride: spatial_merge_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj").pp("0"),
        )?;

        // Second Conv2d: mid_channels -> proj_out_channels
        let proj_conv2 = conv2d(
            mid_channels,
            proj_out_channels,
            1,
            Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj").pp("2"),
        )?;

        // MLP projection: proj_out_channels -> out_channels
        let mlp = linear(proj_out_channels, out_channels, vb.pp("mlp"))?;

        // Special tokens
        let image_newline = vb.get((proj_out_channels,), "image_newline")?;
        let image_begin = vb.get((out_channels,), "image_begin")?;
        let image_end = vb.get((out_channels,), "image_end")?;

        Ok(Self {
            before_rms,
            proj_conv1,
            proj_conv2,
            mlp,
            after_rms,
            image_newline,
            image_begin,
            image_end,
            config: cfg.clone(),
        })
    }

    fn forward(&self, x: &Tensor, grid_thw: &[(i64, i64, i64)]) -> Result<Tensor> {
        let (batch_size, _seq_len, hidden_size) = x.dims3()?;

        // Step 1: Apply RMSNorm
        let x = self.before_rms.forward(x)?;

        // Step 2: Process each image
        let mut processed_images = Vec::new();
        let mut start_idx = 0;
        let out_channels = self.config.out_hidden_size;

        for (_t, h, w) in grid_thw {
            let num_patches = (h * w) as usize;
            let end_idx = start_idx + num_patches;

            // Extract patches for current image
            let image_patches = x.narrow(1, start_idx, num_patches)?;

            // Reshape to 2D spatial format
            let image_patches = image_patches.permute((0, 2, 1))?;
            let image_patches =
                image_patches.reshape((batch_size, hidden_size, *h as usize, *w as usize))?;

            // Step 3: Spatial merge - first Conv2d
            let mut x_img = self.proj_conv1.forward(&image_patches)?;

            // Step 4: GELU activation
            x_img = x_img.gelu_erf()?;

            // Step 5: Second Conv2d
            x_img = self.proj_conv2.forward(&x_img)?;

            let (b, c, h_new, w_new) = x_img.dims4()?;

            // Step 6: Add image_newline token to end of each row
            let newline = self
                .image_newline
                .reshape((1, c, 1, 1))?
                .broadcast_as((b, c, h_new, 1))?;
            x_img = Tensor::cat(&[&x_img, &newline], D::Minus1)?;

            // Step 7: Reshape back to sequence format
            let x_img = x_img.reshape((b, c, h_new * (w_new + 1)))?;
            let x_img = x_img.permute((0, 2, 1))?;

            // Step 8: MLP projection
            let x_img = self.mlp.forward(&x_img)?;

            // Step 9: Add image_begin and image_end tokens
            let begin = self
                .image_begin
                .reshape((1, 1, out_channels))?
                .broadcast_as((b, 1, out_channels))?;
            let end = self
                .image_end
                .reshape((1, 1, out_channels))?
                .broadcast_as((b, 1, out_channels))?;

            let x_img = Tensor::cat(&[&begin, &x_img, &end], 1)?;

            processed_images.push(x_img);
            start_idx = end_idx;
        }

        // Step 10: Concatenate all images
        let x = if processed_images.len() == 1 {
            processed_images[0].clone()
        } else {
            Tensor::cat(&processed_images, 1)?
        };

        // Step 11: Final RMSNorm
        self.after_rms.forward(&x)
    }
}

// ============================================================================
// Complete Vision Model
// ============================================================================

/// Complete vision encoder model.
pub struct VisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("embeddings"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("layers.{}", i)))?);
        }

        // Note: weight path uses "perceive" not "merger"
        let merger = PatchMerger::new(cfg, vb.pp("perceive"))?;

        Ok(Self {
            patch_embed,
            blocks,
            merger,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `pixel_values` - Preprocessed image patches: [num_patches, 3, patch_size, patch_size]
    /// * `grid_thw` - Grid info for each image: [(t, h, w), ...]
    ///
    /// # Returns
    /// Features projected to text space: [batch, seq_len, out_hidden_size]
    pub fn forward(
        &mut self,
        pixel_values: &Tensor,
        grid_thw: &[(i64, i64, i64)],
    ) -> Result<Tensor> {
        let embeddings = self.patch_embed.forward(pixel_values, grid_thw)?;

        let mut hidden_states = embeddings;
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        self.merger.forward(&hidden_states, grid_thw)
    }
}
