//! MultiModalProjector for Mistral3
//!
//! Projects vision features to the language model's hidden dimension.
//! Includes RMSNorm, PatchMerger, and a 2-layer MLP.

use candle::{Module, Result, Tensor};
use candle_nn::{linear_b, rms_norm, Linear, RmsNorm, VarBuilder};

use super::config::Mistral3Config;
use super::patch_merger::PatchMerger;

/// MultiModalProjector projects vision features to text embedding space.
///
/// Architecture:
/// 1. RMSNorm on vision features
/// 2. PatchMerger to reduce spatial tokens
/// 3. MLP: linear_1 -> GELU -> linear_2
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    norm: RmsNorm,
    patch_merger: PatchMerger,
    linear_1: Linear,
    act: candle_nn::Activation,
    linear_2: Linear,
}

impl MultiModalProjector {
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let vision_hidden_size = cfg.vision_config.hidden_size;
        let text_hidden_size = cfg.text_config.hidden_size;
        let rms_norm_eps = cfg.text_config.rms_norm_eps;
        let bias = cfg.multimodal_projector_bias;

        let norm = rms_norm(vision_hidden_size, rms_norm_eps, vb.pp("norm"))?;
        let patch_merger = PatchMerger::new(cfg, vb.pp("patch_merger"))?;

        let linear_1 = linear_b(
            vision_hidden_size,
            text_hidden_size,
            bias,
            vb.pp("linear_1"),
        )?;
        let linear_2 = linear_b(text_hidden_size, text_hidden_size, bias, vb.pp("linear_2"))?;

        Ok(Self {
            norm,
            patch_merger,
            linear_1,
            act: cfg.projector_hidden_act,
            linear_2,
        })
    }

    /// Forward pass for MultiModalProjector.
    ///
    /// # Arguments
    /// * `image_features` - Vision encoder output: (total_tokens, vision_hidden_size)
    /// * `image_sizes` - Original pixel sizes for each image: [(height, width), ...]
    ///
    /// # Returns
    /// Projected features: (total_merged_tokens, text_hidden_size)
    pub fn forward(
        &self,
        image_features: &Tensor,
        image_sizes: &[(usize, usize)],
    ) -> Result<Tensor> {
        // 1. RMSNorm
        let x = self.norm.forward(image_features)?;

        // 2. Patch Merger
        let x = self.patch_merger.forward(&x, image_sizes)?;

        // 3. MLP: linear_1 -> GELU -> linear_2
        let x = self.linear_1.forward(&x)?;
        let x = x.apply(&self.act)?;
        self.linear_2.forward(&x)
    }
}
