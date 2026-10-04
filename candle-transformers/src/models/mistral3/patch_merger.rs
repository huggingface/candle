//! PatchMerger for Mistral3
//!
//! Merges spatial_merge_size² adjacent patches into one, reducing the number of image tokens.
//! This is a key component that differentiates Mistral3 from Pixtral LLaVA.

use candle::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use super::config::Mistral3Config;

/// PatchMerger merges adjacent patches to reduce the number of image tokens.
///
/// For spatial_merge_size=2, it merges 2x2=4 patches into 1, reducing tokens by 4x.
/// This is implemented using reshape and permute operations (equivalent to PyTorch's unfold
/// with kernel_size == stride, i.e., no overlap).
#[derive(Debug, Clone)]
pub struct PatchMerger {
    merging_layer: Linear,
    spatial_merge_size: usize,
    patch_size: usize,
}

impl PatchMerger {
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.vision_config.hidden_size;
        let spatial_merge_size = cfg.spatial_merge_size;
        let input_size = hidden_size * spatial_merge_size * spatial_merge_size;

        let merging_layer = linear_no_bias(input_size, hidden_size, vb.pp("merging_layer"))?;

        Ok(Self {
            merging_layer,
            spatial_merge_size,
            patch_size: cfg.vision_config.patch_size,
        })
    }

    /// Merge patches for a single image.
    ///
    /// Input: (h * w, hidden_size) where h, w are patch grid dimensions
    /// Output: (h/k * w/k, hidden_size) where k = spatial_merge_size
    fn merge_single_image(&self, image_tokens: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let d = image_tokens.dim(1)?; // hidden_size
        let k = self.spatial_merge_size;

        // (h * w, d) -> (h, w, d)
        let image_grid = image_tokens.reshape((h, w, d))?;

        // (h, w, d) -> (d, h, w) -> (1, d, h, w)
        let image_grid = image_grid.permute((2, 0, 1))?.unsqueeze(0)?;

        // Apply unfold (no-overlap case: kernel_size == stride)
        // This is equivalent to PyTorch's nn.functional.unfold with kernel_size=stride=k
        let new_h = h / k;
        let new_w = w / k;

        // (1, d, h, w) -> (1, d, new_h, k, new_w, k)
        let x = image_grid.reshape((1, d, new_h, k, new_w, k))?;

        // (1, d, new_h, k, new_w, k) -> (1, new_h, new_w, d, k, k)
        let x = x.permute((0, 2, 4, 1, 3, 5))?;

        // (1, new_h, new_w, d, k, k) -> (new_h * new_w, d * k * k)
        x.reshape((new_h * new_w, d * k * k))
    }

    /// Forward pass for PatchMerger.
    ///
    /// # Arguments
    /// * `image_features` - Concatenated features from all images: (total_tokens, hidden_size)
    /// * `image_sizes` - Original pixel sizes for each image: [(height, width), ...]
    ///
    /// # Returns
    /// Merged features: (total_merged_tokens, hidden_size)
    pub fn forward(
        &self,
        image_features: &Tensor,
        image_sizes: &[(usize, usize)],
    ) -> Result<Tensor> {
        // Calculate patch grid sizes for each image
        let patch_sizes: Vec<(usize, usize)> = image_sizes
            .iter()
            .map(|(h, w)| (h / self.patch_size, w / self.patch_size))
            .collect();

        // Calculate number of tokens per image
        let tokens_per_image: Vec<usize> = patch_sizes.iter().map(|(h, w)| h * w).collect();

        // Process each image separately
        let mut permuted_tensors = Vec::new();
        let mut offset = 0;

        for (idx, &(h, w)) in patch_sizes.iter().enumerate() {
            let num_tokens = tokens_per_image[idx];
            let image_tokens = image_features.narrow(0, offset, num_tokens)?;
            let grid = self.merge_single_image(&image_tokens, h, w)?;
            permuted_tensors.push(grid);
            offset += num_tokens;
        }

        // Concatenate all merged features
        let merged = Tensor::cat(&permuted_tensors, 0)?;

        // Apply linear projection
        merged.apply(&self.merging_layer)
    }
}
