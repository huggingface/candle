//! Segment Anything Model 2 (SAM 2), image prediction.
//!
//! SAM 2 keeps the promptable segmentation interface of SAM 1 but replaces the ViT-det backbone
//! with a hierarchical Hiera trunk plus an FPN neck, and augments the mask decoder with an object
//! score head and a fusion of the high resolution encoder features.
//!
//! This module implements the single image path only (the equivalent of `SAM2ImagePredictor`);
//! the memory attention and memory encoder used for video are not included, apart from the
//! `no_mem_embed` parameter which the image path adds to the stride 16 feature map.
//!
//! - 💻 [GH Link](https://github.com/facebookresearch/sam2)
//! - 📝 [Paper](https://arxiv.org/abs/2408.00714)
//!
//! ## Example
//!
//! ```bash
//! cargo run --example segment-anything-2 --release -- \
//!     --image candle-examples/examples/yolo-v8/assets/bike.jpg \
//!     --which tiny --point 0.6,0.6
//! ```
pub use crate::models::segment_anything::{linear, LayerNorm2d};
pub use crate::models::with_tracing::Linear;

pub mod hiera;
pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod sam2;
pub mod transformer;

pub const IMAGE_SIZE: usize = 1024;
pub const PROMPT_EMBED_DIM: usize = 256;

/// Configuration of the Hiera trunk and the FPN neck. The rest of the model does not vary across
/// the released checkpoints.
#[derive(Debug, Clone)]
pub struct Config {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub stages: [usize; 4],
    pub global_att_blocks: Vec<usize>,
    pub window_pos_embed_bkg_spatial_size: (usize, usize),
    pub window_spec: [usize; 4],
    pub fpn_top_down_levels: Vec<usize>,
}

impl Config {
    pub fn tiny() -> Self {
        Self {
            embed_dim: 96,
            num_heads: 1,
            stages: [1, 2, 7, 2],
            global_att_blocks: vec![5, 7, 9],
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: [8, 4, 14, 7],
            fpn_top_down_levels: vec![2, 3],
        }
    }

    pub fn small() -> Self {
        Self {
            embed_dim: 96,
            num_heads: 1,
            stages: [1, 2, 11, 2],
            global_att_blocks: vec![7, 10, 13],
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: [8, 4, 14, 7],
            fpn_top_down_levels: vec![2, 3],
        }
    }

    pub fn base_plus() -> Self {
        Self {
            embed_dim: 112,
            num_heads: 2,
            stages: [2, 3, 16, 3],
            global_att_blocks: vec![12, 16, 20],
            window_pos_embed_bkg_spatial_size: (14, 14),
            window_spec: [8, 4, 14, 7],
            fpn_top_down_levels: vec![2, 3],
        }
    }

    pub fn large() -> Self {
        Self {
            embed_dim: 144,
            num_heads: 2,
            stages: [2, 6, 36, 4],
            global_att_blocks: vec![23, 33, 43],
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: [8, 4, 16, 8],
            fpn_top_down_levels: vec![2, 3],
        }
    }
}

/// The `MLP` helper of `sam2/modeling/sam2_utils.py`: a stack of linear layers with a ReLU in
/// between, and an optional sigmoid on the output.
#[derive(Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    sigmoid_output: bool,
}

impl Mlp {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: candle_nn::VarBuilder,
    ) -> candle::Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(linear(vb.pp(i), in_dim, out_dim, true)?)
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }
}

impl candle_nn::Module for Mlp {
    fn forward(&self, xs: &candle::Tensor) -> candle::Result<candle::Tensor> {
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = candle_nn::Module::forward(layer, &xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?
            }
        }
        if self.sigmoid_output {
            candle_nn::ops::sigmoid(&xs)
        } else {
            Ok(xs)
        }
    }
}
