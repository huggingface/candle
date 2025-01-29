#[path = "config.rs"]
mod config;

use std::collections::HashMap;

// use candle_nn::Conv2d;
use candle::{Result, Tensor};
use candle_nn::{embedding, VarBuilder};
use config::MllamaVisionConfig;
use imageproc::definitions::Image;
use serde::{Deserialize, Serialize};

pub struct MllamaVisionModel {
    pub image_size: usize,
    pub patch_size: usize,
    pub max_num_tiles: usize,
    pub hidden_size: usize,
    pub num_channels: usize,
    pub intermediate_layers_indices: Vec<i32>,
    pub num_patches: usize,
    pub scale: f32,
    pub patch_embedding: candle_nn::Conv2d,
    pub class_embedding: Tensor,
    pub gated_positional_embedding: MllamaPrecomputedPositionEmbedding, // self.image_size = config.image_size
                                                                        // self.patch_size = config.patch_size
                                                                        // self.max_num_tiles = config.max_num_tiles
                                                                        // self.hidden_size = config.hidden_size
                                                                        // self.num_channels = config.num_channels
                                                                        // self.intermediate_layers_indices = config.intermediate_layers_indices

                                                                        // self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
                                                                        // self.scale = config.hidden_size**-0.5

                                                                        // self.patch_embedding = nn.Conv2d(
                                                                        //     in_channels=config.num_channels,
                                                                        //     out_channels=self.hidden_size,
                                                                        //     kernel_size=self.patch_size,
                                                                        //     stride=self.patch_size,
                                                                        //     padding="valid",
                                                                        //     bias=False,
                                                                        // )

                                                                        // self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
                                                                        // self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

                                                                        // self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
                                                                        // self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

                                                                        // # layer norms
                                                                        // self.layernorm_pre = nn.LayerNorm(self.hidden_size)
                                                                        // self.layernorm_post = nn.LayerNorm(self.hidden_size)

                                                                        // # encoders
                                                                        // self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
                                                                        // self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)
}

impl MllamaVisionModel {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let image_size = cfg.image_size;
        let patch_size = cfg.patch_size;
        let max_num_tiles = cfg.max_num_tiles;
        let hidden_size = cfg.hidden_size;
        let num_channels = cfg.num_channels;
        let intermediate_layers_indices = cfg.intermediate_layers_indices.clone();
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        let scale = (cfg.hidden_size as f32).powf(-0.5);
        let patch_embedding = candle_nn::conv2d(
            cfg.num_channels,
            hidden_size,
            patch_size,
            candle_nn::Conv2dConfig {
                padding: 0,
                stride: patch_size,
                dilation: 1 as usize,
                groups: 1 as usize,
            },
            vb.pp("patch_embedding"),
        )?;
        let class_embedding = vb.get(hidden_size, "class_embedding")?;
        let gated_positional_embedding =
            MllamaPrecomputedPositionEmbedding::new(vb.pp("gated_positional_embedding"), cfg)?;
        Ok(Self {
            image_size,
            patch_size,
            max_num_tiles,
            hidden_size,
            num_channels,
            intermediate_layers_indices,
            num_patches,
            scale,
            patch_embedding,
            class_embedding,
            gated_positional_embedding,
        })
    }
}

pub struct MllamaPrecomputedAspectRatioEmbedding {
    pub max_num_tiles: usize,
    pub hidden_size: usize,
    pub max_aspect_ratio_id: usize,
    pub is_gated: bool,
    pub embedding: candle_nn::Embedding,
    pub gate: Tensor,
}
impl MllamaPrecomputedAspectRatioEmbedding {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let max_num_tiles = cfg.max_num_tiles;
        let hidden_size = cfg.hidden_size;
        let max_aspect_ratio_id = cfg.supported_aspect_ratios.len();
        let is_gated = true;
        let embedding = candle_nn::embedding(
            max_aspect_ratio_id + 1,
            max_num_tiles * hidden_size,
            vb.pp("embedding"),
        )?;
        let gate = vb.get(1, "gate")?;
        Ok(Self {
            max_num_tiles,
            hidden_size,
            max_aspect_ratio_id,
            is_gated,
            embedding,
            gate,
        })
    }
}

pub struct MllamaPrecomputedPositionEmbedding {
    max_num_tiles: usize,
    max_aspect_ratio_id: usize,
    num_patches: usize,
    hidden_size: usize,
    scale: Tensor,
    gate: Tensor,
    embedding: Tensor,
    tile_embedding: candle_nn::Embedding,
}
impl MllamaPrecomputedPositionEmbedding {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        let scale = Tensor::new((cfg.hidden_size as f32).powf(-0.5), vb.device())?;

        let gate = vb.get(1, "gate")?;
        let embedding =
            scale.broadcast_mul(&vb.get((num_patches, cfg.hidden_size), "embedding")?)?;
        let tile_embedding = candle_nn::embedding(
            cfg.supported_aspect_ratios.len() + 1,
            cfg.max_num_tiles * num_patches * cfg.hidden_size,
            vb.pp("tile_embedding"),
        )?;

        Ok(Self {
            max_num_tiles: cfg.max_num_tiles,
            max_aspect_ratio_id: cfg.supported_aspect_ratios.len(),
            num_patches: num_patches,
            hidden_size: cfg.hidden_size,
            scale: scale,
            gate: gate,
            embedding: embedding,
            tile_embedding: tile_embedding,
        })
    }

    // pub fn forward(&self, hidden_state: Tensor, aspect_ratio_ids: Tensor) -> Result<Tensor> {
    //     // let gated_position_embedding = ((Tensor::ones(1, DType::F16, self.gate.device()) - self.gate.tanh()?) * &self.embedding)?;
    //     let x = Tensor::ones(1, DType::F64, self.gate.device())?;
    //     let y = self.gate.tanh()?;
    //     let gated_position_embedding = self.embedding.broadcast_mul(&(x - y)?)?;
    //     let mut hidden_state = (hidden_state
    //         + gated_position_embedding.reshape((1, 1, self.num_patches, self.hidden_size)))?;

    //     let mut tile_posistion_embedding = self.tile_embedding.forward(&aspect_ratio_ids)?;
    //     let batch_size = hidden_state.dim(0)?;
    //     tile_posistion_embedding = tile_posistion_embedding.reshape((
    //         batch_size,
    //         self.max_num_tiles,
    //         self.num_patches,
    //         self.hidden_size,
    //     ))?;
    //     let gated_tile_position_embedding =
    //         (self.gate.tanh()?).broadcast_mul(&tile_posistion_embedding)?;
    //     hidden_state = (hidden_state + gated_tile_position_embedding)?;
    //     Ok(hidden_state)
    // }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFPreProcessorConfig {
    pub do_convert_rgb: bool,
    pub do_normalize: bool,
    pub do_pad: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub max_image_tiles: usize,
    pub resample: usize,
    pub rescale_factor: f32,
    pub size: HashMap<String, f32>,
}
impl MllamaImageProcessor {
    pub fn from_hf_preprocessor_config(hf_preprocessor_config: &)
}
