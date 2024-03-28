//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/openai/CLIP
//! https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip

use candle::{IndexOp, Result, Shape, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;
use nn::Conv2dConfig;

use super::{
    text_model::{Activation, ClipEncoder},
    EncoderConfig,
};

#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub embed_dim: usize,
    pub activation: Activation,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[allow(dead_code)]
    pub projection_dim: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

impl ClipVisionConfig {
    // The config details can be found in the "vision_config" section of this json file:
    // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
    pub fn vit_base_patch32() -> Self {
        Self {
            embed_dim: 768,
            activation: Activation::QuickGelu,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 512,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
        }
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L112
#[derive(Clone, Debug)]
struct ClipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: candle_nn::Embedding,
}

impl ClipVisionEmbeddings {
    fn new(vs: candle_nn::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        // originally nn.Parameter
        let class_embedding = if vs.contains_tensor("class_embedding") {
            vs.get(c.embed_dim, "class_embedding")?
        } else {
            Tensor::randn(0f32, 1f32, c.embed_dim, vs.device())?
        };

        let num_patches = (c.image_size / c.patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as i64, vs.device())?;

        let conv2dconfig = Conv2dConfig {
            stride: c.patch_size,
            ..Default::default()
        };
        let position_embedding =
            candle_nn::embedding(num_positions, c.embed_dim, vs.pp("position_embedding"))?;
        let patch_embedding = candle_nn::conv2d_no_bias(
            c.num_channels,
            c.embed_dim,
            c.patch_size,
            conv2dconfig,
            vs.pp("patch_embedding"),
        )?;
        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for ClipVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims();
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let class_embeds = self.class_embedding.expand(shape)?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L743
#[derive(Clone, Debug)]
pub struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    encoder: ClipEncoder,
    pre_layer_norm: candle_nn::LayerNorm,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ClipVisionTransformer {
    pub fn new(vs: candle_nn::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::new(vs.pp("embeddings"), c)?;
        let pre_layer_norm = candle_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("pre_layrnorm"))?;
        let encoder = ClipEncoder::new(vs.pp("encoder"), &EncoderConfig::Vision(c.clone()))?;
        let final_layer_norm = candle_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            pre_layer_norm,
        })
    }
}

impl Module for ClipVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;

        let encoder_outputs = self.encoder.forward(&hidden_states, None)?;
        // https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L787
        // pooled_output = encoder_outputs[:, 0, :]
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        self.final_layer_norm.forward(&pooled_output)
    }
}
