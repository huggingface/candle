#![allow(unused)]
use super::blip_text;
use super::with_tracing::{conv2d, linear, Conv2d, Linear};
use candle::{Module, Result, Tensor};
use candle_nn::{Conv2dConfig, LayerNorm, VarBuilder};

#[derive(Debug, Clone)]
struct VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    projection_dim: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    image_size: usize,
    patch_size: usize,
    hidden_act: candle_nn::Activation,
    layer_norm_eps: f64,
}

#[derive(Debug, Clone)]
struct Config {
    text_config: blip_text::Config,
    vision_config: VisionConfig,
    projection_dim: usize,
    image_text_hidden_size: usize,
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Tensor,
    num_positions: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb.get((1, 1, cfg.hidden_size), "class_embedding")?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_patches1 = cfg.image_size / cfg.patch_size;
        let num_patches = num_patches1 * num_patches1;
        let num_positions = num_patches + 1;
        let position_embedding =
            vb.get((1, num_positions, cfg.hidden_size), "position_embedding")?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
            num_positions,
        })
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    projection: Linear,
    scale: f64,
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: MLP,
    layer_norm2: LayerNorm,
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

#[derive(Debug, Clone)]
struct VisionModel {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}

#[derive(Debug, Clone)]
struct BlipForConditionalGeneration {
    vision_model: VisionModel,
    text_decoder: blip_text::TextLMHeadModel,
}
