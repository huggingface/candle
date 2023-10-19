#![allow(unused)]
use candle::{Result, Tensor};
use candle_nn::VarBuilder;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py
pub struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: candle_nn::Activation,
    layer_norm_eps: f64,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    qkv_bias: bool,
}

impl Config {
    // https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json
    pub fn vit_base_patch16_224() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: true,
        }
    }
}

struct PatchEmbeddings {
    num_patches: usize,
    projection: candle_nn::Conv2d,
}

impl PatchEmbeddings {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let image_size = cfg.image_size;
        let patch_size = cfg.patch_size;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let projection = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            patch_size,
            conv_cfg,
            vb.pp("projection"),
        )?;
        Ok(Self {
            num_patches,
            projection,
        })
    }
}

struct Embeddings {
    cls_token: Tensor,
    mask_token: Option<Tensor>,
    patch_embeddings: PatchEmbeddings,
    position_embeddings: Tensor,
}

impl Embeddings {
    fn new(cfg: &Config, use_mask_token: bool, vb: VarBuilder) -> Result<Self> {
        let cls_token = vb.get((1, 1, cfg.hidden_size), "cls_token")?;
        let mask_token = if use_mask_token {
            Some(vb.get((1, 1, cfg.hidden_size), "mask_token")?)
        } else {
            None
        };
        let patch_embeddings = PatchEmbeddings::new(cfg, vb.pp("patch_embeddings"))?;
        let num_patches = patch_embeddings.num_patches;
        let position_embeddings =
            vb.get((1, num_patches + 1, cfg.hidden_size), "position_embeddings")?;
        Ok(Self {
            cls_token,
            mask_token,
            patch_embeddings,
            position_embeddings,
        })
    }
}
