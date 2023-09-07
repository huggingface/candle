use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

use crate::model_image_encoder::ImageEncoderViT;
use crate::model_mask_decoder::MaskDecoder;
use crate::model_prompt_encoder::PromptEncoder;

#[derive(Debug)]
pub struct Sam {
    image_encoder: ImageEncoderViT,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    pixel_mean: Tensor,
    pixel_std: Tensor,
}

impl Sam {
    pub fn new(
        encoder_embed_dim: usize,
        encoder_depth: usize,
        encoder_num_heads: usize,
        encoder_global_attn_indexes: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        const PROMPT_EMBED_DIM: usize = 256;
        const IMAGE_SIZE: usize = 1024;
        const VIT_PATCH_SIZE: usize = 16;

        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = ImageEncoderViT::new(
            IMAGE_SIZE,
            VIT_PATCH_SIZE,
            3,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            PROMPT_EMBED_DIM,
            /* qkv_bias */ true,
            /* use_rel_pos */ true,
            /* use_abs_pos */ true,
            /* window_size */ 14,
            vb.pp("image_encoder"),
        )?;
        let prompt_encoder = PromptEncoder::new(
            PROMPT_EMBED_DIM,
            (image_embedding_size, image_embedding_size),
            (IMAGE_SIZE, IMAGE_SIZE),
            16,
            vb.pp("prompt_encoder"),
        )?;
        let mask_decoder = MaskDecoder::new(
            PROMPT_EMBED_DIM,
            /* num_multitask_outputs */ 3,
            /* iou_head_depth */ 3,
            /* iou_head_hidden_dim */ 256,
            vb.pp("mask_decoder"),
        )?;
        let pixel_mean = vb.get(3, "pixel_mean")?;
        let pixel_std = vb.get(3, "pixel_std")?;
        Ok(Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_std,
            pixel_mean,
        })
    }
}
