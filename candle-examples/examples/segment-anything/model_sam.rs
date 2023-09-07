use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

use crate::model_image_encoder::ImageEncoderViT;
use crate::model_mask_decoder::MaskDecoder;
use crate::model_prompt_encoder::PromptEncoder;

const PROMPT_EMBED_DIM: usize = 256;
const IMAGE_SIZE: usize = 1024;
const VIT_PATCH_SIZE: usize = 16;

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
            /* global_attn_indexes */ encoder_global_attn_indexes,
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
        let pixel_mean =
            Tensor::new(&[123.675f32, 116.28, 103.53], vb.device())?.reshape((3, 1, 1))?;
        let pixel_std =
            Tensor::new(&[58.395f32, 57.12, 57.375], vb.device())?.reshape((3, 1, 1))?;
        Ok(Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_std,
            pixel_mean,
        })
    }

    pub fn forward(&self, img: &Tensor, multimask_output: bool) -> Result<(Tensor, Tensor)> {
        let img = self.preprocess(img)?.unsqueeze(0)?;
        let img_embeddings = self.image_encoder.forward(&img)?;
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(None, None, None)?;
        let (low_res_mask, iou_predictions) = self.mask_decoder.forward(
            &img_embeddings,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            multimask_output,
        )?;
        // TODO: post-processing.
        Ok((low_res_mask, iou_predictions))
    }

    fn preprocess(&self, img: &Tensor) -> Result<Tensor> {
        let (c, h, w) = img.dims3()?;
        let img = img
            .broadcast_sub(&self.pixel_mean)?
            .broadcast_div(&self.pixel_std)?;
        if h > IMAGE_SIZE || w > IMAGE_SIZE {
            candle::bail!("image is too large ({w}, {h}), maximum size {IMAGE_SIZE}")
        }
        let img = img.pad_with_zeros(1, 0, IMAGE_SIZE - h)?;
        img.pad_with_zeros(2, 0, IMAGE_SIZE - w)
    }
}
