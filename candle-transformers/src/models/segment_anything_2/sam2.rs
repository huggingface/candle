//! Top level SAM 2 model for single image prediction.
use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use super::image_encoder::ImageEncoder;
use super::mask_decoder::MaskDecoder;
use super::prompt_encoder::PromptEncoder;
use super::{Config, IMAGE_SIZE, PROMPT_EMBED_DIM};

/// A point prompt: `x` and `y` are normalized coordinates in `[0, 1]` and `is_foreground` is true
/// for points that should be part of the mask.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub is_foreground: bool,
}

/// A box prompt, as normalized `[0, 1]` coordinates.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// The per image features, computed once and reused across prompts.
#[derive(Debug, Clone)]
pub struct ImageFeatures {
    /// Stride 16 feature map, `(1, 256, 64, 64)`.
    pub image_embed: Tensor,
    /// Projected stride 4 feature map, `(1, 32, 256, 256)`.
    pub high_res_feat_s0: Tensor,
    /// Projected stride 8 feature map, `(1, 64, 128, 128)`.
    pub high_res_feat_s1: Tensor,
}

#[derive(Debug)]
pub struct Sam2 {
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    no_mem_embed: Tensor,
    pixel_mean: Tensor,
    pixel_std: Tensor,
}

impl Sam2 {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let image_encoder = ImageEncoder::new(cfg, vb.pp("image_encoder"))?;
        let prompt_encoder = PromptEncoder::new(
            PROMPT_EMBED_DIM,
            (IMAGE_SIZE / 16, IMAGE_SIZE / 16),
            (IMAGE_SIZE, IMAGE_SIZE),
            /* mask_in_chans */ 16,
            vb.pp("sam_prompt_encoder"),
        )?;
        let mask_decoder = MaskDecoder::new(
            PROMPT_EMBED_DIM,
            /* num_multimask_outputs */ 3,
            /* iou_head_depth */ 3,
            /* iou_head_hidden_dim */ 256,
            vb.pp("sam_mask_decoder"),
        )?;
        // Only used through `directly_add_no_mem_embed` on the image path, but leaving it out
        // silently changes the predictions.
        let no_mem_embed = vb.get((1, 1, PROMPT_EMBED_DIM), "no_mem_embed")?;
        // ImageNet statistics, expressed on the 0..255 scale.
        let pixel_mean =
            Tensor::new(&[123.675f32, 116.28, 103.53], vb.device())?.reshape((3, 1, 1))?;
        let pixel_std =
            Tensor::new(&[58.395f32, 57.12, 57.375], vb.device())?.reshape((3, 1, 1))?;
        Ok(Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            no_mem_embed,
            pixel_mean,
            pixel_std,
        })
    }

    /// Normalizes a `(3, 1024, 1024)` tensor with values in `0..255`.
    ///
    /// Unlike SAM 1, SAM 2 does not preserve the aspect ratio: the image is expected to have been
    /// resized to a square `IMAGE_SIZE` by `IMAGE_SIZE`, with no padding.
    pub fn preprocess(&self, img: &Tensor) -> Result<Tensor> {
        let (_c, h, w) = img.dims3()?;
        if h != IMAGE_SIZE || w != IMAGE_SIZE {
            candle::bail!("expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, got ({h}, {w})")
        }
        img.to_dtype(DType::F32)?
            .broadcast_sub(&self.pixel_mean)?
            .broadcast_div(&self.pixel_std)?
            .unsqueeze(0)
    }

    /// Runs the image encoder. `img` is the output of [`Self::preprocess`].
    pub fn embeddings(&self, img: &Tensor) -> Result<ImageFeatures> {
        let features = self.image_encoder.forward(img)?;
        // The high resolution projections only depend on the image, so they are hoisted out of
        // the mask decoder and computed once here.
        let high_res_feat_s0 = self.mask_decoder.conv_s0(&features[0])?;
        let high_res_feat_s1 = self.mask_decoder.conv_s1(&features[1])?;
        let image_embed =
            features[2].broadcast_add(&self.no_mem_embed.reshape((1, (), 1, 1))?)?;
        Ok(ImageFeatures {
            image_embed,
            high_res_feat_s0,
            high_res_feat_s1,
        })
    }

    /// Predicts masks for the given prompts from precomputed image features.
    ///
    /// Returns the low resolution mask logits `(1, n, 256, 256)`, the IoU predictions `(1, n)` and
    /// the object score logits `(1, 1)`. A negative object score means the model believes the
    /// prompted object is not present.
    pub fn forward_for_embeddings(
        &self,
        features: &ImageFeatures,
        points: &[Point],
        bbox: Option<BBox>,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = features.image_embed.device();
        let scale = IMAGE_SIZE as f32;
        // SAM 2 feeds boxes through the point path, as a pair of labelled corners placed before
        // the user clicks.
        let mut coords: Vec<f32> = vec![];
        let mut labels: Vec<f32> = vec![];
        if let Some(b) = bbox {
            coords.extend_from_slice(&[
                b.x0 as f32 * scale,
                b.y0 as f32 * scale,
                b.x1 as f32 * scale,
                b.y1 as f32 * scale,
            ]);
            labels.extend_from_slice(&[2., 3.]);
        }
        for p in points.iter() {
            coords.extend_from_slice(&[p.x as f32 * scale, p.y as f32 * scale]);
            labels.push(if p.is_foreground { 1. } else { 0. });
        }
        let points = if labels.is_empty() {
            None
        } else {
            let n = labels.len();
            Some((
                Tensor::from_vec(coords, (1, n, 2), device)?,
                Tensor::from_vec(labels, (1, n), device)?,
            ))
        };
        let points = points.as_ref().map(|(c, l)| (c, l));
        let (sparse, dense) = self.prompt_encoder.forward(points, None)?;
        self.mask_decoder.forward(
            &features.image_embed,
            &self.prompt_encoder.get_dense_pe()?,
            &sparse,
            &dense,
            (&features.high_res_feat_s0, &features.high_res_feat_s1),
            multimask_output,
        )
    }

    /// Convenience wrapper running the encoder and the decoder in one go, and resizing the masks
    /// back to `(original_h, original_w)`.
    ///
    /// `img` is a `(3, 1024, 1024)` tensor with values in `0..255`.
    pub fn forward(
        &self,
        img: &Tensor,
        points: &[Point],
        bbox: Option<BBox>,
        multimask_output: bool,
        original_h: usize,
        original_w: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let img = self.preprocess(img)?;
        let features = self.embeddings(&img)?;
        let (low_res_masks, iou, obj_score) =
            self.forward_for_embeddings(&features, points, bbox, multimask_output)?;
        let masks = low_res_masks
            .upsample_bilinear2d(original_h, original_w, false)?
            .i(0)?;
        Ok((masks, iou, obj_score))
    }
}
