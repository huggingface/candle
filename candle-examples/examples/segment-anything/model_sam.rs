use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::model_image_encoder::ImageEncoderViT;
use crate::model_mask_decoder::MaskDecoder;
use crate::model_prompt_encoder::PromptEncoder;

const PROMPT_EMBED_DIM: usize = 256;
pub const IMAGE_SIZE: usize = 1024;
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

    pub fn forward(
        &self,
        img: &Tensor,
        point: Option<(f64, f64)>,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_c, original_h, original_w) = img.dims3()?;
        let img = self.preprocess(img)?.unsqueeze(0)?;
        let img_embeddings = self.image_encoder.forward(&img)?;
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let points = match point {
            None => None,
            Some((x, y)) => {
                let points = Tensor::new(
                    &[[[x as f32 * original_w as f32, y as f32 * original_h as f32]]],
                    img.device(),
                )?;
                let labels = Tensor::ones((1, 1), DType::F32, img.device())?;
                Some((points, labels))
            }
        };
        let points = points.as_ref().map(|(x, y)| (x, y));
        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(points, None, None)?;
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

    pub fn unpreprocess(&self, img: &Tensor) -> Result<Tensor> {
        let img = img
            .broadcast_mul(&self.pixel_std)?
            .broadcast_add(&self.pixel_mean)?;
        img.maximum(&img.zeros_like()?)?
            .minimum(&(img.ones_like()? * 255.)?)
    }

    pub fn preprocess(&self, img: &Tensor) -> Result<Tensor> {
        let (_c, h, w) = img.dims3()?;
        let img = img
            .to_dtype(DType::F32)?
            .broadcast_sub(&self.pixel_mean)?
            .broadcast_div(&self.pixel_std)?;
        if h > IMAGE_SIZE || w > IMAGE_SIZE {
            candle::bail!("image is too large ({w}, {h}), maximum size {IMAGE_SIZE}")
        }
        let img = img.pad_with_zeros(1, 0, IMAGE_SIZE - h)?;
        img.pad_with_zeros(2, 0, IMAGE_SIZE - w)
    }

    fn process_crop(&self, img: &Tensor, cb: CropBox, point_grids: &[(f64, f64)]) -> Result<()> {
        // Crop the image and calculate embeddings.
        let img = img.i((.., cb.y0..cb.y1, cb.x0..cb.x1))?;
        let img = self.preprocess(&img)?.unsqueeze(0)?;
        let img_embeddings = self.image_encoder.forward(&img)?;

        let crop_w = cb.x1 - cb.x0;
        let crop_h = cb.y1 - cb.y0;

        // Generate masks for this crop.
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let points = point_grids
            .iter()
            .map(|&(x, y)| vec![x as f32 * crop_w as f32, y as f32 * crop_h as f32])
            .collect::<Vec<_>>();
        for points in points.chunks(64) {
            let points_len = points.len();
            let in_points = Tensor::new(points.to_vec(), img.device())?.unsqueeze(1)?;
            let in_labels = Tensor::ones((points_len, 1), DType::F32, img.device())?;
            let (sparse_prompt_embeddings, dense_prompt_embeddings) =
                self.prompt_encoder
                    .forward(Some((&in_points, &in_labels)), None, None)?;
            let (_low_res_mask, iou_predictions) = self.mask_decoder.forward(
                &img_embeddings,
                &image_pe,
                &sparse_prompt_embeddings,
                &dense_prompt_embeddings,
                /* multimask_output */ true,
            )?;

            println!("{cb:?} {iou_predictions}");
        }

        // Remove duplicates within this crop.

        // Return to the original image frame.
        Ok(())
    }

    pub fn generate_masks(
        &self,
        img: &Tensor,
        points_per_side: usize,
        crop_n_layer: usize,
        crop_overlap_ratio: f64,
        crop_n_points_downscale_factor: usize,
    ) -> Result<()> {
        let (_c, h, w) = img.dims3()?;
        let point_grids = build_all_layer_point_grids(
            points_per_side,
            crop_n_layer,
            crop_n_points_downscale_factor,
        );
        let crop_boxes = generate_crop_boxes((h, w), crop_n_layer, crop_overlap_ratio);
        for crop_box in crop_boxes.into_iter() {
            let layer_idx = crop_box.layer_idx;
            self.process_crop(img, crop_box, &point_grids[layer_idx])?
        }
        // TODO: remove duplicates
        Ok(())
    }
}

#[derive(Debug)]
struct CropBox {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    layer_idx: usize,
}

impl CropBox {
    fn new(x0: usize, y0: usize, x1: usize, y1: usize, layer_idx: usize) -> Self {
        Self {
            x0,
            y0,
            x1,
            y1,
            layer_idx,
        }
    }
}

fn generate_crop_boxes(
    (im_h, im_w): (usize, usize),
    n_layers: usize,
    overlap_ratio: f64,
) -> Vec<CropBox> {
    fn crop_len(orig_len: usize, n_crops: usize, overlap: usize) -> usize {
        f64::ceil((overlap * (n_crops - 1) + orig_len) as f64 / n_crops as f64) as usize
    }

    let short_side = usize::min(im_h, im_w);

    let mut crop_boxes = Vec::new();

    // Original image.
    crop_boxes.push(CropBox::new(0, 0, im_w, im_h, 0));

    for layer_idx in 1..=n_layers {
        let n_crops_per_side = 1 << layer_idx;
        let overlap = (overlap_ratio * short_side as f64 * 2. / n_crops_per_side as f64) as usize;
        let crop_w = crop_len(im_w, n_crops_per_side, overlap);
        let crop_h = crop_len(im_w, n_crops_per_side, overlap);

        for i_x in 0..n_crops_per_side {
            let x0 = (crop_w - overlap) * i_x;
            for i_y in 0..n_crops_per_side {
                let y0 = (crop_h - overlap) * i_y;
                let x1 = usize::min(im_w, x0 + crop_w);
                let y1 = usize::min(im_h, y0 + crop_h);
                crop_boxes.push(CropBox::new(x0, y0, x1, y1, layer_idx));
            }
        }
    }

    crop_boxes
}

// Generates a 2D grid of points evenly spaced in [0,1]x[0,1].
fn build_point_grid(n_per_side: usize) -> Vec<(f64, f64)> {
    let offset = 1f64 / (2 * n_per_side) as f64;
    let mut points = Vec::with_capacity(n_per_side * n_per_side);
    for i_x in 0..n_per_side {
        let x = offset + i_x as f64 / n_per_side as f64;
        for i_y in 0..n_per_side {
            let y = offset + i_y as f64 / n_per_side as f64;
            points.push((x, y))
        }
    }
    points
}

fn build_all_layer_point_grids(
    n_per_side: usize,
    n_layers: usize,
    scale_per_layer: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut points_by_layer = Vec::with_capacity(n_layers + 1);
    for i in 0..=n_layers {
        let n_points = n_per_side / scale_per_layer.pow(i as u32);
        points_by_layer.push(build_point_grid(n_points))
    }
    points_by_layer
}
