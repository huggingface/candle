use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::image_encoder::ImageEncoderViT;
use super::mask_decoder::MaskDecoder;
use super::prompt_encoder::PromptEncoder;
use super::tiny_vit::{tiny_vit_5m, TinyViT};

const PROMPT_EMBED_DIM: usize = 256;
pub const IMAGE_SIZE: usize = 1024;
const VIT_PATCH_SIZE: usize = 16;
const PRED_IOU_THRESH: f32 = 0.88;
const STABILITY_SCORE_OFFSET: f32 = 1.0;
const STABILITY_SCORE_THRESHOLD: f32 = 0.95;
const MODEL_MASK_THRESHOLD: f32 = 0.0;
const CROP_NMS_THRESH: f32 = 0.7;

#[derive(Debug)]
enum ImageEncoder {
    Original(ImageEncoderViT),
    TinyViT(TinyViT),
}

impl Module for ImageEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Original(vit) => vit.forward(xs),
            Self::TinyViT(vit) => vit.forward(xs),
        }
    }
}

#[derive(Debug)]
pub struct Sam {
    image_encoder: ImageEncoder,
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
            image_encoder: ImageEncoder::Original(image_encoder),
            prompt_encoder,
            mask_decoder,
            pixel_std,
            pixel_mean,
        })
    }

    pub fn new_tiny(vb: VarBuilder) -> Result<Self> {
        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = tiny_vit_5m(vb.pp("image_encoder"))?;
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
            image_encoder: ImageEncoder::TinyViT(image_encoder),
            prompt_encoder,
            mask_decoder,
            pixel_std,
            pixel_mean,
        })
    }

    pub fn embeddings(&self, img: &Tensor) -> Result<Tensor> {
        let img = self.preprocess(img)?.unsqueeze(0)?;
        self.image_encoder.forward(&img)
    }

    pub fn forward(
        &self,
        img: &Tensor,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_c, original_h, original_w) = img.dims3()?;
        let img = self.preprocess(img)?.unsqueeze(0)?;
        let img_embeddings = self.image_encoder.forward(&img)?;
        let (low_res_mask, iou) = self.forward_for_embeddings(
            &img_embeddings,
            original_h,
            original_w,
            points,
            multimask_output,
        )?;
        let mask = low_res_mask
            .upsample_nearest2d(IMAGE_SIZE, IMAGE_SIZE)?
            .get(0)?
            .i((.., ..original_h, ..original_w))?;
        Ok((mask, iou))
    }

    /// Generate the mask and IOU predictions from some image embeddings and prompt.
    ///
    /// The prompt is specified as a list of points `(x, y, b)`. `x` and `y` are the point
    /// coordinates (between 0 and 1) and `b` is `true` for points that should be part of the mask
    /// and `false` for points that should be part of the background and so excluded from the mask.
    pub fn forward_for_embeddings(
        &self,
        img_embeddings: &Tensor,
        original_h: usize,
        original_w: usize,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let points = if points.is_empty() {
            None
        } else {
            let n_points = points.len();
            let xys = points
                .iter()
                .flat_map(|(x, y, _b)| {
                    let x = (*x as f32) * (original_w as f32);
                    let y = (*y as f32) * (original_h as f32);
                    [x, y]
                })
                .collect::<Vec<_>>();
            let labels = points
                .iter()
                .map(|(_x, _y, b)| if *b { 1f32 } else { 0f32 })
                .collect::<Vec<_>>();
            let points = Tensor::from_vec(xys, (1, n_points, 2), img_embeddings.device())?;
            let labels = Tensor::from_vec(labels, (1, n_points), img_embeddings.device())?;
            Some((points, labels))
        };
        let points = points.as_ref().map(|xy| (&xy.0, &xy.1));
        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(points, None, None)?;
        self.mask_decoder.forward(
            img_embeddings,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            multimask_output,
        )
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

    fn process_crop(
        &self,
        img: &Tensor,
        cb: CropBox,
        point_grids: &[(f64, f64)],
    ) -> Result<Vec<crate::object_detection::Bbox<Tensor>>> {
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

        let mut bboxes = Vec::new();
        for points in points.chunks(64) {
            // Run the model on this batch.
            let points_len = points.len();
            let in_points = Tensor::new(points.to_vec(), img.device())?.unsqueeze(1)?;
            let in_labels = Tensor::ones((points_len, 1), DType::F32, img.device())?;
            let (sparse_prompt_embeddings, dense_prompt_embeddings) =
                self.prompt_encoder
                    .forward(Some((&in_points, &in_labels)), None, None)?;

            let (low_res_mask, iou_predictions) = self.mask_decoder.forward(
                &img_embeddings,
                &image_pe,
                &sparse_prompt_embeddings,
                &dense_prompt_embeddings,
                /* multimask_output */ true,
            )?;
            let low_res_mask = low_res_mask.flatten(0, 1)?;
            let iou_predictions = iou_predictions.flatten(0, 1)?.to_vec1::<f32>()?;
            let dev = low_res_mask.device();

            for (i, iou) in iou_predictions.iter().enumerate() {
                // Filter by predicted IoU.
                if *iou < PRED_IOU_THRESH {
                    continue;
                }
                let low_res_mask = low_res_mask.get(i)?;

                // Calculate stability score.
                let bound = Tensor::new(MODEL_MASK_THRESHOLD + STABILITY_SCORE_OFFSET, dev)?
                    .broadcast_as(low_res_mask.shape())?;
                let intersections = low_res_mask
                    .ge(&bound)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_vec0::<f32>()?;
                let bound = Tensor::new(MODEL_MASK_THRESHOLD - STABILITY_SCORE_OFFSET, dev)?
                    .broadcast_as(low_res_mask.shape())?;
                let unions = low_res_mask
                    .ge(&bound)?
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_vec0::<f32>()?;
                let stability_score = intersections / unions;
                if stability_score < STABILITY_SCORE_THRESHOLD {
                    continue;
                }

                // Threshold masks and calculate boxes.
                let low_res_mask = low_res_mask
                    .ge(&Tensor::new(0f32, dev)?.broadcast_as(low_res_mask.shape())?)?
                    .to_dtype(DType::U32)?;
                let low_res_mask_per_x = low_res_mask.sum(0)?.to_vec1::<u32>()?;
                let low_res_mask_per_y = low_res_mask.sum(1)?.to_vec1::<u32>()?;
                let min_max_x = min_max_indexes(&low_res_mask_per_x);
                let min_max_y = min_max_indexes(&low_res_mask_per_y);
                if let Some(((x0, x1), (y0, y1))) = min_max_x.zip(min_max_y) {
                    let bbox = crate::object_detection::Bbox {
                        xmin: x0 as f32,
                        ymin: y0 as f32,
                        xmax: x1 as f32,
                        ymax: y1 as f32,
                        confidence: *iou,
                        data: low_res_mask,
                    };
                    bboxes.push(bbox);
                }
                // TODO:
                // Filter boxes that touch crop boundaries
                // Compress to RLE.
            }
        }

        let mut bboxes = vec![bboxes];
        // Remove duplicates within this crop.
        crate::object_detection::non_maximum_suppression(&mut bboxes, CROP_NMS_THRESH);

        // TODO: Return to the original image frame.
        Ok(bboxes.remove(0))
    }

    pub fn generate_masks(
        &self,
        img: &Tensor,
        points_per_side: usize,
        crop_n_layer: usize,
        crop_overlap_ratio: f64,
        crop_n_points_downscale_factor: usize,
    ) -> Result<Vec<crate::object_detection::Bbox<Tensor>>> {
        let (_c, h, w) = img.dims3()?;
        let point_grids = build_all_layer_point_grids(
            points_per_side,
            crop_n_layer,
            crop_n_points_downscale_factor,
        );
        let crop_boxes = generate_crop_boxes((h, w), crop_n_layer, crop_overlap_ratio);
        let mut bboxes = Vec::new();
        for crop_box in crop_boxes.into_iter() {
            let layer_idx = crop_box.layer_idx;
            let b = self.process_crop(img, crop_box, &point_grids[layer_idx])?;
            bboxes.extend(b)
        }
        // TODO: remove duplicates
        Ok(bboxes)
    }
}

// Return the first and last indexes i for which values[i] > 0
fn min_max_indexes(values: &[u32]) -> Option<(usize, usize)> {
    let (mut min_i, mut max_i) = (usize::MAX, usize::MIN);
    for (i, &s) in values.iter().enumerate() {
        if s == 0 {
            continue;
        }
        min_i = usize::min(i, min_i);
        max_i = usize::max(i, max_i);
    }
    if max_i < min_i {
        None
    } else {
        Some((min_i, max_i))
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
