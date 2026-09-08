//! SAM 2 prompt encoder.
//!
//! Structurally this matches the SAM 1 encoder, but SAM 2 feeds boxes through the point path with
//! labels 2 and 3 (top-left and bottom-right corner) rather than through a dedicated box path, so
//! all four point embeddings are applied here.
use candle::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::LayerNorm2d;

#[derive(Debug)]
struct PositionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}

impl PositionEmbeddingRandom {
    fn new(num_pos_feats: usize, vb: VarBuilder) -> Result<Self> {
        let positional_encoding_gaussian_matrix =
            vb.get((2, num_pos_feats), "positional_encoding_gaussian_matrix")?;
        Ok(Self {
            positional_encoding_gaussian_matrix,
        })
    }

    fn pe_encoding(&self, coords: &Tensor) -> Result<Tensor> {
        let coords = coords.affine(2., -1.)?;
        let coords = coords.broadcast_matmul(&self.positional_encoding_gaussian_matrix)?;
        let coords = (coords * (2. * std::f64::consts::PI))?;
        Tensor::cat(&[coords.sin()?, coords.cos()?], D::Minus1)
    }

    fn forward(&self, h: usize, w: usize) -> Result<Tensor> {
        let device = self.positional_encoding_gaussian_matrix.device();
        let x_embed = (Tensor::arange(0u32, w as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
        let y_embed = (Tensor::arange(0u32, h as u32, device)?.to_dtype(DType::F32)? + 0.5)?;
        let x_embed = (x_embed / w as f64)?
            .reshape((1, ()))?
            .broadcast_as((h, w))?;
        let y_embed = (y_embed / h as f64)?
            .reshape(((), 1))?
            .broadcast_as((h, w))?;
        let coords = Tensor::stack(&[&x_embed, &y_embed], D::Minus1)?;
        self.pe_encoding(&coords)?.permute((2, 0, 1))
    }

    fn forward_with_coords(
        &self,
        coords_input: &Tensor,
        image_size: (usize, usize),
    ) -> Result<Tensor> {
        let coords0 = (coords_input.narrow(D::Minus1, 0, 1)? / image_size.1 as f64)?;
        let coords1 = (coords_input.narrow(D::Minus1, 1, 1)? / image_size.0 as f64)?;
        let coords = Tensor::cat(&[&coords0, &coords1], D::Minus1)?;
        self.pe_encoding(&coords)
    }
}

#[derive(Debug)]
pub struct PromptEncoder {
    pe_layer: PositionEmbeddingRandom,
    point_embeddings: Vec<candle_nn::Embedding>,
    not_a_point_embed: candle_nn::Embedding,
    mask_downscaling_conv1: candle_nn::Conv2d,
    mask_downscaling_ln1: LayerNorm2d,
    mask_downscaling_conv2: candle_nn::Conv2d,
    mask_downscaling_ln2: LayerNorm2d,
    mask_downscaling_conv3: candle_nn::Conv2d,
    no_mask_embed: candle_nn::Embedding,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    embed_dim: usize,
    span: tracing::Span,
}

impl PromptEncoder {
    pub fn new(
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
        mask_in_chans: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_points_embeddings = 4;
        let pe_layer = PositionEmbeddingRandom::new(embed_dim / 2, vb.pp("pe_layer"))?;
        let not_a_point_embed = candle_nn::embedding(1, embed_dim, vb.pp("not_a_point_embed"))?;
        let no_mask_embed = candle_nn::embedding(1, embed_dim, vb.pp("no_mask_embed"))?;
        let cfg = candle_nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let mask_downscaling_conv1 =
            candle_nn::conv2d(1, mask_in_chans / 4, 2, cfg, vb.pp("mask_downscaling.0"))?;
        let mask_downscaling_conv2 = candle_nn::conv2d(
            mask_in_chans / 4,
            mask_in_chans,
            2,
            cfg,
            vb.pp("mask_downscaling.3"),
        )?;
        let mask_downscaling_conv3 = candle_nn::conv2d(
            mask_in_chans,
            embed_dim,
            1,
            Default::default(),
            vb.pp("mask_downscaling.6"),
        )?;
        let mask_downscaling_ln1 =
            LayerNorm2d::new(mask_in_chans / 4, 1e-6, vb.pp("mask_downscaling.1"))?;
        let mask_downscaling_ln2 =
            LayerNorm2d::new(mask_in_chans, 1e-6, vb.pp("mask_downscaling.4"))?;
        let mut point_embeddings = Vec::with_capacity(num_points_embeddings);
        let vb_e = vb.pp("point_embeddings");
        for i in 0..num_points_embeddings {
            point_embeddings.push(candle_nn::embedding(1, embed_dim, vb_e.pp(i))?)
        }
        let span = tracing::span!(tracing::Level::TRACE, "prompt-encoder");
        Ok(Self {
            pe_layer,
            point_embeddings,
            not_a_point_embed,
            mask_downscaling_conv1,
            mask_downscaling_ln1,
            mask_downscaling_conv2,
            mask_downscaling_ln2,
            mask_downscaling_conv3,
            no_mask_embed,
            image_embedding_size,
            input_image_size,
            embed_dim,
            span,
        })
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        self.pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1)?
            .unsqueeze(0)
    }

    fn embed_masks(&self, masks: &Tensor) -> Result<Tensor> {
        masks
            .apply(&self.mask_downscaling_conv1)?
            .apply(&self.mask_downscaling_ln1)?
            .gelu_erf()?
            .apply(&self.mask_downscaling_conv2)?
            .apply(&self.mask_downscaling_ln2)?
            .gelu_erf()?
            .apply(&self.mask_downscaling_conv3)
    }

    /// `points` has shape `(b, n, 2)` and `labels` shape `(b, n)`, with -1 for the padding point,
    /// 0/1 for background/foreground clicks and 2/3 for the two corners of a box.
    fn embed_points(&self, points: &Tensor, labels: &Tensor, pad: bool) -> Result<Tensor> {
        // Shift to the center of the pixel.
        let points = (points + 0.5)?;
        let device = points.device();
        let (points, labels) = if pad {
            let padding_point = Tensor::zeros((points.dim(0)?, 1, 2), DType::F32, device)?;
            let padding_label = (Tensor::ones((labels.dim(0)?, 1), DType::F32, device)? * -1f64)?;
            (
                Tensor::cat(&[&points, &padding_point], 1)?,
                Tensor::cat(&[labels, &padding_label], 1)?,
            )
        } else {
            (points, labels.clone())
        };
        let point_embedding = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size)?;
        let labels = labels.unsqueeze(2)?.broadcast_as(point_embedding.shape())?;
        let zeros = point_embedding.zeros_like()?;
        // The padding point does not get a positional encoding at all.
        let mut point_embedding = labels.lt(0f32)?.where_cond(
            &self
                .not_a_point_embed
                .embeddings()
                .broadcast_as(zeros.shape())?,
            &point_embedding,
        )?;
        for (i, embedding) in self.point_embeddings.iter().enumerate() {
            let selected = labels
                .eq(i as f32)?
                .where_cond(&embedding.embeddings().broadcast_as(zeros.shape())?, &zeros)?;
            point_embedding = (point_embedding + selected)?;
        }
        Ok(point_embedding)
    }

    pub fn forward(
        &self,
        points: Option<(&Tensor, &Tensor)>,
        masks: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let sparse_embeddings = match points {
            Some((coords, labels)) => self.embed_points(coords, labels, true)?,
            None => {
                let device = self.no_mask_embed.embeddings().device();
                Tensor::zeros((1, 0, self.embed_dim), DType::F32, device)?
            }
        };
        let dense_embeddings = match masks {
            None => {
                let emb = self.no_mask_embed.embeddings();
                emb.reshape((1, (), 1, 1))?.expand((
                    1,
                    emb.elem_count(),
                    self.image_embedding_size.0,
                    self.image_embedding_size.1,
                ))?
            }
            Some(masks) => self.embed_masks(masks)?,
        };
        Ok((sparse_embeddings, dense_embeddings))
    }
}
