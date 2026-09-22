//! SAM 2 mask decoder.
//!
//! Compared to the SAM 1 decoder this adds an object score token (so every token index is shifted
//! by one), an object score head, a fusion of the stride 4 and stride 8 encoder features into the
//! upscaling path, and the stability based fallback used when a single mask is requested.
use candle::{IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::transformer::TwoWayTransformer;
use super::{LayerNorm2d, Mlp};

const STABILITY_DELTA: f32 = 0.05;
const STABILITY_THRESH: f32 = 0.98;

#[derive(Debug)]
pub struct MaskDecoder {
    obj_score_token: candle_nn::Embedding,
    iou_token: candle_nn::Embedding,
    mask_tokens: candle_nn::Embedding,
    iou_prediction_head: Mlp,
    pred_obj_score_head: Mlp,
    output_upscaling_conv1: candle_nn::ConvTranspose2d,
    output_upscaling_ln: LayerNorm2d,
    output_upscaling_conv2: candle_nn::ConvTranspose2d,
    conv_s0: candle_nn::Conv2d,
    conv_s1: candle_nn::Conv2d,
    num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<Mlp>,
    transformer: TwoWayTransformer,
    span: tracing::Span,
}

impl MaskDecoder {
    pub fn new(
        transformer_dim: usize,
        num_multimask_outputs: usize,
        iou_head_depth: usize,
        iou_head_hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_mask_tokens = num_multimask_outputs + 1;
        let obj_score_token = candle_nn::embedding(1, transformer_dim, vb.pp("obj_score_token"))?;
        let iou_token = candle_nn::embedding(1, transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens =
            candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
        let iou_prediction_head = Mlp::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            /* sigmoid_output */ true,
            vb.pp("iou_prediction_head"),
        )?;
        let pred_obj_score_head = Mlp::new(
            transformer_dim,
            transformer_dim,
            1,
            3,
            false,
            vb.pp("pred_obj_score_head"),
        )?;
        let cfg = candle_nn::ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let output_upscaling_conv1 = candle_nn::conv_transpose2d(
            transformer_dim,
            transformer_dim / 4,
            2,
            cfg,
            vb.pp("output_upscaling.0"),
        )?;
        let output_upscaling_ln =
            LayerNorm2d::new(transformer_dim / 4, 1e-6, vb.pp("output_upscaling.1"))?;
        let output_upscaling_conv2 = candle_nn::conv_transpose2d(
            transformer_dim / 4,
            transformer_dim / 8,
            2,
            cfg,
            vb.pp("output_upscaling.3"),
        )?;
        let conv_s0 = candle_nn::conv2d(
            transformer_dim,
            transformer_dim / 8,
            1,
            Default::default(),
            vb.pp("conv_s0"),
        )?;
        let conv_s1 = candle_nn::conv2d(
            transformer_dim,
            transformer_dim / 4,
            1,
            Default::default(),
            vb.pp("conv_s1"),
        )?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        let vb_o = vb.pp("output_hypernetworks_mlps");
        for i in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(Mlp::new(
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false,
                vb_o.pp(i),
            )?)
        }
        let transformer = TwoWayTransformer::new(
            /* depth */ 2,
            /* embedding_dim */ transformer_dim,
            /* num_heads */ 8,
            /* mlp_dim */ 2048,
            vb.pp("transformer"),
        )?;
        let span = tracing::span!(tracing::Level::TRACE, "mask-decoder");
        Ok(Self {
            obj_score_token,
            iou_token,
            mask_tokens,
            iou_prediction_head,
            pred_obj_score_head,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            conv_s0,
            conv_s1,
            num_mask_tokens,
            output_hypernetworks_mlps,
            transformer,
            span,
        })
    }

    /// Projection of the stride 4 encoder feature, precomputed once per image.
    pub fn conv_s0(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv_s0)
    }

    /// Projection of the stride 8 encoder feature, precomputed once per image.
    pub fn conv_s1(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv_s1)
    }

    /// Returns the low resolution mask logits, the IoU predictions and the object score logits.
    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        high_res_features: (&Tensor, &Tensor),
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let (masks, iou_pred, object_score_logits) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_features,
        )?;
        let (masks, iou_pred) = if multimask_output {
            (masks.i((.., 1..))?, iou_pred.i((.., 1..))?)
        } else {
            dynamic_multimask_via_stability(&masks, &iou_pred)?
        };
        Ok((masks, iou_pred, object_score_logits))
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        (feat_s0, feat_s1): (&Tensor, &Tensor),
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // The object score token comes first, hence the index shift compared to SAM 1.
        let output_tokens = Tensor::cat(
            &[
                self.obj_score_token.embeddings(),
                self.iou_token.embeddings(),
                self.mask_tokens.embeddings(),
            ],
            0,
        )?;
        let (d1, d2) = output_tokens.dims2()?;
        let output_tokens =
            output_tokens
                .unsqueeze(0)?
                .expand((sparse_prompt_embeddings.dim(0)?, d1, d2))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;

        let src = repeat_interleave(image_embeddings, tokens.dim(0)?, 0)?;
        let src = src.broadcast_add(dense_prompt_embeddings)?;
        let pos_src = repeat_interleave(image_pe, tokens.dim(0)?, 0)?;
        let (b, c, h, w) = src.dims4()?;

        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., 1))?;
        let mask_tokens_out = hs.i((.., 2..2 + self.num_mask_tokens))?;

        // Upscale, fusing in the two high resolution encoder features.
        let src = src.transpose(1, 2)?.contiguous()?.reshape((b, c, h, w))?;
        let upscaled_embedding = self
            .output_upscaling_conv1
            .forward(&src)?
            .broadcast_add(feat_s1)?
            .apply(&self.output_upscaling_ln)?
            .gelu_erf()?
            .apply(&self.output_upscaling_conv2)?
            .broadcast_add(feat_s0)?
            .gelu_erf()?;

        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for (i, mlp) in self.output_hypernetworks_mlps.iter().enumerate() {
            hyper_in_list.push(mlp.forward(&mask_tokens_out.i((.., i))?)?)
        }
        let hyper_in = Tensor::stack(hyper_in_list.as_slice(), 1)?.contiguous()?;
        let (b, c, h, w) = upscaled_embedding.dims4()?;
        let masks = hyper_in
            .matmul(&upscaled_embedding.reshape((b, c, h * w))?)?
            .reshape((b, (), h, w))?;

        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        let object_score_logits = self.pred_obj_score_head.forward(&hs.i((.., 0))?)?;
        Ok((masks, iou_pred, object_score_logits))
    }
}

/// Fraction of the mask area that survives raising the logit threshold by `STABILITY_DELTA`.
fn stability_scores(mask_logits: &Tensor) -> Result<Vec<f32>> {
    let (b, _n, h, w) = mask_logits.dims4()?;
    let logits = mask_logits.reshape((b, h * w))?.to_vec2::<f32>()?;
    let scores = logits
        .iter()
        .map(|row| {
            let area_i = row.iter().filter(|&&v| v > STABILITY_DELTA).count();
            let area_u = row.iter().filter(|&&v| v > -STABILITY_DELTA).count();
            if area_u > 0 {
                area_i as f32 / area_u as f32
            } else {
                1.
            }
        })
        .collect();
    Ok(scores)
}

/// When a single mask is requested, SAM 2 returns the mask from token 0 unless it is unstable, in
/// which case it falls back to the best scoring of the three multimask tokens.
fn dynamic_multimask_via_stability(
    all_mask_logits: &Tensor,
    all_iou_scores: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let multimask_logits = all_mask_logits.i((.., 1..))?;
    let multimask_iou = all_iou_scores.i((.., 1..))?;
    let singlemask_logits = all_mask_logits.i((.., 0..1))?;
    let singlemask_iou = all_iou_scores.i((.., 0..1))?;

    let best_inds = multimask_iou.argmax(1)?.to_vec1::<u32>()?;
    let stability = stability_scores(&singlemask_logits)?;

    let mut masks = Vec::with_capacity(best_inds.len());
    let mut ious = Vec::with_capacity(best_inds.len());
    for (b, &best) in best_inds.iter().enumerate() {
        if stability[b] >= STABILITY_THRESH {
            masks.push(singlemask_logits.i(b..b + 1)?);
            ious.push(singlemask_iou.i(b..b + 1)?);
        } else {
            let best = best as usize;
            masks.push(multimask_logits.i((b..b + 1, best..best + 1))?);
            ious.push(multimask_iou.i((b..b + 1, best..best + 1))?);
        }
    }
    Ok((Tensor::cat(&masks, 0)?, Tensor::cat(&ious, 0)?))
}

// Equivalent to torch.repeat_interleave
fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let xs = xs.unsqueeze(dim + 1)?;
    let mut dims = xs.dims().to_vec();
    dims[dim + 1] = repeats;
    xs.broadcast_as(dims)?.flatten(dim, dim + 1)
}
