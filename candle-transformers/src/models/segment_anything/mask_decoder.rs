use candle::{IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::transformer::TwoWayTransformer;

#[derive(Debug)]
struct MlpMaskDecoder {
    layers: Vec<super::Linear>,
    sigmoid_output: bool,
    span: tracing::Span,
}

impl MlpMaskDecoder {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            let layer = super::linear(vb.pp(i), in_dim, out_dim, true)?;
            layers.push(layer)
        }
        let span = tracing::span!(tracing::Level::TRACE, "mlp-mask-decoder");
        Ok(Self {
            layers,
            sigmoid_output,
            span,
        })
    }
}

impl Module for MlpMaskDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?
            }
        }
        if self.sigmoid_output {
            candle_nn::ops::sigmoid(&xs)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
pub struct MaskDecoder {
    iou_token: candle_nn::Embedding,
    mask_tokens: candle_nn::Embedding,
    iou_prediction_head: MlpMaskDecoder,
    output_upscaling_conv1: candle_nn::ConvTranspose2d,
    output_upscaling_ln: super::LayerNorm2d,
    output_upscaling_conv2: candle_nn::ConvTranspose2d,
    num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<MlpMaskDecoder>,
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
        let iou_prediction_head = MlpMaskDecoder::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            false,
            vb.pp("iou_prediction_head"),
        )?;
        let iou_token = candle_nn::embedding(1, transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens =
            candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
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
            super::LayerNorm2d::new(transformer_dim / 4, 1e-6, vb.pp("output_upscaling.1"))?;
        let output_upscaling_conv2 = candle_nn::conv_transpose2d(
            transformer_dim / 4,
            transformer_dim / 8,
            2,
            cfg,
            vb.pp("output_upscaling.3"),
        )?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        let vb_o = vb.pp("output_hypernetworks_mlps");
        for i in 0..num_mask_tokens {
            let mlp = MlpMaskDecoder::new(
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false,
                vb_o.pp(i),
            )?;
            output_hypernetworks_mlps.push(mlp)
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
            iou_token,
            mask_tokens,
            iou_prediction_head,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            num_mask_tokens,
            output_hypernetworks_mlps,
            transformer,
            span,
        })
    }

    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        )?;
        let masks = if multimask_output {
            masks.i((.., 1..))?
        } else {
            masks.i((.., 0..1))?
        };
        let iou_pred = if multimask_output {
            iou_pred.i((.., 1..))?
        } else {
            iou_pred.i((.., 0..1))?
        };
        Ok((masks, iou_pred))
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Concatenate output tokens.
        let output_tokens = Tensor::cat(
            &[self.iou_token.embeddings(), self.mask_tokens.embeddings()],
            0,
        )?;
        let (d1, d2) = output_tokens.dims2()?;
        let output_tokens =
            output_tokens
                .unsqueeze(0)?
                .expand((sparse_prompt_embeddings.dim(0)?, d1, d2))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;

        // Expand per-image data in batch direction to be per mask
        let src = repeat_interleave(image_embeddings, tokens.dim(0)?, 0)?;
        let src = src.broadcast_add(dense_prompt_embeddings)?;
        let pos_src = repeat_interleave(image_pe, tokens.dim(0)?, 0)?;
        let (b, c, h, w) = src.dims4()?;

        // Run the transformer
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., 0))?;
        let mask_tokens_out = hs.i((.., 1..1 + self.num_mask_tokens))?;

        // Upscale mask embeddings and predict masks using the masks tokens.
        let src = src.transpose(1, 2)?.reshape((b, c, h, w))?;
        let upscaled_embedding = self
            .output_upscaling_conv1
            .forward(&src)?
            .apply(&self.output_upscaling_ln)?
            .gelu()?
            .apply(&self.output_upscaling_conv2)?
            .gelu()?;
        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for (i, mlp) in self.output_hypernetworks_mlps.iter().enumerate() {
            let h = mlp.forward(&mask_tokens_out.i((.., i))?)?;
            hyper_in_list.push(h)
        }
        let hyper_in = Tensor::stack(hyper_in_list.as_slice(), 1)?.contiguous()?;
        let (b, c, h, w) = upscaled_embedding.dims4()?;
        let masks = hyper_in.matmul(&upscaled_embedding.reshape((b, c, h * w))?)?;
        let masks = masks.reshape((b, (), h, w))?;

        // Generate mask quality predictions.
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        Ok((masks, iou_pred))
    }
}

// Equivalent to torch.repeat_interleave
fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let img = img.unsqueeze(dim + 1)?;
    let mut dims = img.dims().to_vec();
    dims[dim + 1] = repeats;
    img.broadcast_as(dims)?.flatten(dim, dim + 1)
}
