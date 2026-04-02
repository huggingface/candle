//! Granite-Docling (Idefics3) model for document understanding.
//!
//! Converts document page images into structured DocTags markup.
//! Architecture: SigLIP vision encoder → pixel shuffle connector → Llama-style causal decoder.
//!
//! References:
//! - [Model Card](https://huggingface.co/ibm-granite/granite-docling-258M)
//! - [Idefics3 (HuggingFace)](https://huggingface.co/docs/transformers/model_doc/idefics3)

pub mod config;
pub mod quantized;
pub mod text;

use crate::models::siglip;
use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{linear_no_bias, VarBuilder};
use config::Config;
use text::TextModel;

/// Idefics3 pixel shuffle: spatially downsample vision tokens.
///   (B, H*W, D) → (B, H/s * W/s, D * s²)
/// Shared by both f32 and quantized paths.
pub fn pixel_shuffle(xs: &Tensor, scale_factor: usize) -> Result<Tensor> {
    let (b, seq_len, dim) = xs.dims3()?;
    let s = scale_factor;
    let h = (seq_len as f64).sqrt() as usize;
    let w = h;
    if h * w != seq_len {
        candle::bail!("pixel shuffle requires square patch grid, got {seq_len} patches (sqrt={h})");
    }
    if !h.is_multiple_of(s) {
        candle::bail!("patch grid size {h} not divisible by scale_factor {s}");
    }

    let xs = xs.reshape((b, h, w, dim))?;
    let xs = xs.reshape((b, h, w / s, dim * s))?;
    let xs = xs.permute((0, 2, 1, 3))?;
    let xs = xs.reshape((b, h / s, w / s, dim * s * s))?;
    xs.reshape((b, (h / s) * (w / s), dim * s * s))
}

/// Replace `<image>` placeholder token embeddings with projected vision features.
/// Identifies contiguous runs of image/text tokens and uses sliced Tensor::cat
/// instead of per-token allocation.
/// Shared by both f32 and quantized paths.
pub fn merge_image_tokens(
    text_embeds: &Tensor,
    image_features: &Tensor,
    input_ids: &Tensor,
    image_token_id: u32,
) -> Result<Tensor> {
    let (b, seq_len, _hidden) = text_embeds.dims3()?;
    let input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
    let text_embeds = text_embeds.to_dtype(image_features.dtype())?;

    let mut batch_results = Vec::with_capacity(b);
    for batch_idx in 0..b {
        let ids = &input_ids_vec[batch_idx * seq_len..(batch_idx + 1) * seq_len];
        let mut segments: Vec<Tensor> = Vec::new();
        let mut img_idx = 0;
        let mut pos = 0;

        while pos < seq_len {
            if ids[pos] == image_token_id {
                // Count contiguous image tokens
                let start = pos;
                while pos < seq_len && ids[pos] == image_token_id {
                    pos += 1;
                }
                let count = pos - start;
                let available = count.min(image_features.dim(1)? - img_idx);
                if available > 0 {
                    segments.push(image_features.i((batch_idx, img_idx..img_idx + available))?);
                    img_idx += available;
                }
            } else {
                // Count contiguous text tokens
                let start = pos;
                while pos < seq_len && ids[pos] != image_token_id {
                    pos += 1;
                }
                segments.push(text_embeds.i((batch_idx, start..pos))?);
            }
        }

        let batch_embeds = Tensor::cat(&segments, 0)?.unsqueeze(0)?;
        batch_results.push(batch_embeds);
    }
    Tensor::cat(&batch_results, 0)
}

// ---------------------------------------------------------------------------
// Connector (f32 path)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Connector {
    modality_projection: candle_nn::Linear,
    scale_factor: usize,
}

impl Connector {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_dim = cfg.connector_input_dim();
        let output_dim = cfg.text_config.hidden_size;
        let modality_projection =
            linear_no_bias(input_dim, output_dim, vb.pp("modality_projection.proj"))?;
        Ok(Self {
            modality_projection,
            scale_factor: cfg.scale_factor,
        })
    }
}

impl Module for Connector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = pixel_shuffle(xs, self.scale_factor)?;
        xs.apply(&self.modality_projection)
    }
}

// ---------------------------------------------------------------------------
// Idefics3ForConditionalGeneration (f32 path)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Model {
    vision_model: siglip::VisionModel,
    connector: Connector,
    text_model: TextModel,
    image_token_id: u32,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_model =
            siglip::VisionModel::new(&cfg.vision_config, false, vb.pp("model.vision_model"))?;
        let connector = Connector::new(cfg, vb.pp("model.connector"))?;
        let text_model = TextModel::new(&cfg.text_config, vb.pp("model.text_model"))?;
        Ok(Self {
            vision_model,
            connector,
            text_model,
            image_token_id: cfg.image_token_id,
        })
    }

    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_out = self.vision_model.forward(pixel_values)?;
        let connected = self.connector.forward(&vision_out)?;
        let (n, seq, hidden) = connected.dims3()?;
        connected.reshape((1, n * seq, hidden))
    }

    pub fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.clear_kv_cache();
        let image_features = self.encode_image(pixel_values)?;
        let text_embeds = self.text_model.embed_tokens().forward(input_ids)?;
        let input_embeds = merge_image_tokens(
            &text_embeds,
            &image_features,
            input_ids,
            self.image_token_id,
        )?;
        self.text_model.forward_embeds(&input_embeds)
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward(input_ids)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache();
    }
}
