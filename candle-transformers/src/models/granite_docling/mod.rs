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

// ---------------------------------------------------------------------------
// Pixel Shuffle Connector
//
// Spatially downsamples vision tokens by rearranging patch features:
//   (B, H*W, D) -> reshape -> (B, H/s * W/s, D * s^2) -> linear -> (B, N, text_hidden)
//
// For granite-docling-258M with scale_factor=4:
//   (B, 1024, 768) -> (B, 64, 12288) -> linear -> (B, 64, 576)
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

    /// Pixel shuffle matching HuggingFace Idefics3Connector.pixel_shuffle exactly:
    ///   (B, H*W, D) → view(B, H, W, D) → view(B, H, W/s, D*s)
    ///   → permute(0,2,1,3) → reshape(B, H/s, W/s, D*s²) → view(B, N, D*s²)
    fn pixel_shuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, dim) = xs.dims3()?;
        let s = self.scale_factor;
        let h = (seq_len as f64).sqrt() as usize;
        let w = h;
        assert_eq!(h * w, seq_len, "pixel shuffle requires square patch grid");
        assert_eq!(h % s, 0, "patch grid must be divisible by scale_factor");

        // (B, H*W, D) → (B, H, W, D)
        let xs = xs.reshape((b, h, w, dim))?;
        // (B, H, W, D) → (B, H, W/s, D*s) — merge s adjacent width patches
        let xs = xs.reshape((b, h, w / s, dim * s))?;
        // (B, H, W/s, D*s) → (B, W/s, H, D*s) — transpose H and W/s
        let xs = xs.permute((0, 2, 1, 3))?;
        // (B, W/s, H, D*s) → (B, H/s, W/s, D*s²) — merge s adjacent height rows
        let xs = xs.reshape((b, h / s, w / s, dim * s * s))?;
        // (B, H/s, W/s, D*s²) → (B, N, D*s²)
        xs.reshape((b, (h / s) * (w / s), dim * s * s))
    }
}

impl Module for Connector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.pixel_shuffle(xs)?;
        xs.apply(&self.modality_projection)
    }
}

// ---------------------------------------------------------------------------
// Idefics3ForConditionalGeneration
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
        let vision_model = siglip::VisionModel::new(
            &cfg.vision_config,
            false, // no pooling head — we need all patch tokens
            vb.pp("model.vision_model"),
        )?;
        let connector = Connector::new(cfg, vb.pp("model.connector"))?;
        let text_model = TextModel::new(&cfg.text_config, vb.pp("model.text_model"))?;
        Ok(Self {
            vision_model,
            connector,
            text_model,
            image_token_id: cfg.image_token_id,
        })
    }

    /// Encode images through the vision encoder and connector.
    /// Input: (num_images, 3, H, W) — multiple tiles from image splitting.
    /// Returns (1, num_images * image_seq_len, text_hidden_size) — flattened for token merging.
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_out = self.vision_model.forward(pixel_values)?; // (N, 1024, 768)
        let connected = self.connector.forward(&vision_out)?; // (N, 64, 576)
        let (n, seq, hidden) = connected.dims3()?;
        connected.reshape((1, n * seq, hidden))
    }

    /// Initial forward pass: encode image, merge with text token embeddings
    /// at <image> token positions, run through decoder.
    /// Returns logits (B, seq, vocab).
    pub fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.clear_kv_cache();

        let image_features = self.encode_image(pixel_values)?;
        let text_embeds = self.text_model.embed_tokens().forward(input_ids)?;

        // Replace <image> placeholder tokens with actual image embeddings
        let input_embeds =
            self.merge_image_tokens(&text_embeds, &image_features, input_ids)?;

        self.text_model.forward_embeds(&input_embeds)
    }

    /// Subsequent forward passes during autoregressive generation (no image).
    /// Returns logits (B, 1, vocab).
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.forward(input_ids)
    }

    /// Replace <image> token embeddings with projected vision features.
    /// Builds a new embedding tensor where <image> placeholder positions
    /// are filled with the corresponding projected vision tokens.
    fn merge_image_tokens(
        &self,
        text_embeds: &Tensor,
        image_features: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = text_embeds.dims3()?;
        let image_seq_len = image_features.dim(1)?;
        let input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let text_embeds = text_embeds.to_dtype(image_features.dtype())?;

        let mut batch_results = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let mut img_idx = 0;
            let mut tokens = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                if input_ids_vec[batch_idx * seq_len + pos] == self.image_token_id
                    && img_idx < image_seq_len
                {
                    tokens.push(image_features.i((batch_idx, img_idx))?.unsqueeze(0)?);
                    img_idx += 1;
                } else {
                    tokens.push(text_embeds.i((batch_idx, pos))?.unsqueeze(0)?);
                }
            }
            let batch_embeds = Tensor::cat(&tokens, 0)?.unsqueeze(0)?;
            batch_results.push(batch_embeds);
        }
        Tensor::cat(&batch_results, 0)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache();
    }
}
