//! Gemma 4 multimodal model (text + vision + audio).
//!
//! See:
//! - [Google Blog](https://blog.google/technology/developers/gemma-4/)

pub mod audio;
pub mod config;
pub mod multimodal_embedding;
pub mod text;
pub mod vision;

use candle::{DType, Result, Tensor, D};

use config::Gemma4Config;
use multimodal_embedding::MultimodalEmbedder;
use text::TextModel;
use vision::VisionTower;

pub use audio::AudioModel;
pub use config::{Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig};

/// Full Gemma4 multimodal model.
pub struct Model {
    pub language_model: TextModel,
    pub vision_tower: VisionTower,
    pub embed_vision: MultimodalEmbedder,
    pub audio_tower: Option<AudioModel>,
    pub embed_audio: Option<MultimodalEmbedder>,
    pub cfg: Gemma4Config,
}

impl Model {
    pub fn new(cfg: &Gemma4Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");

        let vision_tower = VisionTower::new(&cfg.vision_config, vb.pp("vision_tower"))?;

        let vis_hidden = cfg.vision_config.hidden_size;
        let text_hidden = cfg.text_config.hidden_size;
        let embed_vision = MultimodalEmbedder::new(
            vis_hidden,
            text_hidden,
            cfg.vision_config.rms_norm_eps,
            vb.pp("embed_vision"),
        )?;

        let (audio_tower, embed_audio) = if let Some(ref audio_cfg) = cfg.audio_config {
            let tower = AudioModel::new(audio_cfg, vb.pp("audio_tower"))?;
            let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
            let embed = MultimodalEmbedder::new(
                audio_hidden,
                text_hidden,
                audio_cfg.rms_norm_eps,
                vb.pp("embed_audio"),
            )?;
            (Some(tower), Some(embed))
        } else {
            (None, None)
        };

        let language_model = TextModel::new(&cfg.text_config, vb.pp("language_model"))?;

        Ok(Self {
            language_model,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
            cfg: cfg.clone(),
        })
    }

    /// Text-only forward pass.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.language_model.forward(input_ids, seqlen_offset)
    }

    /// Forward with multimodal inputs.
    ///
    /// `pixel_values`: optional batch of images, each `(1, C, H, W)`.
    /// `audio_mel`: optional `(batch, time, mel_bins)` mel spectrogram.
    /// `audio_mel_mask`: optional `(batch, time)` mask (1.0 = padding).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&[Tensor]>,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;

        // ── Vision embedding injection ──────────────────────────────────
        if let Some(pixel_values) = pixel_values {
            let image_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.image_token_id as f64)?;

            let vision_features = self.vision_tower.forward(pixel_values)?;
            let image_embeds = self
                .embed_vision
                .forward(&vision_features)?
                .to_dtype(input_embeds.dtype())?;

            // Replace image token positions with vision embeddings
            let image_embeds_flat = image_embeds.squeeze(0)?;
            let mask_expanded = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(input_embeds.dtype())?;
            let image_embeds_broadcast = broadcast_embed_to_mask(&image_embeds_flat, &image_mask)?;
            input_embeds = ((mask_expanded.clone() * image_embeds_broadcast)?
                + ((1.0 - mask_expanded)? * input_embeds)?)?;
        }

        // ── Audio embedding injection ───────────────────────────────────
        if let (
            Some(audio_mel),
            Some(audio_mel_mask),
            Some(ref audio_tower),
            Some(ref embed_audio),
        ) = (
            audio_mel,
            audio_mel_mask,
            &self.audio_tower,
            &self.embed_audio,
        ) {
            let audio_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.audio_token_id as f64)?;

            let (audio_features, enc_mask) = audio_tower.forward(audio_mel, audio_mel_mask)?;
            // Filter valid frames: where enc_mask == 0
            let valid = enc_mask.eq(0.0)?;
            let batch = audio_features.dim(0)?;
            let mut all_feats = Vec::new();
            for b in 0..batch {
                let valid_b = valid.get(b)?;
                // Count valid frames
                let valid_sum = valid_b
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;
                if valid_sum > 0 {
                    // Take the first valid_sum frames (they are contiguous after masking)
                    all_feats.push(audio_features.get(b)?.narrow(0, 0, valid_sum)?);
                }
            }
            if !all_feats.is_empty() {
                let audio_feats = Tensor::cat(&all_feats, 0)?.unsqueeze(0)?;
                let audio_embeds = embed_audio
                    .forward(&audio_feats)?
                    .to_dtype(input_embeds.dtype())?;

                let audio_embeds_flat = audio_embeds.squeeze(0)?;
                let mask_expanded = audio_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as(input_embeds.shape())?
                    .to_dtype(input_embeds.dtype())?;
                let audio_embeds_broadcast =
                    broadcast_embed_to_mask(&audio_embeds_flat, &audio_mask)?;
                input_embeds = ((mask_expanded.clone() * audio_embeds_broadcast)?
                    + ((1.0 - mask_expanded)? * input_embeds)?)?;
            }
        }

        self.language_model
            .forward_embeds(&input_embeds, seqlen_offset, b_size, seq_len)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}

/// Broadcast encoder embeddings (num_tokens, hidden) into positions marked by
/// a boolean mask (batch, seq_len), producing (batch, seq_len, hidden).
/// Token embeddings are placed sequentially where the mask is true.
fn broadcast_embed_to_mask(embeds: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len) = mask.dims2()?;
    let hidden = embeds.dim(D::Minus1)?;
    let device = embeds.device();
    let dtype = embeds.dtype();

    let mask_f32 = mask.to_dtype(DType::F32)?;
    let mut results = Vec::with_capacity(b_sz);
    let mut embed_offset = 0usize;
    let total_embeds = embeds.dim(0)?;

    for b in 0..b_sz {
        let mask_b = mask_f32.get(b)?.to_vec1::<f32>()?;
        let num_masked: usize = mask_b.iter().filter(|&&v| v > 0.5).count();

        if num_masked == 0 || embed_offset >= total_embeds {
            results.push(Tensor::zeros((1, seq_len, hidden), dtype, device)?);
            embed_offset += num_masked;
            continue;
        }

        let available = (total_embeds - embed_offset).min(num_masked);

        // Build a lookup table: index 0 → zero vector, indices 1..=available → embeds
        let zero_row = Tensor::zeros((1, hidden), dtype, device)?;
        let embed_slice = embeds.narrow(0, embed_offset, available)?;
        let lookup = Tensor::cat(&[&zero_row, &embed_slice], 0)?;

        // Map each sequence position to a lookup index:
        // non-masked → 0 (zeros), masked → 1-based sequential index
        let mut indices = vec![0u32; seq_len];
        let mut counter = 0u32;
        for (pos, &m) in mask_b.iter().enumerate() {
            if m > 0.5 {
                counter += 1;
                if counter <= available as u32 {
                    indices[pos] = counter;
                }
            }
        }

        let idx = Tensor::from_vec(indices, (seq_len,), device)?;
        results.push(lookup.index_select(&idx, 0)?.unsqueeze(0)?);

        embed_offset += num_masked;
    }

    Tensor::cat(&results, 0)
}
