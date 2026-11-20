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

/// Audio and language branches from a multimodal checkpoint, without loading
/// unused vision weights. Audio remains conditioned on the surrounding tokens.
pub struct AudioTextModel {
    pub language_model: TextModel,
    pub audio_tower: AudioModel,
    pub embed_audio: MultimodalEmbedder,
    pub cfg: Gemma4Config,
}
impl AudioTextModel {
    pub fn new(cfg: &Gemma4Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        let audio_cfg = cfg.audio_config.as_ref().ok_or_else(|| {
            candle::Error::Msg("Gemma4 checkpoint has no audio configuration".into())
        })?;
        let vb = vb.pp("model");
        Ok(Self {
            language_model: TextModel::new(&cfg.text_config, vb.pp("language_model"))?,
            audio_tower: AudioModel::new(audio_cfg, vb.pp("audio_tower"))?,
            embed_audio: MultimodalEmbedder::new(
                audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size),
                cfg.text_config.hidden_size,
                audio_cfg.rms_norm_eps,
                vb.pp("embed_audio"),
            )?,
            cfg: cfg.clone(),
        })
    }
    pub fn encode_audio(&self, mel: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
        let (features, mask) = self.audio_tower.forward(mel, padding_mask)?;
        let mask = mask.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &padding)| (padding == 0.).then_some(idx as u32))
            .collect();
        let features = features.flatten_to(1)?;
        let indices = Tensor::new(indices, features.device())?;
        self.embed_audio
            .forward(&features.index_select(&indices, 0)?)
    }
    pub fn forward_with_audio_embeddings(
        &mut self,
        input_ids: &Tensor,
        audio: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let ids = masked_token_ids(&self.cfg, input_ids)?;
        let text = self.language_model.embed_tokens(&ids)?;
        let mask = input_ids
            .to_dtype(DType::F32)?
            .eq(self.cfg.audio_token_id as f64)?;
        let inserted = broadcast_embed_to_mask(&audio.to_dtype(text.dtype())?, &mask)?;
        let mask = mask.unsqueeze(D::Minus1)?.to_dtype(text.dtype())?;
        let text = (text.broadcast_mul(&(1. - &mask)?)? + inserted)?;
        self.language_model
            .forward_embeds_with_tokens(&text, &ids, offset)
    }
    pub fn forward(&mut self, ids: &Tensor, offset: usize) -> Result<Tensor> {
        self.language_model.forward(ids, offset)
    }
    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}

fn masked_token_ids(cfg: &Gemma4Config, input_ids: &Tensor) -> Result<Tensor> {
    let ids = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let ids: Vec<Vec<_>> = ids
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|id| {
                    if [cfg.audio_token_id, cfg.image_token_id, cfg.video_token_id]
                        .contains(&(id as usize))
                    {
                        cfg.text_config.pad_token_id as u32
                    } else {
                        id
                    }
                })
                .collect()
        })
        .collect();
    Tensor::new(ids, input_ids.device())
}

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
        input_ids.dims2()?;
        if audio_mel.is_some() != audio_mel_mask.is_some() {
            candle::bail!("Gemma4 audio features and their mask must be provided together");
        }
        if audio_mel.is_some() && (self.audio_tower.is_none() || self.embed_audio.is_none()) {
            candle::bail!("this Gemma4 checkpoint has no audio encoder");
        }
        let token_ids = self.multimodal_token_ids(input_ids)?;
        let mut input_embeds = self.language_model.embed_tokens(&token_ids)?;

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
            .forward_embeds_with_tokens(&input_embeds, &token_ids, seqlen_offset)
    }

    fn multimodal_token_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        masked_token_ids(&self.cfg, input_ids)
    }

    /// Encode audio once, returning only valid frames in row-major batch order.
    /// `audio_mel_mask` uses 1 for padding and 0 for valid frames.
    pub fn encode_audio(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<Tensor> {
        let tower = self
            .audio_tower
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("Gemma4 checkpoint has no audio encoder".into()))?;
        let embed = self.embed_audio.as_ref().ok_or_else(|| {
            candle::Error::Msg("Gemma4 checkpoint has no audio projection".into())
        })?;
        let (features, mask) = tower.forward(audio_mel, audio_mel_mask)?;
        let mask = mask.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &padding)| (padding == 0.).then_some(idx as u32))
            .collect();
        let features = features.flatten_to(1)?;
        let indices = Tensor::new(indices, features.device())?;
        embed.forward(&features.index_select(&indices, 0)?)
    }

    /// Prefill from already encoded audio. The number of placeholder tokens
    /// must exactly match the number of audio embeddings.
    pub fn forward_with_audio_embeddings(
        &mut self,
        input_ids: &Tensor,
        audio: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let ids = self.multimodal_token_ids(input_ids)?;
        let text = self.language_model.embed_tokens(&ids)?;
        let mask = input_ids
            .to_dtype(DType::F32)?
            .eq(self.cfg.audio_token_id as f64)?;
        let inserted = broadcast_embed_to_mask(&audio.to_dtype(text.dtype())?, &mask)?;
        let mask = mask.unsqueeze(D::Minus1)?.to_dtype(text.dtype())?;
        let text = (text.broadcast_mul(&(1. - &mask)?)? + inserted)?;
        self.language_model
            .forward_embeds_with_tokens(&text, &ids, seqlen_offset)
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
    let (embed_len, hidden) = embeds.dims2()?;
    let values = mask.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
    let count = values.iter().filter(|&&v| v != 0).count();
    if count != embed_len {
        candle::bail!(
            "Gemma4 modality token count {count} does not match embedding count {embed_len}"
        );
    }
    if count == 0 {
        return Tensor::zeros((b_sz, seq_len, hidden), embeds.dtype(), embeds.device());
    }
    let mut next = 0u32;
    let indices: Vec<u32> = values
        .into_iter()
        .map(|v| {
            if v != 0 {
                let idx = next;
                next += 1;
                idx
            } else {
                0
            }
        })
        .collect();
    let indices = Tensor::new(indices, embeds.device())?;
    let expanded = embeds
        .index_select(&indices, 0)?
        .reshape((b_sz, seq_len, hidden))?;
    expanded.broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(embeds.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modality_embeddings_follow_placeholder_order_across_batches() -> Result<()> {
        let dev = &candle::Device::Cpu;
        let mask = Tensor::new(&[[0u8, 1, 0, 1], [1, 0, 0, 1]], dev)?;
        let embeds = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.]], dev)?;
        assert_eq!(
            broadcast_embed_to_mask(&embeds, &mask)?.to_vec3::<f32>()?,
            vec![
                vec![vec![0., 0.], vec![1., 2.], vec![0., 0.], vec![3., 4.]],
                vec![vec![5., 6.], vec![0., 0.], vec![0., 0.], vec![7., 8.]],
            ]
        );
        assert!(broadcast_embed_to_mask(&embeds.narrow(0, 0, 3)?, &mask).is_err());
        Ok(())
    }
}
