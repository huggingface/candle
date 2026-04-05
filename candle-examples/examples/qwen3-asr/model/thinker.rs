//! Thinker multimodal wrapper (merge audio features into text embeddings).
//!
//! Mirrors the behavior of `Qwen3ASRThinkerForConditionalGeneration` in the
//! official Python implementation.

use anyhow::{bail, Context, Result};
use candle::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::asr_config::ThinkerConfig;
use crate::model::audio_encoder::AudioTower;
use crate::model::kv_cache::KVCache;
use crate::model::thinker_text::ThinkerTextModel;

pub fn get_rope_index(attention_mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch, seq_len) = attention_mask
        .dims2()
        .context("attention_mask must be 2D")?;
    if seq_len == 0 {
        bail!("attention_mask seq_len must be > 0");
    }

    // Qwen3-ASR uses left-padding for batched generation. This routine computes
    // the same `position_ids` as the official implementation, but without
    // materializing `attention_mask` back to the host.
    //
    // For each sample, valid tokens form a suffix of the sequence:
    //   attention_mask = [0, ..., 0, 1, ..., 1]
    // For valid tokens we set position_ids to 0..len-1, while padding tokens
    // use position_id=1 (matching the Python code).
    let device = attention_mask.device();
    let seq_len_u32 =
        u32::try_from(seq_len).map_err(|_| anyhow::anyhow!("seq_len overflows u32: {seq_len}"))?;

    let mask_u8 = attention_mask.ne(0u32)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;
    let sum_mask = mask_f32.sum(1)?; // (batch,)
    let pad = (sum_mask.neg()? + seq_len as f64)?; // (batch,)

    let pos = Tensor::arange(0u32, seq_len_u32, device)?.to_dtype(DType::F32)?; // (seq_len,)
    let pos = pos.unsqueeze(0)?.broadcast_as((batch, seq_len))?;
    let pad = pad.unsqueeze(1)?.broadcast_as((batch, seq_len))?;
    let shifted = (&pos - &pad)?; // (batch, seq_len)

    let ones = Tensor::ones((batch, seq_len), DType::F32, device)?;
    let base_pos = mask_u8.where_cond(&shifted, &ones)?; // (batch, seq_len)
    let base_pos_i64 = base_pos.round()?.to_dtype(DType::I64)?;

    let position_ids = base_pos_i64
        .unsqueeze(0)?
        .broadcast_as((3usize, batch, seq_len))?;

    let max_pos = base_pos.max(1)?.maximum(1f64)?;
    let deltas = ((&max_pos + 1f64)? - sum_mask)?;
    let rope_deltas = deltas.round()?.to_dtype(DType::I64)?.unsqueeze(1)?;

    Ok((position_ids, rope_deltas))
}

fn merge_audio_features(
    input_ids: &Tensor,
    inputs_embeds: &Tensor,
    audio_token_id: u32,
    audio_features: &Tensor,
    audio_placeholder_count: usize,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2().context("input_ids must be 2D")?;
    let (b2, s2, hidden) = inputs_embeds.dims3().context("inputs_embeds must be 3D")?;
    if (b2, s2) != (batch, seq_len) {
        bail!(
            "inputs_embeds dims mismatch: expected=({batch},{seq_len},*), got=({b2},{s2},{hidden})"
        );
    }
    let (num_audio, audio_hidden) = audio_features
        .dims2()
        .context("audio_features must be 2D")?;
    if audio_hidden != hidden {
        bail!("audio_features hidden mismatch: expected={hidden}, got={audio_hidden}");
    }

    let total = batch
        .checked_mul(seq_len)
        .ok_or_else(|| anyhow::anyhow!("input size overflow: batch={batch} seq_len={seq_len}"))?;
    if audio_placeholder_count > total {
        bail!(
            "audio placeholder count exceeds total tokens: placeholders={audio_placeholder_count} total_tokens={total}"
        );
    }
    if audio_placeholder_count != num_audio {
        bail!(
            "audio placeholder count mismatch: placeholders={audio_placeholder_count}, audio_features.shape[0]={num_audio}"
        );
    }
    if audio_placeholder_count == 0 {
        return Ok(inputs_embeds.clone());
    }

    let mask_u8 = input_ids.eq(audio_token_id)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;

    // Compute placeholder indices in (batch, seq) order using only device ops.
    //
    // `mask_f32` is {0,1}. We compute:
    // - per-row cumulative counts (within each sample)
    // - per-row offsets (total placeholders in prior rows)
    // and combine them into a global audio feature index per token position.
    let in_row = ((mask_f32.cumsum(1)? - 1f64)? * &mask_f32)?;
    let row_counts = mask_f32.sum(1)?;
    let offsets = (row_counts.cumsum(0)? - &row_counts)?;
    let offsets = offsets.unsqueeze(1)?.broadcast_as((batch, seq_len))?;
    let idx = ((&offsets + &in_row)? * &mask_f32)?.reshape((total,))?;

    let idx = idx.to_dtype(DType::U32)?;
    let audio_features = audio_features.to_dtype(inputs_embeds.dtype())?;
    let audio_at_pos = audio_features
        .embedding(&idx)?
        .reshape((batch, seq_len, hidden))?;

    let cond = mask_u8
        .unsqueeze(2)?
        .broadcast_as((batch, seq_len, hidden))?;
    Ok(cond.where_cond(&audio_at_pos, inputs_embeds)?)
}

#[derive(Debug, Clone)]
pub struct ThinkerForConditionalGeneration {
    audio_tower: AudioTower,
    text_model: ThinkerTextModel,
    lm_head: Linear,
    audio_token_id: u32,
}

impl ThinkerForConditionalGeneration {
    pub fn audio_token_id(&self) -> u32 {
        self.audio_token_id
    }

    pub fn audio_config(&self) -> &crate::config::AudioEncoderConfig {
        self.audio_tower.config()
    }

    pub fn audio_uses_flash_attn(&self) -> bool {
        self.audio_tower.uses_flash_attn()
    }

    pub fn load(
        cfg: &ThinkerConfig,
        vb: VarBuilder,
        device: &candle::Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let audio_token_id = cfg
            .audio_token_id
            .context("thinker_config.audio_token_id is required")?;

        let is_forced_aligner = cfg
            .model_type
            .as_deref()
            .unwrap_or_default()
            .contains("forced_aligner");
        let lm_head_out = if is_forced_aligner {
            cfg.classify_num
                .context("thinker_config.classify_num is required for forced aligner")?
        } else {
            cfg.text_config.vocab_size
        };

        let audio_tower =
            AudioTower::load(&cfg.audio_config, vb.pp("audio_tower"), use_flash_attn)?;
        let text_model =
            ThinkerTextModel::load(&cfg.text_config, vb.pp("model"), device, use_flash_attn)?;

        let lm_head = if cfg.text_config.tie_word_embeddings {
            if is_forced_aligner {
                bail!("forced aligner does not support tie_word_embeddings=true");
            }
            Linear::new(text_model.embed_tokens_weight().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.text_config.hidden_size, lm_head_out, vb.pp("lm_head"))?
        };

        Ok(Self {
            audio_tower,
            text_model,
            lm_head,
            audio_token_id,
        })
    }

    pub fn get_audio_features(
        &self,
        input_features: &Tensor,
        feature_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.audio_tower
            .forward(input_features, feature_attention_mask)
            .map_err(Into::into)
    }

    pub fn get_audio_features_with_lens(
        &self,
        input_features: &Tensor,
        feature_lens: &[usize],
    ) -> Result<Tensor> {
        self.audio_tower
            .forward_with_lens(input_features, feature_lens)
            .map_err(Into::into)
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model
            .embed_tokens()
            .forward(input_ids)
            .map_err(Into::into)
    }

    pub fn inputs_embeds_with_audio_features(
        &self,
        input_ids: &Tensor,
        audio_features: Option<&Tensor>,
        audio_placeholder_count: usize,
    ) -> Result<Tensor> {
        let inputs_embeds = self.embed_tokens(input_ids)?;
        match audio_features {
            None => Ok(inputs_embeds),
            Some(af) => Ok(merge_audio_features(
                input_ids,
                &inputs_embeds,
                self.audio_token_id,
                af,
                audio_placeholder_count,
            )?),
        }
    }

    pub fn forward_embeds_with_kv_cache(
        &self,
        attention_mask: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let hidden_states = self.text_model.forward_with_kv_cache(
            attention_mask,
            position_ids,
            inputs_embeds,
            kv_cache,
        )?;
        Ok(self.lm_head.forward(&hidden_states)?)
    }

    pub fn forward_with_audio_features(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        audio_features: Option<&Tensor>,
        audio_placeholder_count: usize,
    ) -> Result<Tensor> {
        let inputs_embeds = self.inputs_embeds_with_audio_features(
            input_ids,
            audio_features,
            audio_placeholder_count,
        )?;

        let (position_ids, _rope_deltas) = get_rope_index(attention_mask)?;

        let hidden_states =
            self.text_model
                .forward(Some(attention_mask), &position_ids, &inputs_embeds)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        input_features: Option<&Tensor>,
        feature_attention_mask: Option<&Tensor>,
        audio_placeholder_count: usize,
    ) -> Result<Tensor> {
        let audio_features = match input_features {
            None => None,
            Some(feats) => Some(self.get_audio_features(feats, feature_attention_mask)?),
        };
        self.forward_with_audio_features(
            input_ids,
            attention_mask,
            audio_features.as_ref(),
            audio_placeholder_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{get_rope_index, merge_audio_features};
    use candle::{Device, Tensor};

    fn reference_rope(attn: &[u32], batch: usize, seq_len: usize) -> (Vec<Vec<i64>>, Vec<i64>) {
        let mut base: Vec<Vec<i64>> = vec![vec![0i64; seq_len]; batch];
        let mut deltas: Vec<i64> = vec![0i64; batch];

        for b in 0..batch {
            let start = b.saturating_mul(seq_len);
            let end = start.saturating_add(seq_len);
            let row = attn.get(start..end).unwrap_or(&[]);

            let mut seen = 0i64;
            let mut max_pos = 1i64;
            let mut sum_mask = 0i64;

            for (i, &m) in row.iter().enumerate() {
                if m != 0 {
                    let pos = seen;
                    seen = seen.saturating_add(1);
                    sum_mask = sum_mask.saturating_add(1);
                    max_pos = max_pos.max(pos);
                    if let Some(dst) = base.get_mut(b).and_then(|r| r.get_mut(i)) {
                        *dst = pos;
                    }
                } else {
                    max_pos = max_pos.max(1);
                    if let Some(dst) = base.get_mut(b).and_then(|r| r.get_mut(i)) {
                        *dst = 1;
                    }
                }
            }

            if let Some(dst) = deltas.get_mut(b) {
                *dst = max_pos.saturating_add(1).saturating_sub(sum_mask);
            }
        }

        (base, deltas)
    }

    #[test]
    fn test_merge_audio_features_replaces_placeholders_row_major() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let audio_token_id = 99u32;

        let input_ids = Tensor::from_vec(
            vec![
                1u32,
                audio_token_id,
                2u32,
                audio_token_id,
                audio_token_id,
                3u32,
                4u32,
                5u32,
            ],
            (2usize, 4usize),
            &device,
        )?;

        let mut embeds = Vec::new();
        for b in 0..2usize {
            for s in 0..4usize {
                for h in 0..3usize {
                    embeds.push((b * 100 + s * 10 + h) as f32);
                }
            }
        }
        let inputs_embeds = Tensor::from_vec(embeds, (2usize, 4usize, 3usize), &device)?;

        let audio_features = Tensor::from_vec(
            vec![
                1000f32, 1001f32, 1002f32, // a0
                2000f32, 2001f32, 2002f32, // a1
                3000f32, 3001f32, 3002f32, // a2
            ],
            (3usize, 3usize),
            &device,
        )?;

        let out = merge_audio_features(
            &input_ids,
            &inputs_embeds,
            audio_token_id,
            &audio_features,
            3,
        )?;
        let got = out.to_vec3::<f32>()?;

        let mut want = inputs_embeds.to_vec3::<f32>()?;
        want[0][1] = vec![1000f32, 1001f32, 1002f32];
        want[0][3] = vec![2000f32, 2001f32, 2002f32];
        want[1][0] = vec![3000f32, 3001f32, 3002f32];

        if got != want {
            anyhow::bail!("unexpected merged embeds: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_audio_features_rejects_placeholder_count_mismatch() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let audio_token_id = 42u32;

        let input_ids = Tensor::from_vec(
            vec![1u32, audio_token_id, 2u32, 3u32],
            (1usize, 4usize),
            &device,
        )?;
        let inputs_embeds = Tensor::zeros((1usize, 4usize, 2usize), candle::DType::F32, &device)?;
        let audio_features = Tensor::zeros((0usize, 2usize), candle::DType::F32, &device)?;

        let err = merge_audio_features(
            &input_ids,
            &inputs_embeds,
            audio_token_id,
            &audio_features,
            1,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected merge_audio_features to fail"))?;
        let msg = format!("{err:#}");
        if !msg.contains("placeholder count mismatch") {
            anyhow::bail!("expected placeholder mismatch error, got: {msg}");
        }
        Ok(())
    }

    #[test]
    fn test_merge_audio_features_allows_empty_when_no_placeholders() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let audio_token_id = 7u32;

        let input_ids = Tensor::from_vec(vec![1u32, 2u32, 3u32], (1usize, 3usize), &device)?;
        let inputs_embeds = Tensor::from_vec(
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
            (1usize, 3usize, 2usize),
            &device,
        )?;
        let audio_features = Tensor::zeros((0usize, 2usize), candle::DType::F32, &device)?;

        let out = merge_audio_features(
            &input_ids,
            &inputs_embeds,
            audio_token_id,
            &audio_features,
            0,
        )?;
        let got = out.to_vec3::<f32>()?;
        let want = inputs_embeds.to_vec3::<f32>()?;
        if got != want {
            anyhow::bail!("expected output to equal inputs when no placeholders");
        }
        Ok(())
    }

    #[test]
    fn test_get_rope_index_matches_reference_for_left_padded_masks() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let batch = 2usize;
        let seq_len = 6usize;

        // Left padding: 0s then 1s.
        let attn: Vec<u32> = vec![0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1];
        let attn_t = Tensor::from_vec(attn.clone(), (batch, seq_len), &device)?;

        let (pos_ids, rope_deltas) = get_rope_index(&attn_t)?;

        let got_pos = pos_ids.to_vec3::<i64>()?;
        let got_delta = rope_deltas.to_vec2::<i64>()?;

        let (want_base, want_deltas) = reference_rope(attn.as_slice(), batch, seq_len);

        if got_pos.len() != 3 {
            anyhow::bail!("expected 3 modalities, got {}", got_pos.len());
        }
        for (m, modality) in got_pos.iter().enumerate() {
            for b in 0..batch {
                let got_row = modality.get(b).cloned().unwrap_or_default();
                let want_row = want_base.get(b).cloned().unwrap_or_default();
                if got_row != want_row {
                    anyhow::bail!(
                        "position_ids mismatch modality={m} batch={b}: got={got_row:?} want={want_row:?}"
                    );
                }
            }
        }

        if got_delta.len() != batch {
            anyhow::bail!(
                "rope_deltas batch mismatch: expected={batch}, got={}",
                got_delta.len()
            );
        }
        for b in 0..batch {
            let got = got_delta.get(b).and_then(|r| r.first()).copied();
            let want = want_deltas.get(b).copied();
            if got != want {
                anyhow::bail!("rope_deltas mismatch batch={b}: got={got:?} want={want:?}");
            }
        }
        Ok(())
    }
}
