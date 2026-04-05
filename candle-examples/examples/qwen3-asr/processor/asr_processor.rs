//! Top-level processor: combines text tokenization and audio feature extraction.

use anyhow::{bail, Result};

use crate::audio::{input::AudioInput, normalize::normalize_audio_input};
use crate::processor::{chat_template, feat_lengths, feature_extractor, tokenizer::Tokenizer};

#[derive(Debug, Clone)]
pub struct AsrProcessor {
    pub tokenizer: Tokenizer,
}

impl AsrProcessor {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn build_text_prompt(&self, context: &str, force_language: Option<&str>) -> String {
        chat_template::build_prompt(context, force_language)
    }

    pub fn prepare_one(&self, prompt: &str, audio: &AudioInput<'_>) -> Result<PreparedInputs> {
        let wav = normalize_audio_input(audio)?;
        let feats = feature_extractor::extract_features(&wav)?;

        // Placeholder expansion must match the audio encoder output length.
        let n_frames = feats
            .feature_attention_mask
            .iter()
            .filter(|&&x| x != 0)
            .count();
        let placeholder_len = feat_lengths::feat_extract_output_length(n_frames);

        let audio_pad_id = self.tokenizer.token_to_id(chat_template::AUDIO_PAD)?;
        let base_ids = self.tokenizer.encode(prompt)?;
        let input_ids =
            expand_audio_pad_ids_first(base_ids.as_slice(), audio_pad_id, placeholder_len);
        let attention_mask = vec![1u32; input_ids.len()];

        Ok(PreparedInputs {
            input_ids,
            attention_mask,
            input_features: feats.input_features,
            feature_attention_mask: feats.feature_attention_mask,
        })
    }

    pub fn prepare_batch(&self, items: &[(&str, &AudioInput<'_>)]) -> Result<Vec<PreparedInputs>> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // WhisperFeatureExtractor uses hop_length=160 for Qwen3-ASR.
        const HOP_LENGTH: usize = 160;

        // Normalize/resample audio first so padding is in the model's native sample rate.
        let mut wavs: Vec<Vec<f32>> = Vec::with_capacity(items.len());
        let mut max_samples: usize = 0;
        for (_, audio) in items {
            let wav = normalize_audio_input(audio)?;
            max_samples = max_samples.max(wav.len());
            wavs.push(wav);
        }

        let pad_id = self.tokenizer.token_to_id("<|endoftext|>")?;
        let audio_pad_id = self.tokenizer.token_to_id(chat_template::AUDIO_PAD)?;

        // Extract features on waveforms padded to the longest sample in the batch.
        let mut feats: Vec<feature_extractor::Features> = Vec::with_capacity(items.len());
        let mut input_ids: Vec<Vec<u32>> = Vec::with_capacity(items.len());

        for ((prompt, _), wav) in items.iter().zip(wavs.iter()) {
            let real_len = wav.len();

            let mut padded = wav.clone();
            if real_len < max_samples {
                padded.resize(max_samples, 0.0);
            }

            let mut f = feature_extractor::extract_features(&padded)?;

            // The feature extractor returns an all-ones mask for the computed frames. In the official
            // implementation, the mask is derived from the *sample-level* attention mask and then
            // rescaled by hop length. Reconstruct the equivalent mask here.
            let frames_total = padded.len() / HOP_LENGTH;
            if f.feature_attention_mask.len() != frames_total {
                bail!(
                    "internal error: feature_attention_mask len {} != frames_total {}",
                    f.feature_attention_mask.len(),
                    frames_total
                );
            }

            let mut mask: Vec<u8> = Vec::with_capacity(frames_total);
            for i in 0..frames_total {
                let sample_idx = i.saturating_mul(HOP_LENGTH);
                mask.push(u8::from(sample_idx < real_len));
            }
            f.feature_attention_mask = mask;

            let n_frames = f.feature_attention_mask.iter().filter(|&&x| x != 0).count();
            let audio_out_len = feat_lengths::feat_extract_output_length(n_frames);

            let base_ids = self.tokenizer.encode(prompt)?;
            let ids = expand_audio_pad_ids_first(base_ids.as_slice(), audio_pad_id, audio_out_len);
            feats.push(f);
            input_ids.push(ids);
        }

        // Left-pad input_ids to the longest sequence (Qwen3-ASR uses left padding).
        let max_tokens = input_ids.iter().map(Vec::len).max().unwrap_or(0);
        let mut out: Vec<PreparedInputs> = Vec::with_capacity(items.len());

        for (ids, f) in input_ids.into_iter().zip(feats.into_iter()) {
            let len = ids.len();
            if len > max_tokens {
                bail!(
                    "internal error: sequence len {} > max_tokens {}",
                    len,
                    max_tokens
                );
            }
            let pad = max_tokens - len;

            let mut padded_ids: Vec<u32> = Vec::with_capacity(max_tokens);
            padded_ids.extend(std::iter::repeat_n(pad_id, pad));
            padded_ids.extend(ids);

            let mut attn: Vec<u32> = Vec::with_capacity(max_tokens);
            attn.extend(std::iter::repeat_n(0u32, pad));
            attn.extend(std::iter::repeat_n(1u32, len));

            out.push(PreparedInputs {
                input_ids: padded_ids,
                attention_mask: attn,
                input_features: f.input_features,
                feature_attention_mask: f.feature_attention_mask,
            });
        }

        Ok(out)
    }

    pub fn require_ready(&self) -> Result<()> {
        self.tokenizer.require_loaded()
    }
}

#[derive(Debug, Clone)]
pub struct PreparedInputs {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub input_features: Vec<Vec<f32>>,
    pub feature_attention_mask: Vec<u8>,
}

pub(crate) fn expand_audio_pad_ids_first(ids: &[u32], audio_pad_id: u32, n: usize) -> Vec<u32> {
    if n == 1 {
        return ids.to_vec();
    }

    let mut out: Vec<u32> = Vec::with_capacity(ids.len().saturating_add(n.saturating_sub(1)));
    let mut expanded = false;
    for &id in ids {
        if !expanded && id == audio_pad_id {
            out.extend(std::iter::repeat_n(audio_pad_id, n));
            expanded = true;
        } else {
            out.push(id);
        }
    }
    out
}

/// Expand the `<|audio_pad|>` token to the exact number of audio encoder output positions.
///
/// The official processor replaces each occurrence of `<|audio_pad|>` with a placeholder repeated `n` times,
/// then restores the audio token. The placeholder step prevents recursive re-expansion.
pub fn expand_audio_placeholder(prompt: &str, n: usize) -> String {
    let audio_token = chat_template::AUDIO_PAD;
    let placeholder = "<|audio_placeholder|>";
    let expanded_placeholder = placeholder.repeat(n);

    // Replace only the first occurrence per audio input.
    let s = prompt.replacen(audio_token, &expanded_placeholder, 1);
    s.replace(placeholder, audio_token)
}

#[cfg(test)]
mod tests {
    use super::expand_audio_pad_ids_first;

    #[test]
    fn test_expand_audio_pad_ids_first_expands_only_first() -> anyhow::Result<()> {
        let base = vec![1u32, 2, 3, 2, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 3);
        let exp = vec![1u32, 2, 2, 2, 3, 2, 4];
        if got != exp {
            anyhow::bail!("expanded mismatch: expected={exp:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_n1_is_noop() -> anyhow::Result<()> {
        let base = vec![9u32, 8, 7, 8, 6];
        let got = expand_audio_pad_ids_first(base.as_slice(), 8, 1);
        if got != base {
            anyhow::bail!("expected noop for n=1: base={base:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_n0_removes_first() -> anyhow::Result<()> {
        let base = vec![1u32, 2, 3, 2, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 0);
        let exp = vec![1u32, 3, 2, 4];
        if got != exp {
            anyhow::bail!("removal mismatch: expected={exp:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_missing_token_noop() -> anyhow::Result<()> {
        let base = vec![1u32, 3, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 5);
        if got != base {
            anyhow::bail!("expected noop when token missing: base={base:?} got={got:?}");
        }
        Ok(())
    }
}
