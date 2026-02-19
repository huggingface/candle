//! Streaming transcription pipeline.
//!
//! This module mirrors the official Python streaming semantics:
//! - buffer arbitrary audio inputs,
//! - consume fixed-size chunks,
//! - re-feed all accumulated audio so far,
//! - update prompt as `prompt_raw + prefix` with rollback.
//!
//! Optional rolling-window mode bounds compute for long-running streams:
//! - `StreamOptions.audio_window_sec` limits the audio context re-fed each step.
//! - `StreamOptions.text_window_tokens` limits the text context re-fed each step
//!   while preserving a committed transcript prefix in the output.

use anyhow::{bail, Result};
use candle::{Device, Tensor};

use crate::audio::input::AudioInput;
use crate::audio::normalize::normalize_audio_input;
use crate::audio::SAMPLE_RATE_HZ;
use crate::inference::transcribe::build_eos_token_ids;
use crate::inference::types::{AsrTranscription, StreamOptions};
use crate::inference::utils::{normalize_language_name, parse_asr_output, validate_language};
use crate::model::generation::greedy_generate_cached;
use crate::model::AsrModel;
use crate::processor::feat_lengths;
use crate::processor::feature_extractor::StreamingFeatureExtractor;
use crate::processor::AsrProcessor;

#[derive(Debug, Clone)]
struct AudioFeaturesCache {
    frames: usize,
    wav_len_samples: usize,
    compressed_max_log: f32,
    audio_features: Tensor,
}

#[derive(Debug, Clone)]
struct RollingTextState {
    committed: String,
    tail_ids: Vec<u32>,
    keep_tail_tokens: usize,
}

fn recompute_from_for_stream_append(cache_frames: usize, segment_frames: usize) -> usize {
    if segment_frames == 0 {
        return 0;
    }
    let changed_start_frame = cache_frames.saturating_sub(1);
    (changed_start_frame / segment_frames) * segment_frames
}

#[derive(Debug)]
pub struct AsrStream<'a> {
    model: &'a AsrModel,
    processor: &'a AsrProcessor,
    device: &'a Device,
    opts: StreamOptions,

    chunk_size_samples: usize,
    chunk_id: usize,

    buffer: Vec<f32>,
    buffer_offset: usize,
    features: StreamingFeatureExtractor,
    audio_cache: Option<AudioFeaturesCache>,

    prompt_raw: String,
    raw_decoded: String,
    rolling_text: Option<RollingTextState>,
    eos_token_ids: Vec<u32>,

    last: Option<AsrTranscription>,
}

impl<'a> AsrStream<'a> {
    fn audio_window_frames(&self) -> Result<Option<usize>> {
        let Some(sec) = self.opts.audio_window_sec else {
            return Ok(None);
        };
        if !sec.is_finite() || sec <= 0.0 {
            bail!("audio_window_sec must be finite and > 0");
        }

        // StreamingFeatureExtractor uses hop_length=160 at 16kHz, i.e. 100 frames/sec.
        let frames_f = f64::from(sec) * 100.0;
        if !frames_f.is_finite() || frames_f <= 0.0 {
            bail!("audio_window_sec produces invalid frame count: sec={sec}");
        }
        if frames_f > (usize::MAX as f64) {
            bail!("audio_window_sec produces too many frames: sec={sec}");
        }

        Ok(Some((frames_f.round() as usize).max(1)))
    }

    fn frames_range_for_step(&self, frames_total: usize) -> Result<std::ops::Range<usize>> {
        let Some(window_frames) = self.audio_window_frames()? else {
            return Ok(0..frames_total);
        };

        if frames_total == 0 {
            return Ok(0..0);
        }

        let window_frames = window_frames.min(frames_total).max(1);
        Ok(frames_total.saturating_sub(window_frames)..frames_total)
    }

    fn feature_dims(&self) -> Result<(usize, usize)> {
        let mel = self.features.input_features().len();
        if mel == 0 {
            bail!("internal error: empty input_features");
        }

        let frames = self
            .features
            .input_features()
            .first()
            .map(|r| r.len())
            .ok_or_else(|| anyhow::anyhow!("internal error: missing first input_features row"))?;

        for (i, row) in self.features.input_features().iter().enumerate() {
            if row.len() != frames {
                bail!(
                    "internal error: input_features row {i} length mismatch: expected={frames} got={}",
                    row.len()
                );
            }
        }

        let mask_len = self.features.feature_attention_mask().len();
        if mask_len != frames {
            bail!(
                "internal error: feature_attention_mask length mismatch: expected={frames} got={mask_len}"
            );
        }

        Ok((mel, frames))
    }

    fn build_feature_tensors(
        &self,
        frames_range: std::ops::Range<usize>,
    ) -> Result<(Tensor, usize)> {
        let (mel, frames_total) = self.feature_dims()?;
        if frames_range.start > frames_range.end {
            bail!(
                "invalid feature range: start {} > end {}",
                frames_range.start,
                frames_range.end
            );
        }
        if frames_range.end > frames_total {
            bail!(
                "invalid feature range: end {} > frames_total {}",
                frames_range.end,
                frames_total
            );
        }

        let frames = frames_range.end - frames_range.start;
        let flat_len = mel.checked_mul(frames).ok_or_else(|| {
            anyhow::anyhow!("input_features size overflow: mel={mel} frames={frames}")
        })?;
        let mut feats_flat: Vec<f32> = Vec::with_capacity(flat_len);
        for (i, row) in self.features.input_features().iter().enumerate() {
            let slice = row.get(frames_range.start..frames_range.end).ok_or_else(|| {
                anyhow::anyhow!(
                    "internal error: input_features slice out of bounds for row {i}: start={} end={} len={}",
                    frames_range.start,
                    frames_range.end,
                    row.len()
                )
            })?;
            feats_flat.extend_from_slice(slice);
        }

        let input_features = Tensor::from_vec(feats_flat, (1usize, mel, frames), self.device)?;

        let mask = self.features.feature_attention_mask();
        let mask_slice = mask.get(frames_range.start..frames_range.end).ok_or_else(|| {
            anyhow::anyhow!(
                "internal error: feature_attention_mask slice out of bounds: start={} end={} len={}",
                frames_range.start,
                frames_range.end,
                mask.len()
            )
        })?;
        let feature_len = mask_slice.iter().filter(|&&x| x != 0).count();

        Ok((input_features, feature_len))
    }

    fn compute_audio_features(&self, frames_range: std::ops::Range<usize>) -> Result<Tensor> {
        let (input_features, feature_len) = self.build_feature_tensors(frames_range)?;
        self.model
            .thinker
            .get_audio_features_with_lens(&input_features, std::slice::from_ref(&feature_len))
    }

    fn audio_cache_segment_frames(&self) -> Result<Option<usize>> {
        if !self.model.thinker.audio_uses_flash_attn() {
            return Ok(None);
        }

        let cfg = self.model.thinker.audio_config();
        let window_frames = cfg
            .n_window
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("audio_config.n_window overflows when doubled"))?;
        if window_frames == 0 {
            bail!("audio_config.n_window must be > 0");
        }

        let ratio = cfg.n_window_infer / window_frames;
        if ratio == 0 {
            return Ok(None);
        }

        let segment_frames = window_frames.checked_mul(ratio).ok_or_else(|| {
            anyhow::anyhow!(
                "audio segment_frames overflow: window_frames={window_frames} ratio={ratio}"
            )
        })?;
        if segment_frames == 0 {
            return Ok(None);
        }

        Ok(Some(segment_frames))
    }

    fn audio_features_for_step(&mut self) -> Result<Tensor> {
        let (_mel, frames_total) = self.feature_dims()?;
        if frames_total == 0 {
            bail!("internal error: empty features (frames_total=0)");
        }

        let frames_range = self.frames_range_for_step(frames_total)?;
        let frames_window = frames_range.end.saturating_sub(frames_range.start);
        if frames_window == 0 {
            bail!("internal error: frames_window is 0");
        }

        let compressed_max_log = self.features.compressed_max_log();
        let wav_len_samples = self.features.wav_len();
        let expected_tokens = feat_lengths::feat_extract_output_length(frames_window);

        let windowed = frames_range.start != 0 || frames_range.end != frames_total;
        if windowed {
            // Windowing shifts the feature index space each step, so caching based on absolute
            // `frames_total` no longer applies.
            self.audio_cache = None;
            let audio_features = self.compute_audio_features(frames_range)?;
            let (rows, _cols) = audio_features.dims2()?;
            if rows != expected_tokens {
                bail!(
                    "internal error: audio_features length mismatch (windowed): expected={expected_tokens} got={rows}"
                );
            }
            return Ok(audio_features);
        }

        let Some(segment_frames) = self.audio_cache_segment_frames()? else {
            let audio_features = self.compute_audio_features(frames_range)?;
            let (rows, _cols) = audio_features.dims2()?;
            if rows != expected_tokens {
                bail!(
                    "internal error: audio_features length mismatch: expected={expected_tokens} got={rows}"
                );
            }
            return Ok(audio_features);
        };

        if let Some(cache) = self.audio_cache.as_ref() {
            if cache.frames == frames_total
                && cache.wav_len_samples == wav_len_samples
                && cache.compressed_max_log == compressed_max_log
            {
                return Ok(cache.audio_features.clone());
            }
        }

        let (audio_features, new_cache) = match self.audio_cache.as_ref() {
            None => {
                let audio_features = self.compute_audio_features(0..frames_total)?;
                (
                    audio_features.clone(),
                    Some(AudioFeaturesCache {
                        frames: frames_total,
                        wav_len_samples,
                        compressed_max_log,
                        audio_features,
                    }),
                )
            }
            Some(cache) => {
                if cache.compressed_max_log != compressed_max_log || cache.frames > frames_total {
                    let audio_features = self.compute_audio_features(0..frames_total)?;
                    (
                        audio_features.clone(),
                        Some(AudioFeaturesCache {
                            frames: frames_total,
                            wav_len_samples,
                            compressed_max_log,
                            audio_features,
                        }),
                    )
                } else {
                    let recompute_from =
                        recompute_from_for_stream_append(cache.frames, segment_frames);
                    if recompute_from == 0 {
                        let audio_features = self.compute_audio_features(0..frames_total)?;
                        (
                            audio_features.clone(),
                            Some(AudioFeaturesCache {
                                frames: frames_total,
                                wav_len_samples,
                                compressed_max_log,
                                audio_features,
                            }),
                        )
                    } else {
                        if recompute_from > frames_total {
                            bail!(
                                "internal error: recompute_from {} > frames_total {}",
                                recompute_from,
                                frames_total
                            );
                        }

                        let prefix_tokens =
                            feat_lengths::feat_extract_output_length(recompute_from);
                        let (cached_tokens, _cols) = cache.audio_features.dims2()?;
                        if prefix_tokens > cached_tokens {
                            bail!(
                                "internal error: prefix_tokens {prefix_tokens} exceed cached_tokens {cached_tokens}"
                            );
                        }

                        let prefix = cache.audio_features.narrow(0, 0, prefix_tokens)?;
                        let tail = self.compute_audio_features(recompute_from..frames_total)?;
                        let audio_features = Tensor::cat(&[&prefix, &tail], 0)?;

                        (
                            audio_features.clone(),
                            Some(AudioFeaturesCache {
                                frames: frames_total,
                                wav_len_samples,
                                compressed_max_log,
                                audio_features,
                            }),
                        )
                    }
                }
            }
        };

        let (rows, _cols) = audio_features.dims2()?;
        if rows != expected_tokens {
            bail!(
                "internal error: audio_features length mismatch after caching: expected={expected_tokens} got={rows}"
            );
        }

        self.audio_cache = new_cache;
        Ok(audio_features)
    }

    fn generate_ids(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
        audio_features: &Tensor,
    ) -> Result<Vec<u32>> {
        greedy_generate_cached(
            &self.model.thinker,
            self.device,
            input_ids,
            attention_mask,
            Some(audio_features),
            self.opts.max_new_tokens,
            self.eos_token_ids.as_slice(),
        )
    }

    fn build_prefix(&self) -> Result<String> {
        if self.chunk_id < self.opts.unfixed_chunk_num {
            return Ok(String::new());
        }

        let cur_ids = self.processor.tokenizer.encode(self.raw_decoded.as_str())?;

        let mut k = self.opts.unfixed_token_num;
        loop {
            let end_idx = cur_ids.len().saturating_sub(k);
            let prefix = match cur_ids.get(..end_idx) {
                Some(ids) if !ids.is_empty() => self.processor.tokenizer.decode(ids)?,
                _ => String::new(),
            };

            if !prefix.contains('\u{FFFD}') {
                return Ok(prefix);
            }

            if end_idx == 0 {
                return Ok(String::new());
            }

            k = k.saturating_add(1);
        }
    }

    fn build_finish_prefix(&self) -> Result<String> {
        if self.chunk_id < self.opts.unfixed_chunk_num {
            return Ok(String::new());
        }

        let cur_ids = self.processor.tokenizer.encode(self.raw_decoded.as_str())?;
        let mut end_idx = cur_ids.len().saturating_sub(self.opts.unfixed_token_num);
        if end_idx == 0 && !cur_ids.is_empty() {
            end_idx = 1;
        }

        match cur_ids.get(..end_idx) {
            Some(ids) if !ids.is_empty() => self.processor.tokenizer.decode(ids),
            _ => Ok(String::new()),
        }
    }

    fn decode_step(&mut self, prefix: String) -> Result<AsrTranscription> {
        let audio_features = self.audio_features_for_step()?;
        let (placeholder_len, _hidden) = audio_features.dims2()?;

        let mut prompt = self.prompt_raw.clone();
        prompt.push_str(prefix.as_str());

        let base_ids = self.processor.tokenizer.encode(prompt.as_str())?;
        let input_ids = crate::processor::asr_processor::expand_audio_pad_ids_first(
            base_ids.as_slice(),
            self.model.thinker.audio_token_id(),
            placeholder_len,
        );
        let attention_mask = vec![1u32; input_ids.len()];

        let gen_ids = self.generate_ids(
            input_ids.as_slice(),
            attention_mask.as_slice(),
            &audio_features,
        )?;
        let gen_text = self.processor.tokenizer.decode(gen_ids.as_slice())?;

        let mut raw_decoded = prefix;
        raw_decoded.push_str(gen_text.as_str());
        self.raw_decoded = raw_decoded;

        let (language, text) = parse_asr_output(
            Some(self.raw_decoded.as_str()),
            self.opts.language.as_deref(),
        );
        let tx = AsrTranscription {
            language,
            text,
            timestamps: None,
        };
        self.last = Some(tx.clone());
        self.chunk_id = self.chunk_id.saturating_add(1);
        Ok(tx)
    }

    fn build_prefix_windowed(&self, rolling: &RollingTextState) -> Result<(Vec<u32>, String)> {
        if self.chunk_id < self.opts.unfixed_chunk_num {
            return Ok((vec![], String::new()));
        }

        let cur_ids = rolling.tail_ids.as_slice();

        let mut k = self.opts.unfixed_token_num;
        loop {
            let end_idx = cur_ids.len().saturating_sub(k);
            let ids = cur_ids.get(..end_idx).unwrap_or(&[]);
            let prefix = if ids.is_empty() {
                String::new()
            } else {
                self.processor.tokenizer.decode(ids)?
            };

            if !prefix.contains('\u{FFFD}') {
                return Ok((ids.to_vec(), prefix));
            }

            if end_idx == 0 {
                return Ok((vec![], String::new()));
            }

            k = k.saturating_add(1);
        }
    }

    fn build_finish_prefix_windowed(
        &self,
        rolling: &RollingTextState,
    ) -> Result<(Vec<u32>, String)> {
        if self.chunk_id < self.opts.unfixed_chunk_num {
            return Ok((vec![], String::new()));
        }

        let cur_ids = rolling.tail_ids.as_slice();
        let mut end_idx = cur_ids.len().saturating_sub(self.opts.unfixed_token_num);
        if end_idx == 0 && !cur_ids.is_empty() {
            end_idx = 1;
        }

        let ids = cur_ids.get(..end_idx).unwrap_or(&[]);
        let prefix = if ids.is_empty() {
            String::new()
        } else {
            self.processor.tokenizer.decode(ids)?
        };
        Ok((ids.to_vec(), prefix))
    }

    fn maybe_commit_windowed_text(&mut self) -> Result<()> {
        let Some(rolling) = self.rolling_text.as_mut() else {
            return Ok(());
        };

        if self.chunk_id < self.opts.unfixed_chunk_num {
            return Ok(());
        }

        if rolling.tail_ids.len() <= rolling.keep_tail_tokens {
            return Ok(());
        }

        let mut commit_n = rolling.tail_ids.len() - rolling.keep_tail_tokens;
        while commit_n > 0 {
            let s = self
                .processor
                .tokenizer
                .decode(&rolling.tail_ids[..commit_n])?;
            if !s.ends_with('\u{FFFD}') {
                rolling.committed.push_str(s.as_str());
                rolling.tail_ids.drain(..commit_n);
                return Ok(());
            }
            commit_n = commit_n.saturating_sub(1);
        }

        Ok(())
    }

    fn decode_step_windowed_with_prefix(
        &mut self,
        prefix_ids: Vec<u32>,
        prefix_str: String,
    ) -> Result<AsrTranscription> {
        let audio_features = self.audio_features_for_step()?;
        let (placeholder_len, _hidden) = audio_features.dims2()?;

        let mut prompt = self.prompt_raw.clone();
        prompt.push_str(prefix_str.as_str());

        let base_ids = self.processor.tokenizer.encode(prompt.as_str())?;
        let input_ids = crate::processor::asr_processor::expand_audio_pad_ids_first(
            base_ids.as_slice(),
            self.model.thinker.audio_token_id(),
            placeholder_len,
        );
        let attention_mask = vec![1u32; input_ids.len()];

        let gen_ids = self.generate_ids(
            input_ids.as_slice(),
            attention_mask.as_slice(),
            &audio_features,
        )?;

        {
            let rolling = self.rolling_text.as_mut().ok_or_else(|| {
                anyhow::anyhow!(
                    "internal error: decode_step_windowed_with_prefix without rolling_text"
                )
            })?;
            rolling.tail_ids.clear();
            rolling.tail_ids.extend(prefix_ids);
            rolling.tail_ids.extend(gen_ids);
        }

        self.chunk_id = self.chunk_id.saturating_add(1);
        self.maybe_commit_windowed_text()?;

        let raw_full = {
            let rolling = self
                .rolling_text
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("internal error: missing rolling_text"))?;
            let tail_text = self
                .processor
                .tokenizer
                .decode(rolling.tail_ids.as_slice())?;
            let mut raw_full = String::with_capacity(rolling.committed.len() + tail_text.len());
            raw_full.push_str(rolling.committed.as_str());
            raw_full.push_str(tail_text.as_str());
            raw_full
        };

        let (language, text) =
            parse_asr_output(Some(raw_full.as_str()), self.opts.language.as_deref());
        let tx = AsrTranscription {
            language,
            text,
            timestamps: None,
        };
        self.last = Some(tx.clone());
        Ok(tx)
    }

    fn decode_step_windowed(&mut self) -> Result<AsrTranscription> {
        if self.rolling_text.is_none() {
            bail!("internal error: decode_step_windowed without rolling_text");
        }

        // During the warmup phase (no rollback prefix), the official algorithm decodes from
        // scratch and replaces prior output. Clear any committed prefix to match.
        if self.chunk_id < self.opts.unfixed_chunk_num {
            if let Some(rolling) = self.rolling_text.as_mut() {
                rolling.committed.clear();
                rolling.tail_ids.clear();
            }
        }

        let (prefix_ids, prefix_str) = {
            let rolling = self
                .rolling_text
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("internal error: missing rolling_text"))?;
            self.build_prefix_windowed(rolling)?
        };
        self.decode_step_windowed_with_prefix(prefix_ids, prefix_str)
    }

    pub fn push_audio_chunk(&mut self, chunk: &AudioInput<'_>) -> Result<Option<AsrTranscription>> {
        let mut wav = normalize_audio_input(chunk)?;
        if wav.is_empty() {
            return Ok(None);
        }

        self.buffer.append(&mut wav);

        if self.buffer_offset > self.buffer.len() {
            bail!(
                "internal error: buffer_offset {} > buffer.len {}",
                self.buffer_offset,
                self.buffer.len()
            );
        }

        let mut out: Option<AsrTranscription> = None;
        while (self.buffer.len() - self.buffer_offset) >= self.chunk_size_samples {
            let start = self.buffer_offset;
            let end = start + self.chunk_size_samples;
            let slice = self
                .buffer
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("internal error: buffer slice out of bounds"))?;
            self.features.append_samples(slice)?;
            self.buffer_offset = end;

            out = if self.rolling_text.is_some() {
                Some(self.decode_step_windowed()?)
            } else {
                let prefix = self.build_prefix()?;
                Some(self.decode_step(prefix)?)
            };
        }

        if self.buffer_offset > 0 {
            self.buffer.drain(..self.buffer_offset);
            self.buffer_offset = 0;
        }

        Ok(out)
    }

    pub fn finish(mut self) -> Result<AsrTranscription> {
        if self.buffer_offset > self.buffer.len() {
            bail!(
                "internal error: buffer_offset {} > buffer.len {}",
                self.buffer_offset,
                self.buffer.len()
            );
        }

        if let Some(tail) = self.buffer.get(self.buffer_offset..) {
            if !tail.is_empty() {
                self.features.append_samples(tail)?;
                self.buffer.clear();
                self.buffer_offset = 0;

                if self.rolling_text.is_some() {
                    if self.chunk_id < self.opts.unfixed_chunk_num {
                        if let Some(rolling) = self.rolling_text.as_mut() {
                            rolling.committed.clear();
                            rolling.tail_ids.clear();
                        }
                    }

                    let (prefix_ids, prefix_str) = {
                        let rolling = self.rolling_text.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("internal error: missing rolling_text")
                        })?;
                        self.build_finish_prefix_windowed(rolling)?
                    };
                    let _ = self.decode_step_windowed_with_prefix(prefix_ids, prefix_str)?;
                } else {
                    let prefix = self.build_finish_prefix()?;
                    let _ = self.decode_step(prefix)?;
                }
            }
        }

        Ok(self.last.unwrap_or_else(|| AsrTranscription {
            language: String::new(),
            text: String::new(),
            timestamps: None,
        }))
    }
}

pub fn start_stream<'a>(
    model: &'a AsrModel,
    processor: &'a AsrProcessor,
    device: &'a Device,
    opts: &StreamOptions,
) -> Result<AsrStream<'a>> {
    let mut opts = opts.clone();

    if !opts.chunk_size_sec.is_finite() || opts.chunk_size_sec <= 0.0 {
        bail!("chunk_size_sec must be finite and > 0");
    }

    if let Some(sec) = opts.audio_window_sec {
        if !sec.is_finite() || sec <= 0.0 {
            bail!("audio_window_sec must be finite and > 0");
        }
    }

    let rolling_text = if let Some(n) = opts.text_window_tokens {
        if n == 0 {
            bail!("text_window_tokens must be > 0");
        }
        let min_keep = opts.unfixed_token_num.saturating_add(1).max(1);
        Some(RollingTextState {
            committed: String::new(),
            tail_ids: vec![],
            keep_tail_tokens: n.max(min_keep),
        })
    } else {
        None
    };

    if let Some(lang) = opts.language.as_deref() {
        let s = lang.trim();
        if s.is_empty() {
            opts.language = None;
        } else {
            let normalized = normalize_language_name(s)?;
            validate_language(normalized.as_str())?;
            opts.language = Some(normalized);
        }
    }

    let mut chunk_size_samples = (opts.chunk_size_sec * SAMPLE_RATE_HZ as f32).round() as usize;
    chunk_size_samples = chunk_size_samples.max(1);

    let prompt_raw = processor.build_text_prompt(opts.context.as_str(), opts.language.as_deref());
    let eos_token_ids = build_eos_token_ids(processor)?;

    Ok(AsrStream {
        model,
        processor,
        device,
        opts,
        chunk_size_samples,
        chunk_id: 0,
        buffer: vec![],
        buffer_offset: 0,
        features: StreamingFeatureExtractor::new(),
        audio_cache: None,
        prompt_raw,
        raw_decoded: String::new(),
        rolling_text,
        eos_token_ids,
        last: None,
    })
}

#[cfg(test)]
mod tests {
    use super::recompute_from_for_stream_append;

    #[test]
    fn test_recompute_from_for_stream_append_matches_segment_boundary() -> anyhow::Result<()> {
        let segment = 800usize;

        let cases = [
            (0usize, 0usize),
            (1usize, 0usize),
            (799usize, 0usize),
            (800usize, 0usize),
            (801usize, 800usize),
            (1599usize, 800usize),
            (1600usize, 800usize),
            (1601usize, 1600usize),
        ];

        for (cache_frames, expected) in cases {
            let got = recompute_from_for_stream_append(cache_frames, segment);
            if got != expected {
                anyhow::bail!(
                    "cache_frames={cache_frames}: expected recompute_from={expected}, got={got}"
                );
            }
        }

        Ok(())
    }
}
