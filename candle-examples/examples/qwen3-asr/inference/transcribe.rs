//! Offline transcription pipeline.

use anyhow::{bail, Result};
use candle::{Device, Tensor};

use crate::audio::input::AudioInput;
use crate::audio::SAMPLE_RATE_HZ;
use crate::inference::types::{AsrTranscription, TranscribeOptions};
use crate::inference::utils::{
    merge_languages, normalize_language_name, parse_asr_output, validate_language,
    MAX_ASR_INPUT_SECONDS,
};
use crate::model::generation::greedy_generate_cached_batch;
#[cfg(feature = "timing")]
use crate::model::generation::{greedy_generate_cached_batch_timed, GenerationTimings};
use crate::model::{thinker::ThinkerForConditionalGeneration, AsrModel};
use crate::processor::chat_template;
use crate::processor::{asr_processor::PreparedInputs, AsrProcessor};

#[cfg(feature = "timing")]
fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TranscribeTimings {
    pub audio_normalize_us: u64,
    pub audio_chunking_us: u64,
    pub processor_prepare_batch_us: u64,
    pub stack_features_us: u64,
    pub audio_encoder_us: u64,
    pub tokenizer_decode_us: u64,
    pub generation: GenerationTimings,
    pub forced_aligner_prep_us: u64,
    pub forced_aligner_stack_features_us: u64,
    pub forced_aligner_forward_us: u64,
    pub forced_aligner_post_us: u64,
    pub forced_aligner_batches: usize,
    pub timestamps_repair_us: u64,
    pub total_us: u64,
    pub chunks: usize,
    pub batches: usize,
}

fn stack_features(prepared: &[PreparedInputs], device: &Device) -> Result<(Tensor, Vec<usize>)> {
    let batch = prepared.len();
    if batch == 0 {
        bail!("prepared is empty");
    }

    let mel = prepared
        .first()
        .map(|p| p.input_features.len())
        .ok_or_else(|| anyhow::anyhow!("prepared missing first item"))?;
    if mel == 0 {
        bail!("prepared input_features has zero mel bins");
    }
    let frames = prepared
        .first()
        .and_then(|p| p.input_features.first())
        .map(|r| r.len())
        .ok_or_else(|| anyhow::anyhow!("prepared input_features missing first row"))?;

    let mut feats: Vec<f32> = Vec::with_capacity(batch.saturating_mul(mel).saturating_mul(frames));
    let mut feature_lens: Vec<usize> = Vec::with_capacity(batch);

    for (i, p) in prepared.iter().enumerate() {
        if p.input_features.len() != mel {
            bail!(
                "prepared[{i}].input_features mel mismatch: expected={mel}, got={}",
                p.input_features.len()
            );
        }
        if p.feature_attention_mask.len() != frames {
            bail!(
                "prepared[{i}].feature_attention_mask len mismatch: expected={frames}, got={}",
                p.feature_attention_mask.len()
            );
        }
        for row in &p.input_features {
            if row.len() != frames {
                bail!(
                    "prepared[{i}].input_features row len mismatch: expected={frames}, got={}",
                    row.len()
                );
            }
            feats.extend_from_slice(row);
        }
        feature_lens.push(p.feature_attention_mask.iter().filter(|&&x| x != 0).count());
    }

    let input_features = Tensor::from_vec(feats, (batch, mel, frames), device)?;
    Ok((input_features, feature_lens))
}

pub(crate) fn build_eos_token_ids(processor: &AsrProcessor) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(2);
    out.push(processor.tokenizer.token_to_id(chat_template::IM_END)?);
    out.push(processor.tokenizer.token_to_id("<|endoftext|>")?);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub(crate) fn generate_raw_prepared_batch(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    prepared: &[PreparedInputs],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<String>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }

    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };

    let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
    let attn_rows: Vec<&[u32]> = prepared
        .iter()
        .map(|p| p.attention_mask.as_slice())
        .collect();

    let (input_features, feature_lens) = stack_features(prepared, device)?;

    let audio_features =
        thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;
    let gen_ids = greedy_generate_cached_batch(
        thinker,
        device,
        ids_rows.as_slice(),
        attn_rows.as_slice(),
        Some(&audio_features),
        max_new_tokens,
        eos_token_ids,
    )?;

    if gen_ids.len() != prepared.len() {
        bail!(
            "internal error: generated batch size mismatch: expected={}, got={}",
            prepared.len(),
            gen_ids.len()
        );
    }

    let mut out: Vec<String> = Vec::with_capacity(gen_ids.len());
    for ids in gen_ids {
        out.push(processor.tokenizer.decode(ids.as_slice())?);
    }
    Ok(out)
}

fn normalize_forced_language(lang: Option<String>) -> Result<Option<String>> {
    let Some(lang) = lang else {
        return Ok(None);
    };
    let s = lang.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let normalized = normalize_language_name(s)?;
    validate_language(normalized.as_str())?;
    Ok(Some(normalized))
}

#[derive(Debug, Clone)]
struct ChunkItem {
    orig_index: usize,
    wav: Vec<f32>,
    #[cfg(feature = "forced-aligner")]
    offset_sec: f32,
}

struct ChunkAsrBatch<'a> {
    thinker: &'a ThinkerForConditionalGeneration,
    processor: &'a AsrProcessor,
    device: &'a Device,
    chunks: &'a [ChunkItem],
    prompts: &'a [String],
    forced_langs_norm: &'a [Option<String>],
    max_new_tokens: usize,
    max_batch_size: usize,
    bucket_by_length: bool,
    eos_token_ids: &'a [u32],
}

fn run_asr_on_chunks_batched(batch: &ChunkAsrBatch<'_>) -> Result<(Vec<String>, Vec<String>)> {
    if batch.chunks.is_empty() {
        return Ok((vec![], vec![]));
    }
    if batch.prompts.len() != batch.forced_langs_norm.len() {
        bail!(
            "internal error: prompts/forced_langs_norm length mismatch: prompts={} forced_langs_norm={}",
            batch.prompts.len(),
            batch.forced_langs_norm.len()
        );
    }

    let batch_size = if batch.max_batch_size == 0 {
        batch.chunks.len().max(1)
    } else {
        batch.max_batch_size
    };

    if !batch.bucket_by_length {
        let mut per_chunk_lang: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let mut per_chunk_text: Vec<String> = Vec::with_capacity(batch.chunks.len());

        for chunk_batch in batch.chunks.chunks(batch_size) {
            let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(chunk_batch.len());
            for c in chunk_batch {
                audio_inputs.push(AudioInput::Waveform {
                    samples: c.wav.as_slice(),
                    sample_rate: SAMPLE_RATE_HZ,
                });
            }

            let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(chunk_batch.len());
            for (c, a) in chunk_batch.iter().zip(audio_inputs.iter()) {
                let prompt = batch.prompts.get(c.orig_index).ok_or_else(|| {
                    anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index)
                })?;
                items.push((prompt.as_str(), a));
            }

            let prepared = batch.processor.prepare_batch(items.as_slice())?;
            let raws = generate_raw_prepared_batch(
                batch.thinker,
                batch.processor,
                batch.device,
                prepared.as_slice(),
                batch.max_new_tokens,
                batch.eos_token_ids,
            )?;

            if raws.len() != chunk_batch.len() {
                bail!(
                    "internal error: raw output batch size mismatch: expected={}, got={}",
                    chunk_batch.len(),
                    raws.len()
                );
            }

            for (c, raw) in chunk_batch.iter().zip(raws.iter()) {
                let forced = batch
                    .forced_langs_norm
                    .get(c.orig_index)
                    .and_then(|x| x.as_deref());
                let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
                per_chunk_lang.push(lang);
                per_chunk_text.push(text);
            }
        }

        return Ok((per_chunk_lang, per_chunk_text));
    }

    let mut indices: Vec<usize> = (0..batch.chunks.len()).collect();
    indices.sort_by(|&a, &b| {
        let la = batch.chunks[a].wav.len();
        let lb = batch.chunks[b].wav.len();
        lb.cmp(&la).then_with(|| a.cmp(&b))
    });

    let mut per_chunk_lang: Vec<Option<String>> = vec![None; batch.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; batch.chunks.len()];

    for idx_batch in indices.chunks(batch_size) {
        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(idx_batch.len());
        let mut batch_chunk_idx: Vec<usize> = Vec::with_capacity(idx_batch.len());
        for &chunk_idx in idx_batch {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
            batch_chunk_idx.push(chunk_idx);
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(idx_batch.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = batch
                .prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let prepared = batch.processor.prepare_batch(items.as_slice())?;
        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        let raws = generate_raw_prepared_batch(
            batch.thinker,
            batch.processor,
            batch.device,
            prepared.as_slice(),
            batch.max_new_tokens,
            batch.eos_token_ids,
        )?;
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        for (k, raw) in raws.iter().enumerate() {
            let chunk_idx = *batch_chunk_idx
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {k}"))?;
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let forced = batch
                .forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);

            let slot_lang = per_chunk_lang
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut langs: Vec<String> = Vec::with_capacity(per_chunk_lang.len());
    for (i, lang) in per_chunk_lang.into_iter().enumerate() {
        langs.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut texts: Vec<String> = Vec::with_capacity(per_chunk_text.len());
    for (i, text) in per_chunk_text.into_iter().enumerate() {
        texts.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    Ok((langs, texts))
}

#[cfg(feature = "timing")]
fn generate_raw_prepared_batch_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    prepared: &[PreparedInputs],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    timings: &mut TranscribeTimings,
) -> Result<Vec<String>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }

    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };

    let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
    let attn_rows: Vec<&[u32]> = prepared
        .iter()
        .map(|p| p.attention_mask.as_slice())
        .collect();

    let start_stack = std::time::Instant::now();
    let (input_features, feature_lens) = stack_features(prepared, device)?;
    timings.stack_features_us = timings
        .stack_features_us
        .saturating_add(duration_to_us(start_stack.elapsed()));

    let start_audio = std::time::Instant::now();
    let audio_features =
        thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;
    timings.audio_encoder_us = timings
        .audio_encoder_us
        .saturating_add(duration_to_us(start_audio.elapsed()));

    let (gen_ids, gen_timings) = greedy_generate_cached_batch_timed(
        thinker,
        device,
        ids_rows.as_slice(),
        attn_rows.as_slice(),
        Some(&audio_features),
        max_new_tokens,
        eos_token_ids,
    )?;
    timings.generation.prompt_tensors_us = timings
        .generation
        .prompt_tensors_us
        .saturating_add(gen_timings.prompt_tensors_us);
    timings.generation.prefill_us = timings
        .generation
        .prefill_us
        .saturating_add(gen_timings.prefill_us);
    timings.generation.decode_us = timings
        .generation
        .decode_us
        .saturating_add(gen_timings.decode_us);
    timings.generation.steps = timings.generation.steps.saturating_add(gen_timings.steps);
    timings.generation.tokens_generated = timings
        .generation
        .tokens_generated
        .saturating_add(gen_timings.tokens_generated);

    if gen_ids.len() != prepared.len() {
        bail!(
            "internal error: generated batch size mismatch: expected={}, got={}",
            prepared.len(),
            gen_ids.len()
        );
    }

    let start_decode = std::time::Instant::now();
    let mut out: Vec<String> = Vec::with_capacity(gen_ids.len());
    for ids in gen_ids {
        out.push(processor.tokenizer.decode(ids.as_slice())?);
    }
    timings.tokenizer_decode_us = timings
        .tokenizer_decode_us
        .saturating_add(duration_to_us(start_decode.elapsed()));
    Ok(out)
}

#[cfg(feature = "timing")]
fn run_asr_on_chunks_batched_timed(
    batch: &ChunkAsrBatch<'_>,
    timings: &mut TranscribeTimings,
) -> Result<(Vec<String>, Vec<String>)> {
    if batch.chunks.is_empty() {
        return Ok((vec![], vec![]));
    }
    if batch.prompts.len() != batch.forced_langs_norm.len() {
        bail!(
            "internal error: prompts/forced_langs_norm length mismatch: prompts={} forced_langs_norm={}",
            batch.prompts.len(),
            batch.forced_langs_norm.len()
        );
    }

    let batch_size = if batch.max_batch_size == 0 {
        batch.chunks.len().max(1)
    } else {
        batch.max_batch_size
    };

    if !batch.bucket_by_length {
        let mut per_chunk_lang: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let mut per_chunk_text: Vec<String> = Vec::with_capacity(batch.chunks.len());

        for chunk_batch in batch.chunks.chunks(batch_size) {
            timings.batches = timings.batches.saturating_add(1);

            let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(chunk_batch.len());
            for c in chunk_batch {
                audio_inputs.push(AudioInput::Waveform {
                    samples: c.wav.as_slice(),
                    sample_rate: SAMPLE_RATE_HZ,
                });
            }

            let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(chunk_batch.len());
            for (c, a) in chunk_batch.iter().zip(audio_inputs.iter()) {
                let prompt = batch.prompts.get(c.orig_index).ok_or_else(|| {
                    anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index)
                })?;
                items.push((prompt.as_str(), a));
            }

            let start_prepare = std::time::Instant::now();
            let prepared = batch.processor.prepare_batch(items.as_slice())?;
            timings.processor_prepare_batch_us = timings
                .processor_prepare_batch_us
                .saturating_add(duration_to_us(start_prepare.elapsed()));

            let raws = generate_raw_prepared_batch_timed(
                batch.thinker,
                batch.processor,
                batch.device,
                prepared.as_slice(),
                batch.max_new_tokens,
                batch.eos_token_ids,
                timings,
            )?;

            if raws.len() != chunk_batch.len() {
                bail!(
                    "internal error: raw output batch size mismatch: expected={}, got={}",
                    chunk_batch.len(),
                    raws.len()
                );
            }

            for (c, raw) in chunk_batch.iter().zip(raws.iter()) {
                let forced = batch
                    .forced_langs_norm
                    .get(c.orig_index)
                    .and_then(|x| x.as_deref());
                let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
                per_chunk_lang.push(lang);
                per_chunk_text.push(text);
            }
        }

        return Ok((per_chunk_lang, per_chunk_text));
    }

    let mut indices: Vec<usize> = (0..batch.chunks.len()).collect();
    indices.sort_by(|&a, &b| {
        let la = batch.chunks[a].wav.len();
        let lb = batch.chunks[b].wav.len();
        lb.cmp(&la).then_with(|| a.cmp(&b))
    });

    let mut per_chunk_lang: Vec<Option<String>> = vec![None; batch.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; batch.chunks.len()];

    for idx_batch in indices.chunks(batch_size) {
        timings.batches = timings.batches.saturating_add(1);

        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(idx_batch.len());
        let mut batch_chunk_idx: Vec<usize> = Vec::with_capacity(idx_batch.len());
        for &chunk_idx in idx_batch {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
            batch_chunk_idx.push(chunk_idx);
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(idx_batch.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = batch
                .prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let start_prepare = std::time::Instant::now();
        let prepared = batch.processor.prepare_batch(items.as_slice())?;
        timings.processor_prepare_batch_us = timings
            .processor_prepare_batch_us
            .saturating_add(duration_to_us(start_prepare.elapsed()));
        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        let raws = generate_raw_prepared_batch_timed(
            batch.thinker,
            batch.processor,
            batch.device,
            prepared.as_slice(),
            batch.max_new_tokens,
            batch.eos_token_ids,
            timings,
        )?;
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        for (k, raw) in raws.iter().enumerate() {
            let chunk_idx = *batch_chunk_idx
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {k}"))?;
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let forced = batch
                .forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);

            let slot_lang = per_chunk_lang
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut langs: Vec<String> = Vec::with_capacity(per_chunk_lang.len());
    for (i, lang) in per_chunk_lang.into_iter().enumerate() {
        langs.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut texts: Vec<String> = Vec::with_capacity(per_chunk_text.len());
    for (i, text) in per_chunk_text.into_iter().enumerate() {
        texts.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    Ok((langs, texts))
}

pub fn transcribe(
    model: &AsrModel,
    processor: &AsrProcessor,
    device: &Device,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<Vec<AsrTranscription>> {
    if opts.return_timestamps {
        bail!(
            "return_timestamps requires forced aligner support; use transcribe_with_forced_aligner (feature `forced-aligner`) instead"
        );
    }

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    // The official Python stack always chunks long audio to a maximum duration to avoid OOM.
    // Allow an optional smaller user override via `chunk_max_sec`.
    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(MAX_ASR_INPUT_SECONDS)
        .min(MAX_ASR_INPUT_SECONDS);

    let eos_token_ids = build_eos_token_ids(processor)?;
    if audio.is_empty() {
        return Ok(vec![]);
    }

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                #[cfg(feature = "forced-aligner")]
                offset_sec: part.offset_sec,
            });
        }
    }

    let asr_batch = ChunkAsrBatch {
        thinker: &model.thinker,
        processor,
        device,
        chunks: chunks.as_slice(),
        prompts: prompts.as_slice(),
        forced_langs_norm: forced_langs_norm.as_slice(),
        max_new_tokens: opts.max_new_tokens,
        max_batch_size: opts.max_batch_size,
        bucket_by_length: opts.bucket_by_length,
        eos_token_ids: eos_token_ids.as_slice(),
    };
    let (per_chunk_lang, per_chunk_text) = run_asr_on_chunks_batched(&asr_batch)?;

    if per_chunk_lang.len() != chunks.len() || per_chunk_text.len() != chunks.len() {
        bail!(
            "internal error: per_chunk outputs length mismatch: chunks={} langs={} texts={}",
            chunks.len(),
            per_chunk_lang.len(),
            per_chunk_text.len()
        );
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());
    }

    let mut out: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for (langs, texts) in out_langs.into_iter().zip(out_texts.into_iter()) {
        out.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        });
    }

    Ok(out)
}

#[cfg(feature = "timing")]
pub fn transcribe_timed(
    model: &AsrModel,
    processor: &AsrProcessor,
    device: &Device,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
    if opts.return_timestamps {
        bail!(
            "transcribe_timed does not support return_timestamps yet; use transcribe_with_forced_aligner (feature `forced-aligner`) instead"
        );
    }

    let start_total = std::time::Instant::now();
    let mut timings = TranscribeTimings::default();

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(MAX_ASR_INPUT_SECONDS)
        .min(MAX_ASR_INPUT_SECONDS);

    let eos_token_ids = build_eos_token_ids(processor)?;
    if audio.is_empty() {
        return Ok((vec![], timings));
    }

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let start_norm = std::time::Instant::now();
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        timings.audio_normalize_us = timings
            .audio_normalize_us
            .saturating_add(duration_to_us(start_norm.elapsed()));

        let start_chunk = std::time::Instant::now();
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        timings.audio_chunking_us = timings
            .audio_chunking_us
            .saturating_add(duration_to_us(start_chunk.elapsed()));

        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                #[cfg(feature = "forced-aligner")]
                offset_sec: part.offset_sec,
            });
        }
    }
    timings.chunks = chunks.len();

    let asr_batch = ChunkAsrBatch {
        thinker: &model.thinker,
        processor,
        device,
        chunks: chunks.as_slice(),
        prompts: prompts.as_slice(),
        forced_langs_norm: forced_langs_norm.as_slice(),
        max_new_tokens: opts.max_new_tokens,
        max_batch_size: opts.max_batch_size,
        bucket_by_length: opts.bucket_by_length,
        eos_token_ids: eos_token_ids.as_slice(),
    };
    let (per_chunk_lang, per_chunk_text) =
        run_asr_on_chunks_batched_timed(&asr_batch, &mut timings)?;

    if per_chunk_lang.len() != chunks.len() || per_chunk_text.len() != chunks.len() {
        bail!(
            "internal error: per_chunk outputs length mismatch: chunks={} langs={} texts={}",
            chunks.len(),
            per_chunk_lang.len(),
            per_chunk_text.len()
        );
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());
    }

    let mut out: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for (langs, texts) in out_langs.into_iter().zip(out_texts.into_iter()) {
        out.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        });
    }

    timings.total_us = duration_to_us(start_total.elapsed());
    Ok((out, timings))
}

#[cfg(feature = "forced-aligner")]
#[derive(Debug, Clone, serde::Serialize)]
struct TimestampItem {
    text: String,
    start_time: f64,
    end_time: f64,
}

#[cfg(feature = "forced-aligner")]
#[derive(Debug, Clone, serde::Serialize)]
struct TimestampResult {
    items: Vec<TimestampItem>,
}

#[cfg(feature = "forced-aligner")]
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(feature = "forced-aligner")]
fn offset_align_items(
    result: &crate::forced_aligner::ForcedAlignResult,
    offset_sec: f32,
) -> Vec<TimestampItem> {
    let offset_sec = f64::from(offset_sec);
    result
        .items
        .iter()
        .map(|it| TimestampItem {
            text: it.text.clone(),
            start_time: round3(it.start_time + offset_sec),
            end_time: round3(it.end_time + offset_sec),
        })
        .collect()
}

#[cfg(feature = "forced-aligner")]
fn repair_merged_align_items(items: &mut [TimestampItem]) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }

    let mut ms_i64: Vec<i64> = Vec::with_capacity(items.len().saturating_mul(2));
    for it in items.iter() {
        if !it.start_time.is_finite() || !it.end_time.is_finite() {
            bail!(
                "invalid timestamp item (non-finite time): start={} end={}",
                it.start_time,
                it.end_time
            );
        }

        let start_ms_f = it.start_time * 1000.0;
        let end_ms_f = it.end_time * 1000.0;

        let start_ms = if start_ms_f >= i64::MAX as f64 {
            i64::MAX
        } else if start_ms_f <= i64::MIN as f64 {
            i64::MIN
        } else {
            start_ms_f.round() as i64
        };
        let end_ms = if end_ms_f >= i64::MAX as f64 {
            i64::MAX
        } else if end_ms_f <= i64::MIN as f64 {
            i64::MIN
        } else {
            end_ms_f.round() as i64
        };

        ms_i64.push(start_ms);
        ms_i64.push(end_ms);
    }

    let mut needs_fix = false;
    for w in ms_i64.windows(2) {
        if w[0] > w[1] {
            needs_fix = true;
            break;
        }
    }
    if !needs_fix {
        return Ok(());
    }

    // The forced aligner fixes non-monotonic timestamps per chunk, but extreme chunking settings
    // can produce cross-chunk overlaps. Repair the merged sequence with the same algorithm.
    let ms_f32 = ms_i64.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let fixed = crate::forced_aligner::fix_timestamp(ms_f32.as_slice());
    if fixed.len() != ms_i64.len() {
        bail!(
            "internal error: fix_timestamp length mismatch: expected={}, got={}",
            ms_i64.len(),
            fixed.len()
        );
    }

    // Apply the repaired ms values back to items, enforcing a final monotonic clamp to guard
    // against any unexpected edge cases (e.g., float precision).
    let mut prev_ms: i64 = 0;
    for (i, it) in items.iter_mut().enumerate() {
        let base = i
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("timestamp index overflow"))?;
        let start_idx = base;
        let end_idx = base
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("timestamp index overflow"))?;

        let start_ms_raw = *fixed
            .get(start_idx)
            .ok_or_else(|| anyhow::anyhow!("missing start timestamp (idx={start_idx})"))?;
        let end_ms_raw = *fixed
            .get(end_idx)
            .ok_or_else(|| anyhow::anyhow!("missing end timestamp (idx={end_idx})"))?;

        let start_ms = start_ms_raw.max(prev_ms);
        let end_ms = end_ms_raw.max(start_ms);
        prev_ms = end_ms;

        it.start_time = round3(start_ms as f64 / 1000.0);
        it.end_time = round3(end_ms as f64 / 1000.0);
    }

    Ok(())
}

#[cfg(all(feature = "forced-aligner", feature = "timing"))]
pub fn transcribe_with_forced_aligner_timed(
    model: &AsrModel,
    processor: &AsrProcessor,
    device: &Device,
    forced_aligner: &crate::forced_aligner::Qwen3ForcedAligner,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
    if !opts.return_timestamps {
        return transcribe_timed(model, processor, device, audio, opts);
    }

    let start_total = std::time::Instant::now();
    let mut timings = TranscribeTimings::default();

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(crate::inference::utils::MAX_FORCE_ALIGN_INPUT_SECONDS)
        .min(crate::inference::utils::MAX_FORCE_ALIGN_INPUT_SECONDS);

    let eos_token_ids = build_eos_token_ids(processor)?;
    if audio.is_empty() {
        return Ok((vec![], timings));
    }

    // Chunk all samples first and keep (orig_index, offset) mapping like the official stack.
    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let start_norm = std::time::Instant::now();
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        timings.audio_normalize_us = timings
            .audio_normalize_us
            .saturating_add(duration_to_us(start_norm.elapsed()));

        let start_chunk = std::time::Instant::now();
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        timings.audio_chunking_us = timings
            .audio_chunking_us
            .saturating_add(duration_to_us(start_chunk.elapsed()));

        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                #[cfg(feature = "forced-aligner")]
                offset_sec: part.offset_sec,
            });
        }
    }
    timings.chunks = chunks.len();

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let batch_size = if opts.max_batch_size == 0 {
        chunks.len().max(1)
    } else {
        opts.max_batch_size
    };

    let mut indices: Vec<usize> = (0..chunks.len()).collect();
    if opts.bucket_by_length {
        indices.sort_by(|&a, &b| {
            let la = chunks[a].wav.len();
            let lb = chunks[b].wav.len();
            lb.cmp(&la).then_with(|| a.cmp(&b))
        });
    }

    let mut per_chunk_lang_opt: Vec<Option<String>> = vec![None; chunks.len()];
    let mut per_chunk_text_opt: Vec<Option<String>> = vec![None; chunks.len()];
    let mut per_chunk_align_items: Vec<Option<Vec<TimestampItem>>> = vec![None; chunks.len()];

    for idx_batch in indices.chunks(batch_size) {
        timings.batches = timings.batches.saturating_add(1);

        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(idx_batch.len());
        let mut batch_chunk_idx: Vec<usize> = Vec::with_capacity(idx_batch.len());
        for &chunk_idx in idx_batch {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
            batch_chunk_idx.push(chunk_idx);
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(idx_batch.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let start_prepare = std::time::Instant::now();
        let prepared = processor.prepare_batch(items.as_slice())?;
        timings.processor_prepare_batch_us = timings
            .processor_prepare_batch_us
            .saturating_add(duration_to_us(start_prepare.elapsed()));

        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        let raws = generate_raw_prepared_batch_timed(
            &model.thinker,
            processor,
            device,
            prepared.as_slice(),
            opts.max_new_tokens,
            eos_token_ids.as_slice(),
            &mut timings,
        )?;
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        let mut batch_lang: Vec<String> = Vec::with_capacity(batch_chunk_idx.len());
        let mut batch_text: Vec<String> = Vec::with_capacity(batch_chunk_idx.len());
        for (&chunk_idx, raw) in batch_chunk_idx.iter().zip(raws.iter()) {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let forced = forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
            batch_lang.push(lang);
            batch_text.push(text);
        }

        let mut align_prepared: Vec<&PreparedInputs> = Vec::new();
        let mut align_text: Vec<&str> = Vec::new();
        let mut align_lang: Vec<&str> = Vec::new();
        let mut align_chunk_idx: Vec<usize> = Vec::new();

        for (rel_idx, t) in batch_text.iter().enumerate() {
            if t.trim().is_empty() {
                continue;
            }

            let chunk_idx = *batch_chunk_idx
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {rel_idx}"))?;
            let l = batch_lang
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing batch language {rel_idx}"))?;
            let p = prepared
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing prepared inputs {rel_idx}"))?;

            align_prepared.push(p);
            align_text.push(t.as_str());
            align_lang.push(l.as_str());
            align_chunk_idx.push(chunk_idx);
        }

        if !align_chunk_idx.is_empty() {
            timings.forced_aligner_batches = timings.forced_aligner_batches.saturating_add(1);

            let mut fa_timings = crate::forced_aligner::model::ForcedAlignTimings::default();
            let aligned = forced_aligner.align_with_features_timed(
                align_prepared.as_slice(),
                align_text.as_slice(),
                align_lang.as_slice(),
                &mut fa_timings,
            )?;

            timings.forced_aligner_prep_us = timings
                .forced_aligner_prep_us
                .saturating_add(fa_timings.prep_us);
            timings.forced_aligner_stack_features_us = timings
                .forced_aligner_stack_features_us
                .saturating_add(fa_timings.stack_features_us);
            timings.forced_aligner_forward_us = timings
                .forced_aligner_forward_us
                .saturating_add(fa_timings.forward_us);
            timings.forced_aligner_post_us = timings
                .forced_aligner_post_us
                .saturating_add(fa_timings.post_us);

            if aligned.len() != align_chunk_idx.len() {
                bail!(
                    "forced aligner batch size mismatch: expected={}, got={}",
                    align_chunk_idx.len(),
                    aligned.len()
                );
            }

            for (k, result) in aligned.iter().enumerate() {
                let chunk_idx = align_chunk_idx
                    .get(k)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing aligned chunk index {k}"))?;
                let c = chunks
                    .get(chunk_idx)
                    .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
                let slot = per_chunk_align_items
                    .get_mut(chunk_idx)
                    .ok_or_else(|| anyhow::anyhow!("missing chunk align slot {chunk_idx}"))?;
                *slot = Some(offset_align_items(result, c.offset_sec));
            }
        }

        for ((chunk_idx, lang), text) in batch_chunk_idx
            .into_iter()
            .zip(batch_lang.into_iter())
            .zip(batch_text.into_iter())
        {
            let slot_lang = per_chunk_lang_opt
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text_opt
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut per_chunk_lang: Vec<String> = Vec::with_capacity(chunks.len());
    for (i, lang) in per_chunk_lang_opt.into_iter().enumerate() {
        per_chunk_lang.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut per_chunk_text: Vec<String> = Vec::with_capacity(chunks.len());
    for (i, text) in per_chunk_text_opt.into_iter().enumerate() {
        per_chunk_text.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    // Merge per-chunk outputs back to original samples.
    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_align: Vec<Vec<TimestampItem>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());

        let aligns = out_align
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output align bucket"))?;
        if let Some(items) = per_chunk_align_items
            .get(idx)
            .and_then(|items| items.as_ref())
        {
            aligns.extend(items.clone());
        }
    }

    let mut results: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for ((langs, texts), align_items) in out_langs
        .into_iter()
        .zip(out_texts.into_iter())
        .zip(out_align.into_iter())
    {
        let mut align_items = align_items;
        let start_repair = std::time::Instant::now();
        repair_merged_align_items(align_items.as_mut_slice())?;
        timings.timestamps_repair_us = timings
            .timestamps_repair_us
            .saturating_add(duration_to_us(start_repair.elapsed()));

        let timestamps = if align_items.is_empty() {
            None
        } else {
            Some(serde_json::to_value(TimestampResult {
                items: align_items,
            })?)
        };

        results.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps,
        });
    }

    timings.total_us = duration_to_us(start_total.elapsed());
    Ok((results, timings))
}

/// Offline transcription with forced aligner timestamps.
///
/// This mirrors `Qwen3ASRModel.transcribe(..., return_time_stamps=True)` from the official Python
/// stack. The aligner is run per audio chunk (<= 180s), and the outputs are time-offset and merged.
#[cfg(feature = "forced-aligner")]
pub fn transcribe_with_forced_aligner(
    model: &AsrModel,
    processor: &AsrProcessor,
    device: &Device,
    forced_aligner: &crate::forced_aligner::Qwen3ForcedAligner,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<Vec<AsrTranscription>> {
    if !opts.return_timestamps {
        return transcribe(model, processor, device, audio, opts);
    }

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(crate::inference::utils::MAX_FORCE_ALIGN_INPUT_SECONDS)
        .min(crate::inference::utils::MAX_FORCE_ALIGN_INPUT_SECONDS);

    // Chunk all samples first and keep (orig_index, offset) mapping like the official stack.
    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                #[cfg(feature = "forced-aligner")]
                offset_sec: part.offset_sec,
            });
        }
    }

    let eos_token_ids = build_eos_token_ids(processor)?;
    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let batch_size = if opts.max_batch_size == 0 {
        chunks.len().max(1)
    } else {
        opts.max_batch_size
    };

    let mut indices: Vec<usize> = (0..chunks.len()).collect();
    if opts.bucket_by_length {
        indices.sort_by(|&a, &b| {
            let la = chunks[a].wav.len();
            let lb = chunks[b].wav.len();
            lb.cmp(&la).then_with(|| a.cmp(&b))
        });
    }

    let mut per_chunk_lang_opt: Vec<Option<String>> = vec![None; chunks.len()];
    let mut per_chunk_text_opt: Vec<Option<String>> = vec![None; chunks.len()];
    let mut per_chunk_align_items: Vec<Option<Vec<TimestampItem>>> = vec![None; chunks.len()];

    // Run ASR and forced alignment per batch to reuse audio features between the two models.
    for idx_batch in indices.chunks(batch_size) {
        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(idx_batch.len());
        let mut batch_chunk_idx: Vec<usize> = Vec::with_capacity(idx_batch.len());
        for &chunk_idx in idx_batch {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
            batch_chunk_idx.push(chunk_idx);
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(idx_batch.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let prepared = processor.prepare_batch(items.as_slice())?;
        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        let raws = generate_raw_prepared_batch(
            &model.thinker,
            processor,
            device,
            prepared.as_slice(),
            opts.max_new_tokens,
            eos_token_ids.as_slice(),
        )?;
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        let mut batch_lang: Vec<String> = Vec::with_capacity(batch_chunk_idx.len());
        let mut batch_text: Vec<String> = Vec::with_capacity(batch_chunk_idx.len());
        for (&chunk_idx, raw) in batch_chunk_idx.iter().zip(raws.iter()) {
            let c = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let forced = forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
            batch_lang.push(lang);
            batch_text.push(text);
        }

        // Align only non-empty chunk outputs, reusing the already-prepared audio features.
        let mut align_prepared: Vec<&PreparedInputs> = Vec::new();
        let mut align_text: Vec<&str> = Vec::new();
        let mut align_lang: Vec<&str> = Vec::new();
        let mut align_chunk_idx: Vec<usize> = Vec::new();

        for (rel_idx, t) in batch_text.iter().enumerate() {
            if t.trim().is_empty() {
                continue;
            }

            let chunk_idx = *batch_chunk_idx
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {rel_idx}"))?;
            let l = batch_lang
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing batch language {rel_idx}"))?;
            let p = prepared
                .get(rel_idx)
                .ok_or_else(|| anyhow::anyhow!("missing prepared inputs {rel_idx}"))?;

            align_prepared.push(p);
            align_text.push(t.as_str());
            align_lang.push(l.as_str());
            align_chunk_idx.push(chunk_idx);
        }

        if !align_chunk_idx.is_empty() {
            let aligned = forced_aligner.align_with_features(
                align_prepared.as_slice(),
                align_text.as_slice(),
                align_lang.as_slice(),
            )?;
            if aligned.len() != align_chunk_idx.len() {
                bail!(
                    "forced aligner batch size mismatch: expected={}, got={}",
                    align_chunk_idx.len(),
                    aligned.len()
                );
            }

            for (k, result) in aligned.iter().enumerate() {
                let chunk_idx = align_chunk_idx
                    .get(k)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing aligned chunk index {k}"))?;
                let c = chunks
                    .get(chunk_idx)
                    .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
                let slot = per_chunk_align_items
                    .get_mut(chunk_idx)
                    .ok_or_else(|| anyhow::anyhow!("missing chunk align slot {chunk_idx}"))?;
                *slot = Some(offset_align_items(result, c.offset_sec));
            }
        }

        for ((chunk_idx, lang), text) in batch_chunk_idx
            .into_iter()
            .zip(batch_lang.into_iter())
            .zip(batch_text.into_iter())
        {
            let slot_lang = per_chunk_lang_opt
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text_opt
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut per_chunk_lang: Vec<String> = Vec::with_capacity(chunks.len());
    for (i, lang) in per_chunk_lang_opt.into_iter().enumerate() {
        per_chunk_lang.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut per_chunk_text: Vec<String> = Vec::with_capacity(chunks.len());
    for (i, text) in per_chunk_text_opt.into_iter().enumerate() {
        per_chunk_text.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    // Merge per-chunk outputs back to original samples.
    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_align: Vec<Vec<TimestampItem>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());

        let aligns = out_align
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output align bucket"))?;
        if let Some(items) = per_chunk_align_items
            .get(idx)
            .and_then(|items| items.as_ref())
        {
            aligns.extend(items.clone());
        }
    }

    let mut results: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for ((langs, texts), align_items) in out_langs
        .into_iter()
        .zip(out_texts.into_iter())
        .zip(out_align.into_iter())
    {
        let mut align_items = align_items;
        repair_merged_align_items(align_items.as_mut_slice())?;
        let timestamps = if align_items.is_empty() {
            None
        } else {
            Some(serde_json::to_value(TimestampResult {
                items: align_items,
            })?)
        };

        results.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps,
        });
    }

    Ok(results)
}

#[cfg(all(test, feature = "forced-aligner"))]
mod tests {
    use crate::forced_aligner::{ForcedAlignItem, ForcedAlignResult};

    #[test]
    fn test_offset_align_items_offsets_and_rounds() -> anyhow::Result<()> {
        let r = ForcedAlignResult {
            items: vec![
                ForcedAlignItem {
                    text: "hello".to_string(),
                    start_time: 0.0,
                    end_time: 0.1,
                },
                ForcedAlignItem {
                    text: "world".to_string(),
                    start_time: 0.1,
                    end_time: 0.2,
                },
            ],
        };

        let items = super::offset_align_items(&r, 1.234);
        if items.len() != 2 {
            anyhow::bail!("expected 2 items, got {}", items.len());
        }

        if items[0].text != "hello" || items[1].text != "world" {
            anyhow::bail!("unexpected texts: {items:?}");
        }

        if items[0].start_time != 1.234 || items[0].end_time != 1.334 {
            anyhow::bail!(
                "unexpected first span: start={} end={}",
                items[0].start_time,
                items[0].end_time
            );
        }
        if items[1].start_time != 1.334 || items[1].end_time != 1.434 {
            anyhow::bail!(
                "unexpected second span: start={} end={}",
                items[1].start_time,
                items[1].end_time
            );
        }

        Ok(())
    }

    #[test]
    fn test_repair_merged_align_items_repairs_cross_chunk_overlap() -> anyhow::Result<()> {
        let mut items = vec![
            super::TimestampItem {
                text: "a".to_string(),
                start_time: 0.800,
                end_time: 0.900,
            },
            super::TimestampItem {
                text: "b".to_string(),
                start_time: 0.400,
                end_time: 0.500,
            },
            super::TimestampItem {
                text: "c".to_string(),
                start_time: 1.000,
                end_time: 1.100,
            },
        ];

        super::repair_merged_align_items(items.as_mut_slice())?;

        let mut prev_start: Option<f64> = None;
        let mut prev_end: f64 = 0.0;
        let eps = 1e-9f64;
        for it in &items {
            if let Some(prev) = prev_start {
                if it.start_time + eps < prev {
                    anyhow::bail!(
                        "expected monotonic start_time, got prev={prev} start={}",
                        it.start_time
                    );
                }
            }
            if it.start_time + eps < prev_end {
                anyhow::bail!(
                    "expected non-overlapping spans, got prev_end={prev_end} start={}",
                    it.start_time
                );
            }
            if it.start_time > it.end_time + eps {
                anyhow::bail!(
                    "expected start_time <= end_time, got start={} end={}",
                    it.start_time,
                    it.end_time
                );
            }

            // Ensure millisecond rounding.
            let start_ms = it.start_time * 1000.0;
            let end_ms = it.end_time * 1000.0;
            if (start_ms - start_ms.round()).abs() > 1e-6 {
                anyhow::bail!("start_time is not ms-rounded: {}", it.start_time);
            }
            if (end_ms - end_ms.round()).abs() > 1e-6 {
                anyhow::bail!("end_time is not ms-rounded: {}", it.end_time);
            }

            prev_start = Some(it.start_time);
            prev_end = it.end_time;
        }

        Ok(())
    }

    #[test]
    fn test_offset_and_repair_merged_align_items_across_chunks() -> anyhow::Result<()> {
        // Mirror the semantics used by `transcribe_with_forced_aligner`:
        // - align per chunk,
        // - offset timestamps by chunk offset,
        // - concatenate,
        // - repair merged timestamps for monotonicity.
        //
        // This specifically covers the edge case where a very short first chunk can have an
        // aligner output that extends beyond the next chunk's offset.
        let chunk_max_sec = 1.0f32;

        let c0 = ForcedAlignResult {
            items: vec![ForcedAlignItem {
                text: "a".to_string(),
                start_time: 0.892,
                end_time: 0.950,
            }],
        };
        let c1 = ForcedAlignResult {
            items: vec![ForcedAlignItem {
                text: "b".to_string(),
                start_time: 0.0,
                end_time: 0.1,
            }],
        };
        let c2 = ForcedAlignResult {
            items: vec![ForcedAlignItem {
                text: "c".to_string(),
                start_time: 0.0,
                end_time: 0.2,
            }],
        };

        let mut items = Vec::new();
        items.extend(super::offset_align_items(&c0, 0.0));
        // Simulate a short first chunk with a small offset for the second chunk.
        items.extend(super::offset_align_items(&c1, 0.426));
        // Add a later chunk to ensure the merged timestamps extend beyond the first chunk.
        items.extend(super::offset_align_items(&c2, 1.2));

        super::repair_merged_align_items(items.as_mut_slice())?;

        let mut prev_start: Option<f64> = None;
        let eps = 1e-9f64;
        let mut max_end = 0.0f64;
        for it in &items {
            if let Some(prev) = prev_start {
                if it.start_time + eps < prev {
                    anyhow::bail!(
                        "expected monotonic start_time, got prev={prev} start={}",
                        it.start_time
                    );
                }
            }
            prev_start = Some(it.start_time);
            if it.end_time > max_end {
                max_end = it.end_time;
            }
        }

        if max_end <= f64::from(chunk_max_sec) + 0.1 {
            anyhow::bail!(
                "expected merged timestamps to extend beyond first chunk: chunk_max_sec={chunk_max_sec} max_end={max_end}"
            );
        }

        Ok(())
    }
}
