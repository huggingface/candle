//! Autoregressive generation loop (greedy first, then sampling).
//!
//! This module is intentionally small and conservative: it prioritizes correctness
//! and parity bring-up over performance. KV caching is a separate concern.

use anyhow::{bail, Result};
use candle::{DType, Device, IndexOp, Tensor};

use crate::model::kv_cache::KVCache;
use crate::model::thinker::get_rope_index;
use crate::model::thinker::ThinkerForConditionalGeneration;

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
pub struct GenerationTimings {
    /// Time spent building the initial prompt tensors and transferring them to device.
    pub prompt_tensors_us: u64,
    /// Time spent running the initial prompt prefill forward pass (including mRoPE indices).
    pub prefill_us: u64,
    /// Time spent in the token-by-token decode loop (including mRoPE indices).
    pub decode_us: u64,
    /// Number of decode loop iterations executed (<= `max_new_tokens`).
    pub steps: usize,
    /// Total non-EOS tokens produced across the batch.
    pub tokens_generated: usize,
}

#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Default)]
pub struct GenerationTimings;

/// Greedy decode for a single sequence (batch=1).
///
/// Returns generated token ids (excluding the input prompt ids).
pub fn greedy_generate(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    if input_ids.is_empty() {
        bail!("input_ids is empty");
    }
    if attention_mask.len() != input_ids.len() {
        bail!(
            "attention_mask length mismatch: expected={}, got={}",
            input_ids.len(),
            attention_mask.len()
        );
    }
    if eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }
    if max_new_tokens == 0 {
        return Ok(vec![]);
    }

    let mut ids: Vec<u32> = input_ids.to_vec();
    let mut attn: Vec<u32> = attention_mask.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders(input_ids, thinker.audio_token_id())
    } else {
        0
    };

    for _step in 0..max_new_tokens {
        let seq_len = ids.len();
        let input_ids_t = Tensor::from_vec(ids.clone(), (1usize, seq_len), device)?;
        let attention_mask_t = Tensor::from_vec(attn.clone(), (1usize, seq_len), device)?;

        let logits = thinker.forward_with_audio_features(
            &input_ids_t,
            &attention_mask_t,
            audio_features,
            audio_placeholder_count,
        )?;

        let last = logits.i((0usize, seq_len.saturating_sub(1)))?;
        let next_id = last.argmax(0usize)?.to_scalar::<u32>()?;

        if eos_token_ids.contains(&next_id) {
            break;
        }

        generated.push(next_id);
        ids.push(next_id);
        attn.push(1);
    }

    Ok(generated)
}

/// Greedy decode with a KV cache (single sequence, batch=1).
///
/// This is the practical default for transcription: without caching, decoding
/// is prohibitively slow because each step would re-run the full prompt.
pub fn greedy_generate_cached(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    if input_ids.is_empty() {
        bail!("input_ids is empty");
    }
    if attention_mask.len() != input_ids.len() {
        bail!(
            "attention_mask length mismatch: expected={}, got={}",
            input_ids.len(),
            attention_mask.len()
        );
    }
    if eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }
    if max_new_tokens == 0 {
        return Ok(vec![]);
    }

    let mut ids: Vec<u32> = input_ids.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    let mut kv_cache = KVCache::new();

    // Prefill the cache with the full prompt (including audio features).
    let seq_len = ids.len();
    let input_ids_t = Tensor::from_vec(ids.clone(), (1usize, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attention_mask.to_vec(), (1usize, seq_len), device)?;
    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders(input_ids, thinker.audio_token_id())
    } else {
        0
    };
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        audio_features,
        audio_placeholder_count,
    )?;
    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let logits = thinker.forward_embeds_with_kv_cache(
        &attention_mask_t,
        &position_ids,
        &inputs_embeds,
        &mut kv_cache,
    )?;

    let mut next_id = logits
        .i((0usize, seq_len.saturating_sub(1)))?
        .argmax(0usize)?
        .to_scalar::<u32>()?;

    let prompt_len = prompt_len_from_left_padded_mask(attention_mask, seq_len)?;
    let ones_col = Tensor::ones((1usize, 1usize), DType::U32, device)?;
    let mut attention_mask_total = attention_mask_t;
    let mut total_len = seq_len;

    for step in 0..max_new_tokens {
        if eos_token_ids.contains(&next_id) {
            break;
        }

        generated.push(next_id);
        ids.push(next_id);

        total_len = total_len
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("total_len overflow"))?;
        attention_mask_total = Tensor::cat(&[&attention_mask_total, &ones_col], 1)?;

        let input_ids_new = Tensor::from_vec(vec![next_id], (1usize, 1usize), device)?;
        let inputs_embeds_new = thinker.embed_tokens(&input_ids_new)?;

        let position_ids_new = position_ids_for_step(&[prompt_len], step, device)?;

        let logits_new = thinker.forward_embeds_with_kv_cache(
            &attention_mask_total,
            &position_ids_new,
            &inputs_embeds_new,
            &mut kv_cache,
        )?;
        next_id = logits_new
            .i((0usize, 0usize))?
            .argmax(0usize)?
            .to_scalar::<u32>()?;
    }

    Ok(generated)
}

/// Greedy decode with a KV cache for a batch of sequences.
///
/// This mirrors the shape semantics used by Transformers `generate()`:
/// - `input_ids` and `attention_mask` are left-padded to a common `seq_len`.
/// - Each step generates one token per sequence until EOS or `max_new_tokens`.
/// - Once a sequence hits EOS, we keep feeding an EOS token for it so we can
///   continue decoding the remaining batch with a shared KV cache.
///
/// Returns generated token ids (excluding the input prompt ids), one vec per sample.
pub fn greedy_generate_cached_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
    };
    greedy_generate_cached_batch_impl(thinker, device, input_ids, attention_mask, opts, None)
}

#[cfg(feature = "timing")]
pub fn greedy_generate_cached_batch_timed(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<(Vec<Vec<u32>>, GenerationTimings)> {
    let mut timings = GenerationTimings::default();
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
    };
    let out = greedy_generate_cached_batch_impl(
        thinker,
        device,
        input_ids,
        attention_mask,
        opts,
        Some(&mut timings),
    )?;
    timings.tokens_generated = out.iter().map(Vec::len).sum();
    Ok((out, timings))
}

struct GreedyGenerateBatchOpts<'a> {
    audio_features: Option<&'a Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &'a [u32],
}

fn greedy_generate_cached_batch_impl(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    opts: GreedyGenerateBatchOpts<'_>,
    #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    #[cfg(not(feature = "timing"))] _timings: Option<&mut GenerationTimings>,
) -> Result<Vec<Vec<u32>>> {
    let batch = input_ids.len();
    if batch == 0 {
        return Ok(vec![]);
    }
    if attention_mask.len() != batch {
        bail!(
            "attention_mask batch mismatch: expected={}, got={}",
            batch,
            attention_mask.len()
        );
    }

    let seq_len = input_ids
        .first()
        .map(|r| r.len())
        .ok_or_else(|| anyhow::anyhow!("input_ids missing first row"))?;
    if seq_len == 0 {
        bail!("input_ids rows are empty");
    }
    if opts.eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }

    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "input_ids[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }
    for (i, row) in attention_mask.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "attention_mask[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }

    if opts.max_new_tokens == 0 {
        return Ok(vec![vec![]; batch]);
    }

    let eos_fill_id = *opts
        .eos_token_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("eos_token_ids is empty"))?;

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    for i in 0..batch {
        let ids = input_ids
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing input_ids row {i}"))?;
        let attn = attention_mask
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing attention_mask row {i}"))?;
        ids_flat.extend_from_slice(ids);
        attn_flat.extend_from_slice(attn);
    }

    #[cfg(feature = "timing")]
    let start_tensors = std::time::Instant::now();
    let input_ids_t = Tensor::from_vec(ids_flat, (batch, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attn_flat, (batch, seq_len), device)?;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prompt_tensors_us = t
            .prompt_tensors_us
            .saturating_add(duration_to_us(start_tensors.elapsed()));
    }

    let mut kv_cache = KVCache::new();

    // Prefill the cache with the full prompt (including audio features).
    #[cfg(feature = "timing")]
    let start_prefill = std::time::Instant::now();
    let audio_placeholder_count = if opts.audio_features.is_some() {
        count_audio_placeholders_batch(input_ids, thinker.audio_token_id())?
    } else {
        0
    };
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        opts.audio_features,
        audio_placeholder_count,
    )?;
    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let logits = thinker.forward_embeds_with_kv_cache(
        &attention_mask_t,
        &position_ids,
        &inputs_embeds,
        &mut kv_cache,
    )?;

    let last_logits = logits.narrow(1, seq_len.saturating_sub(1), 1)?.squeeze(1)?;
    let mut next_ids = last_logits.argmax(1usize)?.to_vec1::<u32>()?;
    if next_ids.len() != batch {
        bail!(
            "internal error: batch argmax mismatch: expected={batch}, got={}",
            next_ids.len()
        );
    }
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_us = t
            .prefill_us
            .saturating_add(duration_to_us(start_prefill.elapsed()));
    }

    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch];
    let mut finished: Vec<bool> = next_ids
        .iter()
        .map(|id| opts.eos_token_ids.contains(id))
        .collect();

    let prompt_lens = prompt_lens_from_attention_masks(attention_mask, seq_len)?;
    let ones_col = Tensor::ones((batch, 1usize), DType::U32, device)?;
    let mut attention_mask_total = attention_mask_t;
    let mut total_len = seq_len;

    #[cfg(feature = "timing")]
    let start_decode = std::time::Instant::now();
    for step in 0..opts.max_new_tokens {
        if finished.iter().all(|&x| x) {
            break;
        }
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.steps = t.steps.saturating_add(1);
        }

        let mut tokens_in: Vec<u32> = Vec::with_capacity(batch);
        for i in 0..batch {
            if finished.get(i).copied().unwrap_or(true) {
                tokens_in.push(eos_fill_id);
                continue;
            }

            let tok = next_ids
                .get(i)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing next_id for batch index {i}"))?;
            if opts.eos_token_ids.contains(&tok) {
                finished[i] = true;
                tokens_in.push(eos_fill_id);
            } else {
                generated[i].push(tok);
                tokens_in.push(tok);
            }
        }

        total_len = total_len
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("total_len overflow"))?;
        attention_mask_total = Tensor::cat(&[&attention_mask_total, &ones_col], 1)?;

        let input_ids_new = Tensor::from_vec(tokens_in, (batch, 1usize), device)?;
        let inputs_embeds_new = thinker.embed_tokens(&input_ids_new)?;

        let position_ids_new = position_ids_for_step(prompt_lens.as_slice(), step, device)?;

        let logits_new = thinker.forward_embeds_with_kv_cache(
            &attention_mask_total,
            &position_ids_new,
            &inputs_embeds_new,
            &mut kv_cache,
        )?;
        let next = logits_new.squeeze(1)?.argmax(1usize)?.to_vec1::<u32>()?;
        if next.len() != batch {
            bail!(
                "internal error: batch argmax mismatch: expected={batch}, got={}",
                next.len()
            );
        }
        next_ids = next;
    }
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.decode_us = t
            .decode_us
            .saturating_add(duration_to_us(start_decode.elapsed()));
    }

    Ok(generated)
}

fn count_audio_placeholders(ids: &[u32], audio_token_id: u32) -> usize {
    ids.iter().filter(|&&id| id == audio_token_id).count()
}

fn count_audio_placeholders_batch(input_ids: &[&[u32]], audio_token_id: u32) -> Result<usize> {
    let mut total = 0usize;
    for (i, row) in input_ids.iter().enumerate() {
        let n = row.iter().filter(|&&id| id == audio_token_id).count();
        total = total.checked_add(n).ok_or_else(|| {
            anyhow::anyhow!("audio placeholder count overflow at batch index {i}: adding {n}")
        })?;
    }
    Ok(total)
}

fn prompt_len_from_left_padded_mask(attention_mask: &[u32], seq_len: usize) -> Result<i64> {
    if attention_mask.len() != seq_len {
        bail!(
            "attention_mask length mismatch: expected={seq_len}, got={}",
            attention_mask.len()
        );
    }

    let mut sum: usize = 0;
    for &v in attention_mask {
        if v != 0 {
            sum = sum
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("attention_mask sum overflow"))?;
        }
    }

    i64::try_from(sum)
        .map_err(|_| anyhow::anyhow!("attention_mask length overflows i64: sum={sum}"))
}

fn prompt_lens_from_attention_masks(attention_mask: &[&[u32]], seq_len: usize) -> Result<Vec<i64>> {
    let batch = attention_mask.len();
    let mut out: Vec<i64> = Vec::with_capacity(batch);
    for (i, row) in attention_mask.iter().enumerate() {
        let len = prompt_len_from_left_padded_mask(row, seq_len).map_err(|e| {
            anyhow::anyhow!("failed to compute prompt length for attention_mask[{i}]: {e:#}")
        })?;
        out.push(len);
    }
    Ok(out)
}

fn position_ids_for_step(prompt_lens: &[i64], step: usize, device: &Device) -> Result<Tensor> {
    if prompt_lens.is_empty() {
        bail!("prompt_lens is empty");
    }
    let batch = prompt_lens.len();
    let step_i64 =
        i64::try_from(step).map_err(|_| anyhow::anyhow!("step overflows i64: {step}"))?;

    let mut pos: Vec<i64> = Vec::with_capacity(batch);
    for (i, &base) in prompt_lens.iter().enumerate() {
        let p = base.checked_add(step_i64).ok_or_else(|| {
            anyhow::anyhow!("position id overflow at batch index {i}: base={base} step={step_i64}")
        })?;
        pos.push(p);
    }

    let pos_t = Tensor::from_vec(pos, (batch, 1usize), device)?;
    Ok(pos_t.unsqueeze(0)?.broadcast_as((3usize, batch, 1usize))?)
}

#[cfg(test)]
mod tests {
    use super::{position_ids_for_step, prompt_lens_from_attention_masks};

    #[test]
    fn test_position_ids_for_step_matches_get_rope_index_last_col() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let masks: Vec<Vec<u32>> = vec![
            vec![0, 0, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1],
        ];

        let batch = masks.len();
        let seq_len = masks
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing masks"))?
            .len();
        let rows: Vec<&[u32]> = masks.iter().map(|r| r.as_slice()).collect();
        let prompt_lens = prompt_lens_from_attention_masks(rows.as_slice(), seq_len)?;

        for step in 0..4usize {
            let mut flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
            for row in &masks {
                flat.extend_from_slice(row.as_slice());
            }
            let base = candle::Tensor::from_vec(flat, (batch, seq_len), &device)?;

            let extra = step.saturating_add(1);
            let ones = candle::Tensor::ones((batch, extra), candle::DType::U32, &device)?;
            let mask_total = candle::Tensor::cat(&[&base, &ones], 1)?;
            let total_len = seq_len.saturating_add(extra);

            let (expected, _deltas) = crate::model::thinker::get_rope_index(&mask_total)?;
            let expected_last = expected.narrow(2, total_len.saturating_sub(1), 1)?;

            let got = position_ids_for_step(prompt_lens.as_slice(), step, &device)?;

            let exp = expected_last.to_vec3::<i64>()?;
            let got = got.to_vec3::<i64>()?;
            if exp != got {
                anyhow::bail!("position_ids mismatch at step {step}: expected={exp:?} got={got:?}");
            }
        }

        Ok(())
    }
}
