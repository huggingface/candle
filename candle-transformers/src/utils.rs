//! Shared utilities: repeat_kv, repeat_penalty, no_repeat_ngram, causal mask.

use candle::{DType, Device, Result, Tensor};

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

/// Build an *additive* causal attention mask of shape `(1, 1, tgt, tgt + offset)`,
/// where masked positions are `-inf` and visible ones are `0`.
///
/// This is the counterpart to [`build_causal_mask`] for models that apply the mask
/// with `scores.broadcast_add(&mask)` rather than by selecting on a `u8` mask.
///
/// The buffer holds `tgt * (tgt + offset)` values and does not depend on the batch
/// size, so the leading dim stays `1` and is broadcast over the batch by the caller.
/// Shaping it `(b, 1, ..)` instead claims `b` times more elements than the buffer
/// holds, which `Tensor::from_slice` does not reject; see
/// https://github.com/huggingface/candle/issues/3582.
///
/// `sw` optionally restricts each query to a sliding window of `sw` preceding keys.
pub fn build_additive_causal_mask(
    tgt: usize,
    offset: usize,
    sw: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<_> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                let past_ok = j <= i + offset;
                // Within the window iff `(i + offset) - j <= w`, rearranged to
                // `j + w >= i + offset` to stay in `usize` (no signed casts, no
                // subtraction underflow when `j > i + offset`).
                let sw_ok = match sw {
                    Some(w) => j + w >= i + offset,
                    None => true,
                };
                if past_ok && sw_ok {
                    0.
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), device)?.to_dtype(dtype)
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Bans tokens that would complete a previously-seen n-gram of length `ngram_size`
/// (equivalent to Hugging Face's `no_repeat_ngram_size`).
///
/// Given the already-generated `context`, the last `ngram_size - 1` tokens form the current
/// prefix. Any token that has previously followed that exact prefix in `context` has its
/// logit set to `-inf` so it cannot be sampled, preventing the model from repeating an
/// n-gram it has already produced.
///
/// `ngram_size == 0`, or a `context` shorter than `ngram_size`, is a no-op. `ngram_size == 1`
/// bans every token already present in `context`.
///
/// Use it in a decode loop just like [`apply_repeat_penalty`], applying it to the logits
/// before sampling the next token:
///
/// ```rust
/// use candle::{Device, Tensor};
/// use candle_transformers::utils::apply_no_repeat_ngram;
///
/// # fn main() -> candle::Result<()> {
/// let device = Device::Cpu;
/// // Tokens generated so far. The last token (`1`) forms the current 1-token prefix.
/// let context = [1u32, 2, 1];
/// let logits = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &device)?;
///
/// // With `ngram_size = 2`, the bigram `[1, 2]` already occurred, so `2` is banned
/// // as the next token after `1` (its logit becomes -inf) and will not be sampled.
/// let logits = apply_no_repeat_ngram(&logits, 2, &context)?;
/// assert_eq!(logits.to_vec1::<f32>()?, [1.0, 1.0, f32::NEG_INFINITY, 1.0]);
/// # Ok(())
/// # }
/// ```
pub fn apply_no_repeat_ngram(
    logits: &Tensor,
    ngram_size: usize,
    context: &[u32],
) -> Result<Tensor> {
    if ngram_size == 0 || context.len() < ngram_size {
        return Ok(logits.clone());
    }
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    // The prefix that must not be extended into a repeated n-gram.
    let prefix = &context[context.len() - (ngram_size - 1)..];
    // Scan every complete n-gram already in the context; when its first `ngram_size - 1`
    // tokens match the current prefix, its final token is a banned continuation.
    for window in context.windows(ngram_size) {
        if &window[..ngram_size - 1] == prefix {
            let banned = window[ngram_size - 1];
            if let Some(logit) = logits.get_mut(banned as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression test for https://github.com/huggingface/candle/issues/3582
    #[test]
    fn additive_causal_mask_is_batch_independent_and_broadcasts() {
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(3, 0, None, &device, DType::F32).unwrap();
        assert_eq!(
            mask.dims(),
            &[1, 1, 3, 3],
            "mask must carry a leading batch dim of 1 so broadcast_add applies it to every row",
        );

        // Broadcasting onto a 2-sequence batch must mask both rows identically and causally.
        let scores = Tensor::zeros((2, 1, 3, 3), DType::F32, &device).unwrap();
        let masked = scores
            .broadcast_add(&mask)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(
            masked[0], masked[1],
            "both batch rows must receive the same causal mask",
        );
        assert_eq!(masked[0][0], vec![0.0, neg, neg]);
        assert_eq!(masked[0][1], vec![0.0, 0.0, neg]);
        assert_eq!(masked[0][2], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn additive_causal_mask_handles_kv_cache_offset() {
        // With `offset` cached keys the mask is rectangular: every query attends to
        // all cached prefix keys, then the usual causal triangle over the new tokens.
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(2, 2, None, &device, DType::F32)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(mask[0], vec![0.0, 0.0, 0.0, neg]);
        assert_eq!(mask[1], vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn additive_causal_mask_respects_sliding_window() {
        // With a sliding window of `w`, query i may attend to key j only when it is
        // both causal (j <= i + offset) and within the window (i + offset - j <= w).
        let device = Device::Cpu;
        let neg = f32::NEG_INFINITY;

        let mask = build_additive_causal_mask(4, 0, Some(1), &device, DType::F32)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // window = 1: each query sees itself and the single preceding position.
        assert_eq!(mask[0], vec![0.0, neg, neg, neg]);
        assert_eq!(mask[1], vec![0.0, 0.0, neg, neg]);
        assert_eq!(mask[2], vec![neg, 0.0, 0.0, neg]);
        assert_eq!(mask[3], vec![neg, neg, 0.0, 0.0]);
    }
}
