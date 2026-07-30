//! Shared utilities: repeat_kv, repeat_penalty, no_repeat_ngram, causal mask.

use candle::{Device, Result, Tensor};

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
