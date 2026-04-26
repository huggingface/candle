//! Shared utilities: repeat_kv, repeat_penalty, causal mask.
use candle::{DType, Device, Result, Tensor};
use std::collections::HashSet;

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

/// Apply repetition penalty to `logits` for tokens in `context`.
///
/// Tokens that appeared previously have their logits divided by `penalty`
/// when positive, or multiplied by `penalty` when negative.
pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    if context.is_empty() {
        return Ok(logits.clone());
    }
    // f32 only for now
    let logits = logits.to_dtype(DType::F32)?;
    let device = logits.device().clone();
    // Dedup so each GPU thread writes to a unique output position.
    let mut seen = HashSet::new();
    let unique: Vec<u32> = context
        .iter()
        .filter(|&&t| seen.insert(t))
        .copied()
        .collect();
    let ctx = Tensor::from_slice(&unique, unique.len(), &device)?;
    logits.apply_repeat_penalty(&ctx, penalty)
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
