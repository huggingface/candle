//! Shared utilities: repeat_kv, repeat_penalty, causal mask.

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

/// Build an additive causal attention bias of shape `(b, 1, tgt, tgt + offset)`:
/// `0.` where attention is allowed and `-inf` where it is masked, for use with
/// `broadcast_add` on attention scores. Queries attend to keys `j <= i + offset`,
/// optionally restricted to a sliding window of `sw` positions back.
///
/// Unlike [`build_causal_mask`], the data is built once and expanded across the
/// batch (a zero-copy view), so every batch row carries the same valid mask.
pub fn additive_causal_mask(
    b: usize,
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
                let sw_ok = match sw {
                    Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
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
    Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), device)?
        .to_dtype(dtype)?
        .expand((b, 1, tgt, tgt + offset))
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
    use super::additive_causal_mask;
    use candle::{DType, Device};

    const NI: f32 = f32::NEG_INFINITY;

    /// Regression test for the per-model `causal_mask` helpers this replaces: the
    /// data was built batch-independent but declared with a batched shape, so
    /// batch rows >= 1 read garbage (see #3582).
    #[test]
    fn batched_rows_are_all_valid_with_offset() -> candle::Result<()> {
        let m = additive_causal_mask(3, 2, 1, None, &Device::Cpu, DType::F32)?;
        assert_eq!(m.dims(), [3, 1, 2, 3]);
        let rows = m.reshape((3, 2, 3))?.to_vec3::<f32>()?;
        let expected = vec![vec![0., 0., NI], vec![0., 0., 0.]];
        for (b, row) in rows.iter().enumerate() {
            assert_eq!(row, &expected, "wrong mask for batch row {b}");
        }
        Ok(())
    }

    #[test]
    fn sliding_window_limits_lookback() -> candle::Result<()> {
        let m = additive_causal_mask(1, 4, 0, Some(1), &Device::Cpu, DType::F32)?;
        let rows = m.reshape((4, 4))?.to_vec2::<f32>()?;
        let expected = vec![
            vec![0., NI, NI, NI],
            vec![0., 0., NI, NI],
            vec![NI, 0., 0., NI],
            vec![NI, NI, 0., 0.],
        ];
        assert_eq!(rows, expected);
        Ok(())
    }
}
