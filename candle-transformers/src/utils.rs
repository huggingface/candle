//! Shared utilities: repeat_kv, repeat_penalty, causal mask.

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
    use candle::{Device, IndexOp};

    #[test]
    fn test_build_causal_mask_square() -> Result<()> {
        let mask = build_causal_mask(4, 0, &Device::Cpu)?;
        assert_eq!(mask.dims(), [4, 4]);
        assert_eq!(
            mask.to_vec2::<u8>()?,
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0],]
        );
        Ok(())
    }

    #[test]
    fn test_build_causal_mask_with_index_pos() -> Result<()> {
        let mask = build_causal_mask(2, 3, &Device::Cpu)?;
        assert_eq!(mask.dims(), [2, 5]);
        assert_eq!(mask.to_vec2::<u8>()?, [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0],]);
        Ok(())
    }

    #[test]
    fn test_build_causal_mask_single_token() -> Result<()> {
        let mask = build_causal_mask(1, 0, &Device::Cpu)?;
        assert_eq!(mask.dims(), [1, 1]);
        assert_eq!(mask.to_vec2::<u8>()?, [[0]]);

        let mask = build_causal_mask(1, 5, &Device::Cpu)?;
        assert_eq!(mask.dims(), [1, 6]);
        assert_eq!(mask.to_vec2::<u8>()?, [[0, 0, 0, 0, 0, 0]]);
        Ok(())
    }

    #[test]
    fn test_apply_repeat_penalty_positive_logits() -> Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let result = apply_repeat_penalty(&logits, 2.0, &[1, 3])?;
        assert_eq!(result.to_vec1::<f32>()?, [1.0, 1.0, 3.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_apply_repeat_penalty_negative_logits() -> Result<()> {
        let logits = Tensor::new(&[-1.0f32, 2.0, -3.0, 4.0], &Device::Cpu)?;
        let result = apply_repeat_penalty(&logits, 2.0, &[0, 2])?;
        assert_eq!(result.to_vec1::<f32>()?, [-2.0, 2.0, -6.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_apply_repeat_penalty_no_op_cases() -> Result<()> {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;

        let result = apply_repeat_penalty(&logits, 2.0, &[])?;
        assert_eq!(result.to_vec1::<f32>()?, [1.0, 2.0, 3.0]);

        let result = apply_repeat_penalty(&logits, 1.0, &[0, 1, 2])?;
        assert_eq!(result.to_vec1::<f32>()?, [1.0, 2.0, 3.0]);

        let result = apply_repeat_penalty(&logits, 2.0, &[1, 1, 1])?;
        assert_eq!(result.to_vec1::<f32>()?, [1.0, 1.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_repeat_kv() -> Result<()> {
        let t = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((1, 2, 3, 4))?;

        let out = repeat_kv(t.clone(), 1)?;
        assert_eq!(out.dims(), [1, 2, 3, 4]);
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            t.flatten_all()?.to_vec1::<f32>()?
        );

        let out = repeat_kv(t, 2)?;
        assert_eq!(out.dims(), [1, 4, 3, 4]);
        let head0 = out.i((0, 0))?.to_vec2::<f32>()?;
        let head1 = out.i((0, 1))?.to_vec2::<f32>()?;
        let head2 = out.i((0, 2))?.to_vec2::<f32>()?;
        let head3 = out.i((0, 3))?.to_vec2::<f32>()?;
        assert_eq!(head0, head1);
        assert_eq!(head2, head3);
        assert_ne!(head0, head2);

        Ok(())
    }
}
