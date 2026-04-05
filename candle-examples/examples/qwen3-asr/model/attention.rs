//! Attention primitives (causal masks, GQA helpers, etc.).

use candle::{DType, Device, Result, Tensor};

/// Repeat key/value heads to match the query head count (GQA/MQA).
///
/// This is equivalent to PyTorch's `repeat_interleave` over the head dimension
/// used by HuggingFace's `repeat_kv`.
pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch, kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    let expanded = hidden_states.unsqueeze(2)?; // (b, kv_heads, 1, s, d)
    let expanded = expanded.broadcast_as((batch, kv_heads, n_rep, seq_len, head_dim))?;
    expanded.reshape((batch, kv_heads * n_rep, seq_len, head_dim))
}

/// Create an additive causal attention mask for eager attention.
///
/// Returns a `(batch, 1, seq_len, seq_len)` mask where values are:
/// - `0.0` for allowed attention positions
/// - a large negative value (dtype min) for masked positions
///
/// This mirrors the eager mask path in Transformers 4.57 (`create_causal_mask` -> `eager_mask`)
/// for the common no-cache prefill case.
pub fn make_causal_mask(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if seq_len == 0 {
        return Tensor::zeros((batch_size, 1usize, 0usize, 0usize), dtype, device);
    }

    let allowed = Tensor::tril2(seq_len, DType::U8, device)?; // (seq, seq)
    let allowed = allowed
        .unsqueeze(0)?
        .broadcast_as((batch_size, seq_len, seq_len))?; // (batch, seq, seq)

    let allowed = match attention_mask {
        None => allowed,
        Some(m) => {
            let (b, s) = m.dims2()?;
            if b != batch_size || s != seq_len {
                candle::bail!(
                    "attention_mask dims mismatch: expected=({batch_size},{seq_len}), got=({b},{s})"
                );
            }

            let key_padding = m.ne(0u32)?; // (batch, seq)
            let key_padding = key_padding
                .unsqueeze(1)?
                .broadcast_as((batch_size, seq_len, seq_len))?; // (batch, seq, seq)
            (&allowed * &key_padding)?
        }
    };

    let allowed = allowed.unsqueeze(1)?; // (batch, 1, seq, seq)
    let shape = (batch_size, 1usize, seq_len, seq_len);
    let zeros = Tensor::zeros(shape, DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, shape, device)?;
    let mask = allowed.where_cond(&zeros, &neg)?;

    if dtype == DType::F32 {
        Ok(mask)
    } else {
        mask.to_dtype(dtype)
    }
}

/// Create an additive causal attention mask for cached decoding.
///
/// Returns a `(batch, 1, new_len, total_len)` mask where:
/// - `total_len = cache_len + new_len`
/// - Each new query position `q` (0..new_len) can attend to key positions
///   up to and including its absolute position `cache_len + q`.
/// - If `attention_mask` is provided, key positions where the mask is `0` are
///   always masked out.
pub fn make_causal_mask_cached(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    cache_len: usize,
    new_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = cache_len.saturating_add(new_len);
    if new_len == 0 {
        return Tensor::zeros((batch_size, 1usize, 0usize, total_len), dtype, device);
    }

    let total_len_u32 = u32::try_from(total_len)
        .map_err(|_| candle::Error::Msg(format!("total_len overflows u32: {total_len}")))?;
    let cache_len_u32 = u32::try_from(cache_len)
        .map_err(|_| candle::Error::Msg(format!("cache_len overflows u32: {cache_len}")))?;
    let new_len_u32 = u32::try_from(new_len)
        .map_err(|_| candle::Error::Msg(format!("new_len overflows u32: {new_len}")))?;
    let q_end = cache_len_u32.checked_add(new_len_u32).ok_or_else(|| {
        candle::Error::Msg(format!(
            "cache_len + new_len overflows u32: cache_len={cache_len} new_len={new_len}"
        ))
    })?;

    let kv = Tensor::arange(0u32, total_len_u32, device)?;
    let q = Tensor::arange(cache_len_u32, q_end, device)?;
    let kv = kv.unsqueeze(0)?.broadcast_as((new_len, total_len))?;
    let q = q.unsqueeze(1)?.broadcast_as((new_len, total_len))?;

    let allowed = kv.le(&q)?; // (new_len, total_len) u8
    let allowed = allowed
        .unsqueeze(0)?
        .broadcast_as((batch_size, new_len, total_len))?; // (batch, new_len, total_len)

    let allowed = match attention_mask {
        None => allowed,
        Some(m) => {
            let (b, s) = m.dims2()?;
            if b != batch_size || s != total_len {
                candle::bail!(
                    "attention_mask dims mismatch: expected=({batch_size},{total_len}), got=({b},{s})"
                );
            }
            let key_padding = m.ne(0u32)?; // (batch, total_len)
            let key_padding = key_padding
                .unsqueeze(1)?
                .broadcast_as((batch_size, new_len, total_len))?;
            (&allowed * &key_padding)?
        }
    };

    let allowed = allowed.unsqueeze(1)?; // (batch, 1, new_len, total_len)
    let shape = (batch_size, 1usize, new_len, total_len);
    let zeros = Tensor::zeros(shape, DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, shape, device)?;
    let mask = allowed.where_cond(&zeros, &neg)?;

    if dtype == DType::F32 {
        Ok(mask)
    } else {
        mask.to_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::{make_causal_mask, make_causal_mask_cached, repeat_kv};

    #[test]
    fn test_repeat_kv_repeats_heads() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let x = candle::Tensor::arange(0f32, 2.0 * 3.0 * 4.0, &device)?.reshape((2, 3, 2, 2))?;
        let y = repeat_kv(&x, 2)?;
        let (b, h, s, d) = y.dims4()?;
        if (b, h, s, d) != (2, 6, 2, 2) {
            anyhow::bail!("unexpected repeat_kv dims: {:?}", y.dims());
        }
        Ok(())
    }

    #[test]
    fn test_make_causal_mask_shape_and_values() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let batch = 1usize;
        let seq = 4usize;
        let mask = make_causal_mask(None, batch, seq, candle::DType::F32, &device)?;
        if mask.dims() != vec![batch, 1, seq, seq] {
            anyhow::bail!("unexpected mask dims: {:?}", mask.dims());
        }
        let m = mask.squeeze(1)?.to_vec3::<f32>()?;
        for (q, row) in m[0].iter().enumerate() {
            for (kv, &v) in row.iter().enumerate() {
                if kv <= q {
                    if v != 0.0 {
                        anyhow::bail!("expected 0 at ({q},{kv}), got {v}");
                    }
                } else if v != f32::MIN {
                    anyhow::bail!("expected f32::MIN at ({q},{kv}), got {v}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_make_causal_mask_cached_shape_and_values() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let batch = 1usize;
        let cache_len = 3usize;
        let new_len = 2usize;
        let total_len = cache_len + new_len;

        let mask =
            make_causal_mask_cached(None, batch, cache_len, new_len, candle::DType::F32, &device)?;
        if mask.dims() != vec![batch, 1, new_len, total_len] {
            anyhow::bail!("unexpected mask dims: {:?}", mask.dims());
        }

        let m = mask.squeeze(1)?.to_vec3::<f32>()?;
        // q=0 => abs_q=3, disallow kv>3
        for (kv, &v) in m[0][0].iter().enumerate() {
            if kv <= 3 {
                if v != 0.0 {
                    anyhow::bail!("expected 0 at (q=0,kv={kv}), got {v}");
                }
            } else if v != f32::MIN {
                anyhow::bail!("expected f32::MIN at (q=0,kv={kv}), got {v}");
            }
        }
        // q=1 => abs_q=4, all kv allowed
        for (kv, &v) in m[0][1].iter().enumerate() {
            if v != 0.0 {
                anyhow::bail!("expected 0 at (q=1,kv={kv}), got {v}");
            }
        }

        Ok(())
    }
}
