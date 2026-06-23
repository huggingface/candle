use candle::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

pub fn tiled_attn_decode(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    if !q.device().is_cuda() {
        candle::bail!("tiled_attn_decode requires CUDA tensors");
    }

    let (b, h, q_len, d) = q.dims4()?;
    let (kb, kh, k_len, kd) = k.dims4()?;
    let (vb, vh, v_len, vd) = v.dims4()?;

    if q_len != 1 {
        candle::bail!("tiled_attn_decode v1 supports only q_len=1");
    }
    if b != kb || b != vb || h != kh || h != vh || d != kd || d != vd || k_len != v_len {
        candle::bail!(
            "tiled_attn_decode shape mismatch: q={:?} k={:?} v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    if !matches!(d, 256 | 512) {
        candle::bail!("tiled_attn_decode supports D=256/512 for now, got {d}");
    }
    if q.dtype() != DType::F16 || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        candle::bail!(
            "tiled_attn_decode v1 supports only F16 tensors, got q={:?} k={:?} v={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
    }
    if k_len == 0 {
        return Tensor::zeros((b, h, q_len, d), q.dtype(), q.device());
    }

    if std::env::var("CANDLE_TILED_ATTN_TRACE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
    {
        eprintln!(
            "[candle_tiled_attn_decode_f16] q={:?} k={:?} v={:?} scale={}",
            q.shape(),
            k.shape(),
            v.shape(),
            softmax_scale
        );
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        cuda::tiled_attn_decode(&q, &k, &v, softmax_scale)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q, k, v, softmax_scale);
        candle::bail!("tiled_attn_decode requires the candle-tiled-attn cuda feature");
    }
}
