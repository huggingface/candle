use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use candle_flashinfer_kernels::flashinfer_decode_attention;

// Naive reference: softmax(q @ k^T * scale) @ v, with q holding a single query token per
// sequence and k/v fewer heads than q (grouped-query attention) repeated to match.
fn decode_attention_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (b, hq, d) = q.dims3()?;
    let (_, hkv, lk, _) = k.dims4()?;
    let group = hq / hkv;
    let k = k
        .unsqueeze(2)?
        .broadcast_as((b, hkv, group, lk, d))?
        .reshape((b, hq, lk, d))?
        .to_dtype(DType::F32)?;
    let v = v
        .unsqueeze(2)?
        .broadcast_as((b, hkv, group, lk, d))?
        .reshape((b, hq, lk, d))?
        .to_dtype(DType::F32)?;
    let q = q.to_dtype(DType::F32)?.unsqueeze(2)?; // (b, hq, 1, d)
    let att = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    let out = att.matmul(&v)?.squeeze(2)?;
    Ok(out)
}

#[test]
fn decode_attention_matches_reference() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };

    let (b, hq, hkv, lk, d) = (2usize, 4usize, 2usize, 17usize, 32usize);
    let q = Tensor::randn(0f32, 1f32, (b, hq, d), &device)?;
    let k = Tensor::randn(0f32, 1f32, (b, hkv, lk, d), &device)?;
    let v = Tensor::randn(0f32, 1f32, (b, hkv, lk, d), &device)?;
    let scale = 1f32 / (d as f32).sqrt();

    let got = flashinfer_decode_attention(&q, &k, &v, scale)?;
    let want = decode_attention_reference(&q, &k, &v, scale)?;

    let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "max abs diff {diff}");
    Ok(())
}
