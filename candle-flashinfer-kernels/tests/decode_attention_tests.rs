use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use candle_flashinfer_kernels::flashinfer_decode_attention;

fn softmax_last_dim(att: &Tensor) -> Result<Tensor> {
    let max = att.max_keepdim(D::Minus1)?;
    let e = att.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&sum)?)
}

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
    let att = softmax_last_dim(&att)?;
    let out = att.matmul(&v)?.squeeze(2)?;
    Ok(out)
}

fn run_case(device: &Device) -> Result<()> {
    let (b, hq, hkv, lk, d) = (2usize, 4usize, 2usize, 17usize, 32usize);
    let q = Tensor::randn(0f32, 1f32, (b, hq, d), device)?;
    let k = Tensor::randn(0f32, 1f32, (b, hkv, lk, d), device)?;
    let v = Tensor::randn(0f32, 1f32, (b, hkv, lk, d), device)?;
    let scale = 1f32 / (d as f32).sqrt();

    let got = flashinfer_decode_attention(&q, &k, &v, scale)?;
    let want = decode_attention_reference(&q, &k, &v, scale)?;

    let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-3, "max abs diff {diff}");
    Ok(())
}

#[test]
fn decode_attention_cpu_matches_reference() -> Result<()> {
    run_case(&Device::Cpu)
}

#[test]
fn decode_attention_cuda_matches_reference() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };
    run_case(&device)
}

#[cfg(feature = "metal")]
#[test]
fn decode_attention_metal_matches_reference() -> Result<()> {
    let device = Device::new_metal(0)?;
    run_case(&device)
}
