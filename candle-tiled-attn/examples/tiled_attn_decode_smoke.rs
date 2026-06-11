use anyhow::{bail, Result};
use candle::{DType, Device, Tensor};

fn reference(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> candle::Result<Tensor> {
    let q = q.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let k = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let v = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * softmax_scale as f64)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(&v)?.to_dtype(DType::F16)
}

fn check_shape(device: &Device, b: usize, h: usize, k_len: usize, d: usize) -> Result<()> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let q = Tensor::randn(0f32, 0.25, (b, h, 1, d), device)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 0.25, (b, h, k_len, d), device)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 0.25, (b, h, k_len, d), device)?.to_dtype(DType::F16)?;

    let got = candle_tiled_attn::tiled_attn_decode(&q, &k, &v, scale)?;
    let expected = reference(&q, &k, &v, scale)?;

    let diff = (got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
        - expected.to_dtype(DType::F32)?)?
    .abs()?;
    let max_abs = diff.max_all()?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    if !max_abs.is_finite() || !mean_abs.is_finite() || max_abs > 5e-2 || mean_abs > 5e-3 {
        bail!("shape B={b} H={h} K={k_len} D={d}: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    }

    println!("B={b} H={h} K={k_len} D={d}: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (b, h, k_len, d) in [
        (1, 8, 14, 256),
        (1, 8, 14, 512),
        (7, 8, 22, 256),
        (7, 8, 22, 512),
        (1, 4, 16, 256),
        (1, 4, 16, 512),
    ] {
        check_shape(&device, b, h, k_len, d)?;
    }
    Ok(())
}
