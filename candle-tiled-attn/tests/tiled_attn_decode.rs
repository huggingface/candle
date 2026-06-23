#[cfg(feature = "cuda")]
use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn cuda_device_or_skip() -> Result<Option<Device>> {
    match Device::new_cuda(0) {
        Ok(device) => Ok(Some(device)),
        Err(err) => {
            eprintln!("skipping tiled_attn_decode CUDA test: {err}");
            Ok(None)
        }
    }
}

#[cfg(feature = "cuda")]
fn patterned_tensor(
    shape: (usize, usize, usize, usize),
    phase: f32,
    device: &Device,
) -> Result<Tensor> {
    let elem_count = shape.0 * shape.1 * shape.2 * shape.3;
    let data = (0..elem_count)
        .map(|i| {
            let wave = ((i as f32 + phase) * 0.013).sin() * 0.25;
            let bias = ((i % 17) as f32 - 8.0) * 0.002;
            wave + bias
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(data, shape, device)?.to_dtype(DType::F16)
}

#[cfg(feature = "cuda")]
fn reference(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let q = q.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let k = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let v = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * softmax_scale as f64)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(&v)?.to_dtype(DType::F16)
}

#[cfg(feature = "cuda")]
#[test]
fn decode_matches_reference_attention() -> Result<()> {
    let Some(device) = cuda_device_or_skip()? else {
        return Ok(());
    };

    for (b, h, kv_len, d) in [(1, 2, 7, 256), (2, 1, 9, 512), (1, 1, 1, 256)] {
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = patterned_tensor((b, h, 1, d), 0.0, &device)?;
        let k = patterned_tensor((b, h, kv_len, d), 10_000.0, &device)?;
        let v = patterned_tensor((b, h, kv_len, d), 20_000.0, &device)?;

        let got = candle_tiled_attn::tiled_attn_decode(&q, &k, &v, scale)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?;
        let expected = reference(&q, &k, &v, scale)?.to_dtype(DType::F32)?;
        let diff = (got - expected)?.abs()?;
        let max_abs = diff.max_all()?.to_scalar::<f32>()?;
        let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;

        assert!(
            max_abs.is_finite() && mean_abs.is_finite() && max_abs <= 5e-2 && mean_abs <= 5e-3,
            "shape B={b} H={h} K={kv_len} D={d}: max_abs={max_abs:.6} mean_abs={mean_abs:.6}"
        );
    }

    Ok(())
}
