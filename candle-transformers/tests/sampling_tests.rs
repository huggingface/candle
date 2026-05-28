use candle::{test_device, Device, Result, Tensor};
use candle_transformers::generation::sampling::apply_repeat_penalty;

fn repeat_penalty(device: &Device) -> Result<()> {
    // Positive logits get divided, negative logits multiplied, and tokens not in ctx are left alone.
    let logits = Tensor::from_slice(&[1.0f32, -1.0, 2.0, -3.0, 0.5], 5, device)?;
    let ctx = Tensor::from_slice(&[0u32, 1, 4], 3, device)?;
    let out = apply_repeat_penalty(&logits, &ctx, 2.0)?;
    let v = out.to_vec1::<f32>()?;
    assert_eq!(v, vec![0.5, -2.0, 2.0, -3.0, 0.25]);
    Ok(())
}

fn repeat_penalty_empty_ctx(device: &Device) -> Result<()> {
    let logits = Tensor::from_slice(&[1.0f32, -1.0, 2.0], 3, device)?;
    let ctx = Tensor::from_slice(&[] as &[u32], 0, device)?;
    let out = apply_repeat_penalty(&logits, &ctx, 2.0)?;
    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, -1.0, 2.0]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn repeat_penalty_f16_cuda() -> Result<()> {
    use candle::DType;
    let device = Device::new_cuda(0)?;
    let logits =
        Tensor::from_slice(&[1.0f32, -1.0, 2.0, -3.0, 0.5], 5, &device)?.to_dtype(DType::F16)?;
    let ctx = Tensor::from_slice(&[0u32, 1, 4], 3, &device)?;
    let out = apply_repeat_penalty(&logits, &ctx, 2.0)?;
    let v = out.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v, vec![0.5, -2.0, 2.0, -3.0, 0.25]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn repeat_penalty_bf16_cuda() -> Result<()> {
    use candle::DType;
    let device = Device::new_cuda(0)?;
    let logits =
        Tensor::from_slice(&[1.0f32, -1.0, 2.0, -3.0, 0.5], 5, &device)?.to_dtype(DType::BF16)?;
    let ctx = Tensor::from_slice(&[0u32, 1, 4], 3, &device)?;
    let out = apply_repeat_penalty(&logits, &ctx, 2.0)?;
    let v = out.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v, vec![0.5, -2.0, 2.0, -3.0, 0.25]);
    Ok(())
}

test_device!(
    repeat_penalty,
    repeat_penalty_cpu,
    repeat_penalty_gpu,
    repeat_penalty_metal
);
#[test]
fn repeat_penalty_empty_cpu() -> Result<()> {
    repeat_penalty_empty_ctx(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn repeat_penalty_empty_gpu() -> Result<()> {
    repeat_penalty_empty_ctx(&Device::new_cuda(0)?)
}
