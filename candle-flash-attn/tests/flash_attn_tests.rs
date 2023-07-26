use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? / softmax_scale as f64)?;
    let att = att.softmax(D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 24, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 4))?;
    let k = (&q * 0.5)?;
    let v = (&q * 0.1)?;
    let ys = candle_flash_attn::flash_attn(&q, &k, &v, 0.5, false)?;
    let ys = ys.i(0)?.to_dtype(DType::F32)?;
    assert_eq!(ys.dims(), &[1]);
    assert_eq!(ys.to_vec3::<f32>()?, &[[[0f32]]]);
    Ok(())
}
