use candle::{DType, Device, Result, Tensor};

#[test]
fn bf16_add_sizes() -> Result<()> {
    let dev = Device::Cpu;
    for n in [1024usize, 32 * 1024, 100 * 1024] {
        let x = Tensor::rand(-2f32, 2f32, (n,), &dev)?;
        let xb = x.to_dtype(DType::BF16)?;
        let y = Tensor::rand(-2f32, 2f32, (n,), &dev)?;
        let yb = y.to_dtype(DType::BF16)?;
        let want = (&x + &y)?;
        let got = (&xb + &yb)?.to_dtype(DType::F32)?;
        let d = (&want - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("n={n}: max diff {d:.4}");
    }
    Ok(())
}
