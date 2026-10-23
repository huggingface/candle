use candle::{DType, Device, Result, Tensor};

fn rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a - &b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let s = a.abs()?.max_all()?.to_scalar::<f32>()?.max(1.0);
    Ok(d / s)
}

#[test]
fn bf16_cpu_op_sweep() -> Result<()> {
    let dev = Device::Cpu;
    let x32 = Tensor::rand(-2f32, 2f32, (4, 8, 2560), &dev)?;
    let xb = x32.to_dtype(DType::BF16)?;

    let checks: Vec<(&str, f32)> = vec![
        (
            "silu",
            rel(&candle_nn::ops::silu(&x32)?, &candle_nn::ops::silu(&xb)?)?,
        ),
        (
            "softmax",
            rel(
                &candle_nn::ops::softmax_last_dim(&x32)?,
                &candle_nn::ops::softmax_last_dim(&xb)?,
            )?,
        ),
        ("mul", rel(&(&x32 * &x32)?, &(&xb * &xb)?)?),
        ("add", rel(&(&x32 + &x32)?, &(&xb + &xb)?)?),
        ("exp", rel(&x32.exp()?, &xb.exp()?)?),
    ];
    let w32 = Tensor::rand(0.5f32, 1.5f32, (2560,), &dev)?;
    let rms32 = candle_nn::ops::rms_norm(&x32, &w32, 1e-6)?;
    let rmsb = candle_nn::ops::rms_norm(&xb, &w32.to_dtype(DType::BF16)?, 1e-6)?;
    let mut checks = checks;
    checks.push(("rms_norm", rel(&rms32, &rmsb)?));

    for (name, r) in &checks {
        println!("{name}: rel {r:.4}");
        assert!(*r < 3e-2, "{name} broken: rel {r}");
    }
    Ok(())
}

#[test]
fn bf16_cpu_matmul_works() -> Result<()> {
    let dev = Device::Cpu;
    let a = Tensor::rand(-1f32, 1f32, (2, 33, 64), &dev)?;
    let b = Tensor::rand(-1f32, 1f32, (2, 64, 17), &dev)?;
    let want = a.matmul(&b)?;
    let got = a
        .to_dtype(DType::BF16)?
        .matmul(&b.to_dtype(DType::BF16)?)?
        .to_dtype(DType::F32)?;
    let d = (&want - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(d < 0.15, "bf16 matmul diff {d}");
    Ok(())
}
