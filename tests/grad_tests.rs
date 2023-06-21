use anyhow::{Context, Result};
use candle::{Device, Tensor};

#[test]
fn simple_grad() -> Result<()> {
    let x = Tensor::var(&[3f32, 1., 4.], Device::Cpu)?;
    let y = (((&x * &x)? + &x * 5f64)? + 4f64)?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(x.to_vec1::<f32>()?, [3., 1., 4.]);
    // y = x^2 + 5.x + 4
    assert_eq!(y.to_vec1::<f32>()?, [28., 10., 40.]);
    // dy/dx = 2.x + 5
    assert_eq!(grad_x.to_vec1::<f32>()?, [11., 7., 13.]);
    Ok(())
}
