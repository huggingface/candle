use anyhow::{Context, Result};
use candle::{Device, Tensor};

#[test]
fn simple_grad() -> Result<()> {
    let x = Tensor::var(&[3f32, 1., 4.], Device::Cpu)?;
    let five = Tensor::new(&[5f32, 5., 5.], Device::Cpu)?;
    let y = x.mul(&x)?.add(&x.mul(&five)?)?.add(&five)?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x.id()).context("no grad for x")?;
    assert_eq!(x.to_vec1::<f32>()?, [3., 1., 4.]);
    // y = x^2 + 5.x + 5
    assert_eq!(y.to_vec1::<f32>()?, [29., 11., 41.]);
    // dy/dx = 2.x + 5
    assert_eq!(grad_x.to_vec1::<f32>()?, [11., 7., 13.]);
    Ok(())
}
