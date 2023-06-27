use anyhow::{Context, Result};
use candle::{Device, Shape, Tensor};

#[test]
fn simple_grad() -> Result<()> {
    let x = Tensor::var(&[3f32, 1., 4.], &Device::Cpu)?;
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

#[test]
fn matmul_grad() -> Result<()> {
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let x = Tensor::var_from_slice(&data, (2, 2, 3), &Device::Cpu)?;
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let y = Tensor::var_from_slice(&data, (2, 3, 2), &Device::Cpu)?;

    let c = x.matmul(&y)?;
    let grads = c.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    let grad_y = grads.get(&y).context("no grad for y")?;
    assert_eq!(grad_x.shape(), &Shape::from((2, 2, 3)));
    assert_eq!(grad_y.shape(), &Shape::from((2, 3, 2)));
    assert_eq!(
        &*grad_x.storage_data::<f32>()?,
        &[1., 5., 9., 1., 5., 9., 13., 17., 21., 13., 17., 21.]
    );
    assert_eq!(
        &*grad_y.storage_data::<f32>()?,
        &[3., 3., 5., 5., 7., 7., 15., 15., 17., 17., 19., 19.]
    );
    Ok(())
}
