use anyhow::{Context, Result};
use candle::{Device, Shape, Var};
mod test_utils;

fn simple_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4.], device)?;
    let x = x.as_tensor();
    let y = (((x * x)? + x * 5f64)? + 4f64)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(x.to_vec1::<f32>()?, [3., 1., 4.]);
    // y = x^2 + 5.x + 4
    assert_eq!(y.to_vec1::<f32>()?, [28., 10., 40.]);
    // dy/dx = 2.x + 5
    assert_eq!(grad_x.to_vec1::<f32>()?, [11., 7., 13.]);
    Ok(())
}

fn sum_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4.], device)?;
    let x = x.as_tensor();
    let y = (x.sqr()?.sum_keepdim(0)? * 2.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1::<f32>()?, [52.]);
    // y = 2.x^2 so dy/dx = 4.x
    assert_eq!(grad_x.to_vec1::<f32>()?, &[12., 4., 16.]);

    // Same test as before but squeezing on the last dimension.
    let y = (x.sqr()?.sum_keepdim(0)? * 2.)?.squeeze(0)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_scalar::<f32>()?, 52.);
    // y = 2.x^2 so dy/dx = 4.x
    assert_eq!(grad_x.to_vec1::<f32>()?, &[12., 4., 16.]);
    Ok(())
}

fn matmul_grad(device: &Device) -> Result<()> {
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let x = Var::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let y = Var::from_slice(&data, (2, 3, 2), device)?;
    let c = x.matmul(&y)?;
    let grads = c.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    let grad_y = grads.get(&y).context("no grad for y")?;
    assert_eq!(grad_x.shape(), &Shape::from((2, 2, 3)));
    assert_eq!(grad_y.shape(), &Shape::from((2, 3, 2)));
    assert_eq!(
        &*grad_x.to_vec3::<f32>()?,
        &[
            [[1., 5., 9.], [1., 5., 9.]],
            [[13., 17., 21.], [13., 17., 21.]]
        ]
    );
    assert_eq!(
        &*grad_y.to_vec3::<f32>()?,
        &[
            [[3., 3.], [5., 5.], [7., 7.]],
            [[15., 15.], [17., 17.], [19., 19.]]
        ]
    );
    Ok(())
}

// The simplest gradient descent, using scalar variable.
fn grad_descent(device: &Device) -> Result<()> {
    let x = Var::new(0f32, device)?;
    let learning_rate = 0.1;
    for _step in 0..100 {
        let xt = x.as_tensor();
        let c = ((xt - 4.2)? * (xt - 4.2)?)?;
        let grads = c.backward()?;
        let x_grad = grads.get(&x).context("no grad for x")?;
        x.set(&(xt - x_grad * learning_rate)?)?
    }
    assert_eq!(x.to_scalar::<f32>()?, 4.199999);
    Ok(())
}

fn unary_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4., 0.15], device)?;
    let x = x.as_tensor();
    let y = (x.log()? + 1.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1::<f32>()?, [2.0986123, 1.0, 2.3862944, -0.89712]);
    assert_eq!(grad_x.to_vec1::<f32>()?, [0.33333334, 1.0, 0.25, 6.6666665]);
    let y = x.exp()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(
        y.to_vec1::<f32>()?,
        [20.085537, 2.7182817, 54.59815, 1.1618342]
    );
    assert_eq!(
        grad_x.to_vec1::<f32>()?,
        [20.085537, 2.7182817, 54.59815, 1.1618342]
    );
    let y = x.exp()?.sqr()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(
        y.to_vec1::<f32>()?,
        [403.4288, 7.3890557, 2980.9578, 1.3498588]
    );
    // exp(x)^2 = exp(2*x)
    assert_eq!(
        grad_x.to_vec1::<f32>()?,
        [806.8576, 14.778111, 5961.9155, 2.6997175]
    );
    Ok(())
}

test_device!(simple_grad, simple_grad_cpu, simple_grad_gpu);
test_device!(sum_grad, sum_grad_cpu, sum_grad_gpu);
test_device!(matmul_grad, matmul_grad_cpu, matmul_grad_gpu);
test_device!(grad_descent, grad_descent_cpu, grad_descent_gpu);
test_device!(unary_grad, unary_grad_cpu, unary_grad_gpu);
