use candle::{Device, Result, Tensor};

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Add sigmoid and inv as binary ops.
    xs.ones_like()? / (xs.neg()?.exp()? - 1.0)?
}

pub fn avg_pool2d(_: &Tensor) -> Result<Tensor> {
    todo!()
}

pub fn pad(_: &Tensor) -> Result<Tensor> {
    todo!()
}

pub fn upsample_nearest2d(_: &Tensor) -> Result<Tensor> {
    todo!()
}

pub fn linspace(start: f64, stop: f64, steps: usize) -> Result<Tensor> {
    if steps < 1 {
        candle::bail!("cannot use linspace with steps {steps} <= 1")
    }
    let delta = (stop - start) / (steps - 1) as f64;
    let vs = (0..steps)
        .map(|step| start + step as f64 * delta)
        .collect::<Vec<_>>();
    Tensor::from_vec(vs, steps, &Device::Cpu)
}
