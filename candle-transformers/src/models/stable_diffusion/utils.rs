use candle::{Device, Result, Tensor};
use candle_nn::Module;

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

// Wrap the conv2d op to provide some tracing.
#[derive(Debug)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    span: tracing::Span,
}

impl Conv2d {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}
