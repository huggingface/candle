use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
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

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}
