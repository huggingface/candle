use candle::{quantized::QuantizedBackend, BackendStorage, Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Embedding<B: BackendStorage> {
    inner: candle_nn::Embedding<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> Embedding<B> {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder<B>) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor<B>) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor<B> {
        self.inner.embeddings()
    }
}

impl<B: BackendStorage> Module<B> for Embedding<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear<B: BackendStorage> {
    inner: candle_nn::Linear<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> Linear<B> {
    pub fn from_weights(weights: Tensor<B>, bias: Option<Tensor<B>>) -> Self {
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

pub fn linear_b<B: BackendStorage>(
    d1: usize,
    d2: usize,
    b: bool,
    vb: VarBuilder<B>,
) -> Result<Linear<B>> {
    let inner = candle_nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear<B: BackendStorage>(d1: usize, d2: usize, vb: VarBuilder<B>) -> Result<Linear<B>> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias<B: BackendStorage>(
    d1: usize,
    d2: usize,
    vb: VarBuilder<B>,
) -> Result<Linear<B>> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl<B: BackendStorage> Module<B> for Linear<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct Conv2d<B: BackendStorage> {
    inner: candle_nn::Conv2d<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> Module<B> for Conv2d<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d<B: BackendStorage>(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder<B>,
) -> Result<Conv2d<B>> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul<B: BackendStorage, QB: QuantizedBackend> {
    inner: candle::quantized::QMatMul<B, QB>,
    span: tracing::Span,
}

impl<B, QB> QMatMul<B, QB>
where
    B: BackendStorage,
    QB: QuantizedBackend<Storage = B>,
{
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder<QB>,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle::quantized::QMatMul::<B, QB>::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    pub fn from_weights(ws: std::sync::Arc<candle::quantized::QTensor<QB>>) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl<B: BackendStorage, QB: QuantizedBackend> Module<B> for QMatMul<B, QB>
where
    candle::quantized::QMatMul<B, QB>: Module<B>,
{
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl<B: BackendStorage, QB: QuantizedBackend> std::fmt::Debug for QMatMul<B, QB> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm<B: BackendStorage> {
    inner: candle_nn::LayerNorm<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> LayerNorm<B> {
    pub fn new(weight: Tensor<B>, bias: Tensor<B>, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self { inner, span }
    }
}

impl<B: BackendStorage> Module<B> for LayerNorm<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

pub fn layer_norm<B: BackendStorage, C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder<B>,
) -> Result<LayerNorm<B>> {
    let inner = candle_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone)]
pub struct RmsNorm<B: BackendStorage> {
    inner: candle_nn::RmsNorm<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> RmsNorm<B> {
    pub fn new(size: usize, eps: f64, vb: VarBuilder<B>) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    pub fn forward_diff(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward_diff(x)
    }
}

impl<B: BackendStorage> Module<B> for RmsNorm<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}
