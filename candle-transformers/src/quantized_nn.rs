//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::{QTensor, QuantizedBackend};
use candle::{Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Embedding<QB: QuantizedBackend> {
    inner: candle_nn::Embedding<QB::Storage>,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> Embedding<QB> {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder<QB>) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor<QB::Storage> {
        self.inner.embeddings()
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for Embedding<QB> {
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear<QB: QuantizedBackend> {
    weight: QMatMul<QB>,
    bias: Option<Tensor<QB::Storage>>,
}

impl<QB: QuantizedBackend> Linear<QB> {
    pub fn from_arc(
        weight: std::sync::Arc<QTensor<QB>>,
        bias: Option<Tensor<QB::Storage>>,
    ) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: QMatMul<QB>, bias: Option<Tensor<QB::Storage>>) -> Self {
        Self { weight, bias }
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for Linear<QB>
where
    QMatMul<QB>: Module<QB::Storage>,
{
    fn forward(&self, x: &Tensor<QB::Storage>) -> candle::Result<Tensor<QB::Storage>> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_b<QB: QuantizedBackend>(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder<QB>,
) -> Result<Linear<QB>> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

pub fn linear<QB: QuantizedBackend>(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder<QB>,
) -> Result<Linear<QB>> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

pub fn layer_norm<QB: QuantizedBackend>(
    size: usize,
    eps: f64,
    vb: VarBuilder<QB>,
) -> Result<candle_nn::LayerNorm<QB::Storage>> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new(weight, bias, eps))
}

pub fn layer_norm_no_bias<QB: QuantizedBackend>(
    size: usize,
    eps: f64,
    vb: VarBuilder<QB>,
) -> Result<candle_nn::LayerNorm<QB::Storage>> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
}

pub fn linear_no_bias<QB: QuantizedBackend>(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder<QB>,
) -> Result<Linear<QB>> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

#[derive(Debug, Clone)]
pub struct RmsNorm<QB: QuantizedBackend> {
    weight: Tensor<QB::Storage>,
    eps: f64,
    span: tracing::Span,
}

impl<QB: QuantizedBackend> RmsNorm<QB> {
    pub fn new(size: usize, eps: f64, vb: VarBuilder<QB>) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps, span })
    }

    pub fn from_qtensor(weight: QTensor<QB>, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for RmsNorm<QB> {
    fn forward(&self, x: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
