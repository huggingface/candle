//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle::{Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = candle_nn::Embedding::new(embeddings, d2);
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
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new(weight, bias, eps))
}

pub fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps, span })
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }

    /// Fused add + RMSNorm: computes `(x + h, RMSNorm(x + h) * self.weight)`.
    /// On the Vulkan backend with f32 inputs this is a single fused dispatch;
    /// otherwise it falls back to the decomposed add + RMSNorm.
    pub fn fused_add_forward(&self, x: &Tensor, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        if x.device().is_vulkan() && x.dtype() == candle::DType::F32 {
            return x.fused_add_rms_norm(h, &self.weight, self.eps as f32);
        }
        let x_new = (x + h)?;
        let x_norm = candle_nn::ops::rms_norm(&x_new, &self.weight, self.eps as f32)?;
        Ok((x_new, x_norm))
    }

    /// Fused post-norm + residual add: computes `x + RMSNorm(h) * weight`
    /// (Gemma-3 style: the sublayer output `h` is normalized first and then
    /// added to the residual `x`). On the Vulkan backend with f32 inputs
    /// this is a single fused dispatch; otherwise it falls back to the
    /// decomposed RMSNorm + add.
    pub fn post_add_forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        if x.device().is_vulkan() && x.dtype() == candle::DType::F32 {
            return x.fused_post_norm_add(h, &self.weight, self.eps as f32);
        }
        let h_norm = candle_nn::ops::rms_norm(h, &self.weight, self.eps as f32)?;
        x + &h_norm
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
/// Concatenates the rows (dim 0) of the quantized weight tensors named
/// `names` into a single QTensor so that one matmul computes every
/// projection (fused qkv / gate+up): the byte layout of a row-major
/// quantized `(n, k)` matrix is a plain concatenation of row blocks, so
/// the concatenation is valid when all tensors share the quantization
/// dtype and the shared (last) dim is a multiple of the dtype's block
/// size. Returns `Ok(None)` when the tensors cannot be fused; the caller
/// then loads them separately.
pub fn cat_quantized_weights<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    names: &[String],
    device: &candle::Device,
) -> Result<Option<(QTensor, Vec<usize>)>> {
    let mut parts: Vec<(Vec<u8>, GgmlDType, candle::Shape)> = Vec::with_capacity(names.len());
    for name in names {
        parts.push(ct.tensor_raw(reader, name)?);
    }
    let (dtype, dims_shape) = (parts[0].1, parts[0].2.clone());
    let dims: Vec<usize> = dims_shape.dims().to_vec();
    let k = *dims
        .last()
        .ok_or_else(|| candle::Error::Msg("cat_quantized_weights: empty dims".to_string()))?;
    if !k.is_multiple_of(dtype.block_size()) {
        return Ok(None);
    }
    for p in &parts[1..] {
        if p.1 != dtype || p.2.dims().to_vec() != dims {
            return Ok(None);
        }
    }
    let ns: Vec<usize> = parts.iter().map(|p| p.2.dims()[0]).collect();
    let mut bytes = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
    for p in &parts {
        bytes.extend_from_slice(&p.0);
    }
    let mut shape = dims;
    shape[0] = ns.iter().sum();
    let storage = QStorage::from_data(bytes.into(), device, dtype)?;
    let qt = QTensor::new(storage, shape)?;
    Ok(Some((qt, ns)))
}
