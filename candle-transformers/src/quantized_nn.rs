//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::QTensor;
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
    /// LoRA down-projection matrix (in_dim × rank) - stored as f32
    lora_a: Option<Tensor>,
    /// LoRA up-projection matrix (rank × out_dim) - stored as f32
    lora_b: Option<Tensor>,
    /// LoRA scaling factor: (alpha / rank) * strength
    lora_scale: Option<f32>,
}

impl Linear {
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self {
            weight,
            bias,
            lora_a: None,
            lora_b: None,
            lora_scale: None,
        })
    }

    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
            lora_a: None,
            lora_b: None,
            lora_scale: None,
        }
    }

    /// Set LoRA parameters for this layer
    ///
    /// # Arguments
    /// * `lora_a` - Down-projection matrix (in_dim × rank)
    /// * `lora_b` - Up-projection matrix (rank × out_dim)
    /// * `scale` - Pre-computed scaling factor: (alpha / rank) * strength
    pub fn set_lora(&mut self, lora_a: Tensor, lora_b: Tensor, scale: f32) -> Result<()> {
        // Validate shapes
        let a_shape = lora_a.dims();
        let b_shape = lora_b.dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            candle::bail!(
                "LoRA matrices must be 2D, got shapes: {:?}, {:?}",
                a_shape,
                b_shape
            );
        }

        if a_shape[1] != b_shape[0] {
            candle::bail!(
                "LoRA matrix dimensions incompatible: A={:?}, B={:?}",
                a_shape,
                b_shape
            );
        }

        tracing::debug!("set_lora called: a={:?}, b={:?}, scale={}", a_shape, b_shape, scale);

        self.lora_a = Some(lora_a);
        self.lora_b = Some(lora_b);
        self.lora_scale = Some(scale);
        Ok(())
    }

    /// Remove LoRA parameters
    pub fn clear_lora(&mut self) {
        self.lora_a = None;
        self.lora_b = None;
        self.lora_scale = None;
    }

    /// Check if LoRA is active
    pub fn has_lora(&self) -> bool {
        self.lora_a.is_some() && self.lora_b.is_some()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        // Base quantized matmul
        let mut y = x.apply(&self.weight)?;

        // Apply LoRA correction if present
        if let (Some(a), Some(b), Some(scale)) = (&self.lora_a, &self.lora_b, self.lora_scale) {
            static LORA_FORWARD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let count = LORA_FORWARD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 100 == 0 {  // Log every 100th call to avoid spam
                tracing::info!("LoRA forward pass called {} times", count + 1);
            }
            tracing::trace!("Applying LoRA in forward: a={:?}, b={:?}, scale={}", a.dims(), b.dims(), scale);

            // Compute low-rank correction: (x @ A) @ B * scale
            // Handle multi-dimensional inputs by preserving leading dimensions
            let original_shape = x.dims().to_vec();
            let x_reshaped = if original_shape.len() > 2 {
                // Flatten to 2D: [..., in_dim] → [prod(...), in_dim]
                let in_dim = original_shape[original_shape.len() - 1];
                let batch_size: usize = original_shape[..original_shape.len() - 1].iter().product();
                x.reshape((batch_size, in_dim))?
            } else {
                x.clone()
            };

            let lora_out = x_reshaped.matmul(a)?.matmul(b)?;

            // Reshape back to original shape if needed
            let lora_out = if original_shape.len() > 2 {
                let mut target_shape = original_shape.clone();
                target_shape[original_shape.len() - 1] = b.dim(1)?;
                lora_out.reshape(target_shape.as_slice())?
            } else {
                lora_out
            };

            let scaled = (lora_out * scale as f64)?;

            // Log the magnitude of LoRA effect for debugging
            if let (Ok(y_mean), Ok(lora_mean)) = (y.mean_all()?.to_scalar::<f32>(), scaled.mean_all()?.to_scalar::<f32>()) {
                tracing::trace!("LoRA effect - base_mean: {:.6}, lora_mean: {:.6}, ratio: {:.2}%",
                    y_mean, lora_mean, (lora_mean / y_mean.abs().max(1e-8)) * 100.0);
            }

            y = y.add(&scaled)?;
        }

        // Apply bias if present
        match &self.bias {
            None => Ok(y),
            Some(bias) => y.broadcast_add(bias),
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
    Ok(Linear {
        weight,
        bias,
        lora_a: None,
        lora_b: None,
        lora_scale: None,
    })
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
        lora_a: None,
        lora_b: None,
        lora_scale: None,
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
    Ok(Linear {
        weight,
        bias: None,
        lora_a: None,
        lora_b: None,
        lora_scale: None,
    })
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
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
