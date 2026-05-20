//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        // When possible, we avoid using a broadcasted matmul as it is much slower
        // than the standard matmul for the cuda and cpu backends.
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Linear layer with weights quantized to {-1, 0, +1} (ternary).
///
/// Weights are quantized at construction: values whose absolute value exceeds
/// `threshold` become ±1; the rest become 0. This matches the BitNet b1.58
/// scheme (arxiv.org/abs/2402.17764, §3.1). The quantized weights are stored
/// as `f32` and the forward pass is identical to [`Linear`] — no special
/// backend required.
///
/// ```rust
/// use candle::{Tensor, Device::Cpu, DType};
/// use candle_nn::LinearTernary;
/// # fn main() -> candle::Result<()> {
///
/// let w = Tensor::new(&[[2.0f32, -3.0], [0.1, 1.5]], &Cpu)?;
/// // threshold = mean(|w|) ≈ 1.65 → |2.0| > 1.65 → +1, |-3.0| > 1.65 → -1, |0.1| < 1.65 → 0, |1.5| < 1.65 → 0
/// let layer = LinearTernary::from_tensor(w, None, None)?;
/// let xs = Tensor::new(&[[1.0f32, 1.0]], &Cpu)?;
/// let ys = layer.forward(&xs)?;
/// # Ok(()) }
/// ```
#[derive(Clone, Debug)]
pub struct LinearTernary {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearTernary {
    /// Quantize `weight` to {-1, 0, +1} using `threshold`.
    ///
    /// If `threshold` is `None`, the mean absolute value of the weights is used
    /// (the BitNet b1.58 prescription). The quantized weights are stored as `f32`.
    pub fn from_tensor(
        weight: Tensor,
        bias: Option<Tensor>,
        threshold: Option<f64>,
    ) -> Result<Self> {
        let t = match threshold {
            Some(v) => v,
            None => weight.abs()?.mean_all()?.to_scalar::<f32>()? as f64,
        };
        let t_tensor = Tensor::full(t as f32, weight.shape(), weight.device())?;
        let abs_w = weight.abs()?;
        // mask: 1 where |w| > threshold, 0 elsewhere
        let mask = abs_w.gt(&t_tensor)?.to_dtype(candle::DType::F32)?;
        // ternary = sign(w) * mask  →  values in {-1, 0, +1}
        let sign_w = weight.sign()?;
        let ternary = sign_w.mul(&mask)?;
        Ok(Self { weight: ternary, bias })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Fraction of weights that are exactly zero (0.0–1.0).
    pub fn sparsity(&self) -> Result<f32> {
        let numel = self.weight.elem_count();
        let zeros = self.weight.eq(0f32)?.to_dtype(candle::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        Ok(zeros / numel as f32)
    }
}

impl super::Module for LinearTernary {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?.matmul(&w)?.reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?.matmul(&w)?.reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}
