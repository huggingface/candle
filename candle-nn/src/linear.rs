//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{CpuDevice, CpuStorage};
//! type Tensor = candle::Tensor<CpuStorage>;
//!
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &CpuDevice)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &CpuDevice)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{BackendStorage, Result, Tensor};

#[derive(Clone, Debug)]
pub struct Linear<B: BackendStorage> {
    weight: Tensor<B>,
    bias: Option<Tensor<B>>,
}

impl<B: BackendStorage> Linear<B> {
    pub fn new(weight: Tensor<B>, bias: Option<Tensor<B>>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<B>> {
        self.bias.as_ref()
    }
}

impl<B: BackendStorage> super::Module<B> for Linear<B> {
    fn forward(&self, x: &Tensor<B>) -> candle::Result<Tensor<B>> {
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

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear<B: BackendStorage>(
    in_dim: usize,
    out_dim: usize,
    vb: crate::VarBuilder<B>,
) -> Result<Linear<B>> {
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
pub fn linear_no_bias<B: BackendStorage>(
    in_dim: usize,
    out_dim: usize,
    vb: crate::VarBuilder<B>,
) -> Result<Linear<B>> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b<B: BackendStorage>(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder<B>,
) -> Result<Linear<B>> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}
