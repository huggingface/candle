//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::Linear;
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use crate::init::{DefaultInit, Initializer, ModelInitializer};
use candle::{Result, Tensor};

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let w = match x.dims() {
            &[bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

impl ModelInitializer for Linear {}

impl Initializer<Linear> for DefaultInit {
    type Config = ((usize, usize), bool);

    fn init(&mut self, (shape, has_bias): Self::Config) -> Result<Linear> {
        let dtype = self.dtype();
        let device = self.device().clone();
        let (out_dim, in_dim) = shape;
        let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
        let ws = init_ws.var(shape, dtype, &device)?;
        self.push_var(ws.clone());
        let ws = ws.as_tensor().clone();
        if has_bias {
            let bound = 1. / (in_dim as f64).sqrt();
            let init_bs = crate::Init::Uniform {
                lo: -bound,
                up: bound,
            };
            let bs = init_bs.var(out_dim, dtype, &device)?;
            self.push_var(bs.clone());
            let bs = bs.as_tensor().clone();
            Ok(Linear::new(ws, Some(bs)))
        } else {
            Ok(Linear::new(ws, None))
        }
    }
}

/// Loads a linear layer.
///
/// This uses some default names for weight and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vs: crate::VarBuilder) -> Result<Linear> {
    let ws = vs.get((out_dim, in_dim), "weight")?;
    let bs = vs.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vs: crate::VarBuilder) -> Result<Linear> {
    let ws = vs.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}
