//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use candle::{CpuDevice, CpuStorage, test_utils::to_vec3_round};
//! use candle_nn::{LayerNorm, Module};
//! # fn main() -> candle::Result<()> {
//! type Tensor = candle::Tensor<CpuStorage>;
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &CpuDevice)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &CpuDevice)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &CpuDevice)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(
//!     to_vec3_round(&ys, 4)?,
//!     &[[[-1.2247, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]]);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450
use candle::{BackendStorage, DType, Module, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f64,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    pub remove_mean: bool,
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm<B: BackendStorage> {
    weight: Tensor<B>,
    bias: Option<Tensor<B>>,
    remove_mean: bool,
    eps: f64,
}

impl<B: BackendStorage> LayerNorm<B> {
    pub fn new(weight: Tensor<B>, bias: Tensor<B>, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor<B>, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    pub fn rms_norm(weight: Tensor<B>, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: false,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<B>> {
        self.bias.as_ref()
    }
}

impl<B: BackendStorage> Module<B> for LayerNorm<B> {
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        if x.is_contiguous() && self.remove_mean {
            if let Some(bias) = self.bias.as_ref() {
                return crate::ops::layer_norm(x, &self.weight, bias, self.eps as f32);
            }
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn layer_norm<C, B>(size: usize, config: C, vb: crate::VarBuilder<B>) -> Result<LayerNorm<B>>
where
    C: Into<LayerNormConfig>,
    B: BackendStorage,
    B::Device: candle::TryConvertStorage<candle::CpuStorage, B>,
{
    let config = config.into();
    let weight = vb.get_with_hints(size, "weight", crate::Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", crate::Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

pub fn layer_norm_no_bias<B>(
    size: usize,
    eps: f64,
    vb: crate::VarBuilder<B>,
) -> Result<LayerNorm<B>>
where
    B: BackendStorage,
    B::Device: candle::TryConvertStorage<candle::CpuStorage, B>,
{
    let config = LayerNormConfig {
        eps,
        remove_mean: true,
        affine: false,
    };
    layer_norm(size, config, vb)
}

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm<B: BackendStorage>(LayerNorm<B>);

impl<B: BackendStorage> RmsNorm<B> {
    pub fn new(weight: Tensor<B>, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    pub fn into_inner(self) -> LayerNorm<B> {
        self.0
    }

    /// Faster variant of the forward kernel, this can only be used on contiguous tensors though.
    pub fn forward_diff(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        self.0.forward(xs)
    }
}

impl<B: BackendStorage> Module<B> for RmsNorm<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        if xs.is_contiguous() {
            crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)
        } else {
            self.0.forward(xs)
        }
    }
}

pub fn rms_norm<B>(size: usize, eps: f64, vb: crate::VarBuilder<B>) -> Result<RmsNorm<B>>
where
    B: BackendStorage,
    B::Device: candle::TryConvertStorage<candle::CpuStorage, B>,
{
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(size, config, vb)?))
}
