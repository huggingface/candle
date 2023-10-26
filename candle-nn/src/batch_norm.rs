//! Batch Normalization.
//!
//! This layer applies Batch Normalization over a mini-batch of inputs as described in [`Batch
//! Normalization`]. The input is expected to have at least three dimensions.
//!
//! Note that this implementation is for inference only, there is no possibility to track the
//! running stats.
//!
//! [`Batch Normalization`]: https://arxiv.org/abs/1502.03167
use candle::{DType, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormConfig {
    pub eps: f64,
    pub remove_mean: bool,
    /// The meaning of affine here is different from LayerNorm: when false there is no learnable
    /// parameter at all, 1 used for gamma and 0 for beta.
    pub affine: bool,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for BatchNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BatchNorm {
    running_mean: Tensor,
    running_var: Tensor,
    weight_and_bias: Option<(Tensor, Tensor)>,
    remove_mean: bool,
    eps: f64,
    num_features: usize,
}

impl BatchNorm {
    pub fn new(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    ) -> Result<Self> {
        if eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {eps}")
        }
        if weight.dims() != [num_features] {
            candle::bail!(
                "batch-norm unexpected weight shape {:?} {num_features}",
                weight.shape()
            )
        }
        if bias.dims() != [num_features] {
            candle::bail!(
                "batch-norm unexpected bias shape {:?} {num_features}",
                bias.shape()
            )
        }
        Ok(Self {
            running_mean,
            running_var,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            num_features,
        })
    }

    pub fn new_no_bias(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
    ) -> Result<Self> {
        if eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {eps}")
        }
        Ok(Self {
            running_mean,
            running_var,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            num_features,
        })
    }

    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }

    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }

    pub fn weight_and_bias(&self) -> Option<(&Tensor, &Tensor)> {
        self.weight_and_bias.as_ref().map(|v| (&v.0, &v.1))
    }

    pub fn forward_learning(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        if x.rank() < 2 {
            candle::bail!(
                "batch-norm input tensor must have at least two dimensions ({:?})",
                x.shape()
            )
        }
        if x.dim(1)? != self.num_features {
            candle::bail!(
                "batch-norm input doesn't have the expected number of features ({:?} <> {})",
                x.shape(),
                self.num_features
            )
        }
        let x = x.to_dtype(internal_dtype)?;
        let x = x.transpose(0, 1)?;
        let x_dims_post_transpose = x.dims();
        let x = x.flatten_from(1)?.contiguous()?;
        let x = if self.remove_mean {
            let mean_x = x.mean_keepdim(1)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = x.sqr()?.mean_keepdim(1)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?;
        let x = match &self.weight_and_bias {
            None => x,
            Some((weight, bias)) => {
                let weight = weight.reshape((self.num_features, 1))?;
                let bias = bias.reshape((self.num_features, 1))?;
                x.broadcast_mul(&weight)?.broadcast_add(&bias)?
            }
        };
        x.reshape(x_dims_post_transpose)?.transpose(0, 1)
    }
}

impl crate::Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let target_shape: Vec<usize> = x
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();
        let target_shape = target_shape.as_slice();
        let x = x
            .broadcast_sub(&self.running_mean.reshape(target_shape)?)?
            .broadcast_div(&(self.running_var.reshape(target_shape)? + self.eps)?.sqrt()?)?;
        match &self.weight_and_bias {
            None => Ok(x),
            Some((weight, bias)) => {
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                x.broadcast_mul(&weight)?.broadcast_add(&bias)
            }
        }
    }
}

pub fn batch_norm<C: Into<BatchNormConfig>>(
    num_features: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<BatchNorm> {
    let config = config.into();
    if config.eps < 0. {
        candle::bail!("batch-norm eps cannot be negative {}", config.eps)
    }
    let running_mean = vb.get_with_hints(num_features, "running_mean", crate::Init::Const(0.))?;
    let running_var = vb.get_with_hints(num_features, "running_var", crate::Init::Const(1.))?;
    let weight_and_bias = if config.affine {
        let weight = vb.get_with_hints(num_features, "weight", crate::Init::Const(1.))?;
        let bias = vb.get_with_hints(num_features, "bias", crate::Init::Const(0.))?;
        Some((weight, bias))
    } else {
        None
    };
    Ok(BatchNorm {
        running_mean,
        running_var,
        weight_and_bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
        num_features,
    })
}
