//! Batch Normalization.
//!
//! This layer applies Batch Normalization over a mini-batch of inputs as described in [`Batch
//! Normalization`]. The input is expected to have at least three dimensions.
//!
//! Note that this implementation is for inference only, there is no possibility to track the
//! running stats.
//!
//! [`Batch Normalization`]: https://arxiv.org/abs/1502.03167
use crate::Init;
use candle::{DType, Module, Result, Tensor, Var};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormConfig {
    pub eps: f64,
    pub remove_mean: bool,

    /// The meaning of affine here is different from LayerNorm: when false there is no learnable
    /// parameter at all, 1 used for gamma and 0 for beta.
    pub affine: bool,

    /// Controls exponential moving average of running stats. Defaults to 0.1
    ///
    /// `running_stat * (1.0 - momentum) + stat * momentum`.
    pub momentum: f64,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        }
    }
}

impl From<f64> for BatchNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BatchNorm {
    running_mean: Var,
    running_var: Var,
    weight_and_bias: Option<(Tensor, Tensor)>,
    remove_mean: bool,
    eps: f64,
    momentum: f64,
}

impl BatchNorm {
    pub fn new(
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {eps}")
        }
        if running_mean.rank() != 1 {
            candle::bail!(
                "batch-norm running mean must have 1 dimension, has {}",
                running_mean.dims().len()
            )
        }
        if running_mean.dims() != running_var.dims() {
            candle::bail!(
                "batch-norm running mean shape {:?} does not match running variance shape {:?}",
                running_mean.shape(),
                running_var.shape(),
            )
        }
        if running_mean.dims() != weight.dims() {
            candle::bail!(
                "batch-norm running mean shape {:?} does not match weight shape {:?}",
                running_mean.shape(),
                weight.shape(),
            )
        }
        if running_mean.dims() != bias.dims() {
            candle::bail!(
                "batch-norm running mean shape {:?} does not match bias shape {:?}",
                running_mean.shape(),
                bias.shape(),
            )
        }
        Ok(Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            momentum,
        })
    }

    pub fn new_no_bias(
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {eps}")
        }
        if running_mean.dims().len() != 1 {
            candle::bail!(
                "batch-norm running mean must have 1 dimension, has {}",
                running_mean.dims().len()
            )
        }
        if running_mean.dims() != running_var.dims() {
            candle::bail!(
                "batch-norm running mean shape {:?} does not match running variance shape {:?}",
                running_mean.shape(),
                running_var.shape(),
            )
        }
        Ok(Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            momentum,
        })
    }

    pub fn running_mean(&self) -> &Tensor {
        self.running_mean.as_tensor()
    }

    pub fn running_var(&self) -> &Tensor {
        self.running_var.as_tensor()
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn weight_and_bias(&self) -> Option<(&Tensor, &Tensor)> {
        self.weight_and_bias.as_ref().map(|v| (&v.0, &v.1))
    }

    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    pub fn forward_learning(&self, x: &Tensor) -> Result<Tensor> {
        let num_features = self.running_mean.as_tensor().dim(0)?;
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
        if x.dim(1)? != num_features {
            candle::bail!(
                "batch-norm input doesn't have the expected number of features ({:?} <> {})",
                x.shape(),
                num_features
            )
        }
        let x = x.to_dtype(internal_dtype)?;
        let x = x.transpose(0, 1)?;
        let x_dims_post_transpose = x.dims();
        let x = x.flatten_from(1)?.contiguous()?;
        let x = if self.remove_mean {
            let mean_x = x.mean_keepdim(1)?;
            {
                // Update running mean
                let new_mean = ((self.running_mean.as_tensor() * (1.0 - self.momentum))?
                    + (mean_x.flatten_all()? * self.momentum)?)?;

                self.running_mean.set(&new_mean)?;
            }
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = x.sqr()?.mean_keepdim(1)?;
        {
            // Update running variance
            let batch_size = x.dim(1)? as f64;
            let running_var_weight = 1.0 - self.momentum;
            let norm_x_weight = self.momentum * batch_size / (batch_size - 1.0);

            let new_var = ((self.running_var.as_tensor() * running_var_weight)?
                + (&norm_x.flatten_all()? * norm_x_weight)?)?;

            self.running_var.set(&new_var)?;
        }
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?;
        let x = match &self.weight_and_bias {
            None => x,
            Some((weight, bias)) => {
                let weight = weight.reshape(((), 1))?;
                let bias = bias.reshape(((), 1))?;
                x.broadcast_mul(&weight)?.broadcast_add(&bias)?
            }
        };
        x.reshape(x_dims_post_transpose)?.transpose(0, 1)
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            self.forward_learning(x)
        } else {
            self.forward(x)
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let target_shape: Vec<usize> = x
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();
        let target_shape = target_shape.as_slice();

        let x = x
            .broadcast_sub(&self.running_mean.as_tensor().reshape(target_shape)?)?
            .broadcast_div(
                &(self.running_var.as_tensor().reshape(target_shape)? + self.eps)?.sqrt()?,
            )?;

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
    let running_mean = vb.get_with_hints(num_features, "running_mean", Init::Const(0.))?;
    let running_var = vb.get_with_hints(num_features, "running_var", Init::Const(1.))?;
    let weight_and_bias = if config.affine {
        let weight = vb.get_with_hints(num_features, "weight", Init::Const(1.))?;
        let bias = vb.get_with_hints(num_features, "bias", Init::Const(0.))?;
        Some((weight, bias))
    } else {
        None
    };
    Ok(BatchNorm {
        running_mean: Var::from_tensor(&running_mean)?,
        running_var: Var::from_tensor(&running_var)?,
        weight_and_bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
        momentum: config.momentum,
    })
}
