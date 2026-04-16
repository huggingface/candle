//! Batch Normalization.
//!
//! This layer applies Batch Normalization over a mini-batch of inputs as described in [`Batch
//! Normalization`]. The input is expected to have at least three dimensions.
//!
//! Note that this implementation is for inference only, there is no possibility to track the
//! running stats.
//!
//! [`Batch Normalization`]: https://arxiv.org/abs/1502.03167
use candle::{DType, Result, Tensor, Var};

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
            ..Default::default()
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
    fn check_validity(&self, num_features: usize) -> Result<()> {
        if self.eps < 0. {
            candle::bail!("batch-norm eps cannot be negative {}", self.eps)
        }
        if !(0.0..=1.0).contains(&self.momentum) {
            candle::bail!(
                "batch-norm momentum must be between 0 and 1, is {}",
                self.momentum
            )
        }
        if self.running_mean.dims() != [num_features] {
            candle::bail!(
                "batch-norm running mean has unexpected shape {:?} should have shape [{num_features}]",
                self.running_mean.shape(),
            )
        }
        if self.running_var.dims() != [num_features] {
            candle::bail!(
                "batch-norm running variance has unexpected shape {:?} should have shape [{num_features}]",
                self.running_var.shape(),
            )
        }
        if let Some((ref weight, ref bias)) = self.weight_and_bias.as_ref() {
            if weight.dims() != [num_features] {
                candle::bail!(
                    "batch-norm weight has unexpected shape {:?} should have shape [{num_features}]",
                    weight.shape(),
                )
            }
            if bias.dims() != [num_features] {
                candle::bail!(
                    "batch-norm weight has unexpected shape {:?} should have shape [{num_features}]",
                    bias.shape(),
                )
            }
        }
        Ok(())
    }

    pub fn new(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            momentum: 0.1,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    pub fn new_no_bias(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            momentum: 0.1,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    pub fn new_with_momentum(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: Some((weight, bias)),
            remove_mean: true,
            eps,
            momentum,
        };
        out.check_validity(num_features)?;
        Ok(out)
    }

    pub fn new_no_bias_with_momentum(
        num_features: usize,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let out = Self {
            running_mean: Var::from_tensor(&running_mean)?,
            running_var: Var::from_tensor(&running_var)?,
            weight_and_bias: None,
            remove_mean: true,
            eps,
            momentum,
        };
        out.check_validity(num_features)?;
        Ok(out)
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

    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
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
        // Flatten all the dimensions exception the channel one as this performs a Spatial Batch
        // Normalization.
        let x = x.flatten_from(1)?.contiguous()?;
        let x = if self.remove_mean {
            // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
            let mean_x = x.mean_keepdim(1)?;
            let updated_running_mean = ((self.running_mean.as_tensor() * (1.0 - self.momentum))?
                + (mean_x.flatten_all()? * self.momentum)?)?;
            self.running_mean.set(&updated_running_mean)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
        let norm_x = x.sqr()?.mean_keepdim(1)?;
        let updated_running_var = {
            let batch_size = x.dim(1)? as f64;
            let running_var_weight = 1.0 - self.momentum;
            let norm_x_weight = self.momentum * batch_size / (batch_size - 1.0);
            ((self.running_var.as_tensor() * running_var_weight)?
                + (&norm_x.flatten_all()? * norm_x_weight)?)?
        };
        self.running_var.set(&updated_running_var)?;
        let x = x
            .broadcast_div(&(norm_x + self.eps)?.sqrt()?)?
            .to_dtype(x_dtype)?;
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

    fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        let target_shape: Vec<usize> = x
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();
        let target_shape = target_shape.as_slice();

        let x = x
            .broadcast_sub(
                &self
                    .running_mean
                    .as_detached_tensor()
                    .reshape(target_shape)?,
            )?
            .broadcast_div(
                &(self
                    .running_var
                    .as_detached_tensor()
                    .reshape(target_shape)?
                    + self.eps)?
                    .sqrt()?,
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

impl crate::ModuleT for BatchNorm {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            self.forward_train(x)
        } else {
            self.forward_eval(x)
        }
    }
}

pub fn batch_norm<C: Into<BatchNormConfig>>(
    num_features: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<BatchNorm> {
    use crate::Init;
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
