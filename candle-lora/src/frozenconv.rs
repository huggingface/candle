use candle::{Module, Result, Tensor};
use candle_nn::{Conv1dConfig, Conv2dConfig};

use crate::{Conv1dLayerLike, Conv2dLayerLike};

/// Conv1d, but with a `new` implementation that ensures the weights are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl FrozenConv1d {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: Conv1dConfig,
    ) -> Result<Self> {
        Ok(Self {
            weight: weight.detach()?,
            bias: bias.cloned(), //Bias is still trainable
            config,
        })
    }

    pub(crate) fn new_from_conv1d(old: &dyn Conv1dLayerLike) -> Result<Self> {
        Self::new(old.weight(), old.bias(), *old.config())
    }

    pub(crate) fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for FrozenConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

impl Conv1dLayerLike for FrozenConv1d {
    fn config(&self) -> &Conv1dConfig {
        &self.config
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
}

/// Conv2d, but with a `new` implementation that ensures the weights are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl FrozenConv2d {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            weight: weight.detach()?,
            bias: bias.cloned(), //Bias is still trainable
            config,
        })
    }

    pub(crate) fn new_from_conv2d(old: &dyn Conv2dLayerLike) -> Result<Self> {
        Self::new(old.weight(), old.bias(), *old.config())
    }

    pub(crate) fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for FrozenConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

impl Conv2dLayerLike for FrozenConv2d {
    fn config(&self) -> &Conv2dConfig {
        &self.config
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
}
