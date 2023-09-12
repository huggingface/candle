use candle::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};

use crate::{Conv1dLayerLike, Conv2dLayerLike};

/// Conv1d, but with a `new` implementation that ensures the weights are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenConv1d {
    conv: Conv1d,
}

impl FrozenConv1d {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: Conv1dConfig,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv1d::new(
                weight.detach()?,
                bias.cloned(), //Bias is still trainable
                config,
            ),
        })
    }

    pub(crate) fn new_from_conv1d(old: &dyn Conv1dLayerLike) -> Result<Self> {
        Self::new(old.weight(), old.bias(), *old.config())
    }

    pub(crate) fn weight(&self) -> &Tensor {
        self.conv.weight()
    }
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.conv.bias()
    }
}

impl Module for FrozenConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

impl Conv1dLayerLike for FrozenConv1d {
    fn config(&self) -> &Conv1dConfig {
        self.conv.config()
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
    conv: Conv2d,
}

impl FrozenConv2d {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(
                weight.detach()?,
                bias.cloned(), //Bias is still trainable
                config,
            ),
        })
    }

    pub(crate) fn new_from_conv2d(old: &dyn Conv2dLayerLike) -> Result<Self> {
        Self::new(old.weight(), old.bias(), *old.config())
    }

    pub(crate) fn weight(&self) -> &Tensor {
        self.conv.weight()
    }
    pub(crate) fn bias(&self) -> Option<&Tensor> {
        self.conv.bias()
    }
}

impl Module for FrozenConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

impl Conv2dLayerLike for FrozenConv2d {
    fn config(&self) -> &Conv2dConfig {
        self.conv.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
}
