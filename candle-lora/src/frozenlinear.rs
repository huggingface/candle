use candle::{Module, Result, Shape, Tensor};
use candle_nn::Linear;

use crate::LinearLayerLike;

/// Linear, but with a `new` implementation that ensures the weight and/or biases are detached (frozen).
#[derive(Debug)]
pub(crate) struct FrozenLinear {
    linear: Linear,
}

impl FrozenLinear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(
                weight.detach()?,
                match bias {
                    Some(bias) => Some(bias.detach()?),
                    None => None,
                },
            ),
        })
    }

    pub(crate) fn new_from_linear(old: &dyn LinearLayerLike) -> Result<Self> {
        Self::new(
            old.weight().detach()?,
            match old.bias() {
                Some(bias) => Some(bias.detach()?),
                None => None,
            },
        )
    }
}

impl Module for FrozenLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

impl LinearLayerLike for FrozenLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.linear.bias()
    }
    fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
}
