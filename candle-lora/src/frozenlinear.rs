use candle::{Module, Result, Shape, Tensor};

use crate::LinearLayerLike;

/// Linear, but with a `new` implementation that ensures the weight and/or biases are detached (frozen).
#[derive(Debug, Clone)]
pub(crate) struct FrozenLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl FrozenLinear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            weight: weight.detach()?,
            bias: match bias {
                Some(bias) => Some(bias.detach()?),
                None => None,
            },
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
        let w = match *x.dims() {
            [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

impl LinearLayerLike for FrozenLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    fn weight(&self) -> &Tensor {
        &self.weight
    }
    fn shape(&self) -> &Shape {
        self.weight.shape()
    }
}
