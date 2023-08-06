//! Group Normalization.
//!
//! This layer applies Group Normalization over a mini-batch of inputs.
use candle::{Result, Tensor};

// This group norm version handles both weight and bias so removes the mean.
#[allow(dead_code)]
#[derive(Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    num_channels: usize,
    num_groups: usize,
}

impl GroupNorm {
    pub fn new(
        weight: Tensor,
        bias: Tensor,
        num_channels: usize,
        num_groups: usize,
        eps: f64,
    ) -> Self {
        Self {
            weight,
            bias,
            eps,
            num_channels,
            num_groups,
        }
    }

    pub fn forward(&self, _: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub fn group_norm(
    num_channels: usize,
    num_groups: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get_or_init(num_channels, "weight", crate::Init::Const(1.))?;
    let bias = vb.get_or_init(num_channels, "bias", crate::Init::Const(0.))?;
    Ok(GroupNorm::new(weight, bias, num_channels, num_groups, eps))
}
