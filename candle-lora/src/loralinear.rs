use std::ops::Mul;

use candle::{DType, Device, Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, VarMap};

use crate::{frozenlinear::FrozenLinear, LinearLayerLike};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
}

/// Configuration for LoraLinear
pub struct LoraLinearConfig<'a> {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: Option<f32>,
    pub device: &'a Device,
    pub dtype: DType,
    pub in_features: usize,
    pub out_features: usize,
}

/// Builder for LoraLinearConfig. Call `build` to construct the config.
pub struct LoraLinearConfigBuilder<'a> {
    pub config: LoraLinearConfig<'a>,
}

impl<'a> LoraLinearConfigBuilder<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        LoraLinearConfigBuilder {
            config: LoraLinearConfig {
                rank: 1,
                alpha: 1.,
                dropout: Some(0.),
                device,
                dtype,
                in_features,
                out_features,
            },
        }
    }

    /// Set the rank parameter
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Construct the config
    pub fn build(self) -> LoraLinearConfig<'a> {
        self.config
    }
}

impl LoraLinear {
    pub fn new(old: &dyn LinearLayerLike, config: &LoraLinearConfig) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (config.rank, config.in_features),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (config.out_features, config.rank),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraLinear {
            old: FrozenLinear::new_from_linear(old)?,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(Dropout::new),
        })
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        //No fan_in_fan_out so no weight.transpose(0,1)
        let mut result = self.old.forward(input)?;
        if let Some(scale) = self.scale {
            if self.dropout.is_some() {
                result = (result + self.dropout.as_ref().unwrap().forward(input, true)?)?;
            } else {
                result = (result + input)?;
            }
            result = result.broadcast_add(
                &result.matmul(&self.b.broadcast_matmul(&self.a.matmul(&result)?)?)?,
            )?;
            result = result.broadcast_add(&result.clone().mul(scale)?)?;
        }
        Ok(result)
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
}
