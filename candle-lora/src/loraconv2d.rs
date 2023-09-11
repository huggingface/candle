use std::ops::Mul;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{init, Conv2d, Conv2dConfig, Dropout, VarMap};

use crate::{frozenconv::FrozenConv2d, Conv2dLayerLike};

#[derive(Debug)]
pub struct LoraConv2d {
    old: FrozenConv2d,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Dropout>,
}

/// Configuration for LoraConv2d. Other configurations are inherited from the `Conv2d` struct.
pub struct LoraConv2dConfig<'a> {
    rank: usize,
    alpha: f64,
    device: &'a Device,
    dtype: DType,
    in_channels: usize,
    out_channels: usize,
    dropout: Option<f32>,
}

/// Builder for LoraConv2dConfig. Call `build` to construct the config.
pub struct LoraConv2dConfigBuilder<'a> {
    pub config: LoraConv2dConfig<'a>,
}

impl<'a> LoraConv2dConfigBuilder<'a> {
    pub fn default(
        device: &'a Device,
        dtype: DType,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        LoraConv2dConfigBuilder {
            config: LoraConv2dConfig {
                rank: 1,
                alpha: 1.,
                device,
                dtype,
                in_channels,
                out_channels,
                dropout: None,
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

    /// Set the dropout
    pub fn dropout(mut self, prob: f32) -> Self {
        self.config.dropout = Some(prob);
        self
    }

    /// Construct the config
    pub fn build(self) -> LoraConv2dConfig<'a> {
        self.config
    }
}

impl LoraConv2d {
    pub fn new(old: &dyn Conv2dLayerLike, config: &LoraConv2dConfig) -> Result<Self> {
        let map = VarMap::new();
        let a = map.get(
            (
                config.rank,
                config.in_channels / old.config().groups,
                old.weight().dim(2).unwrap(),
                old.weight().dim(3).unwrap(),
            ),
            "a.weight",
            init::DEFAULT_KAIMING_NORMAL,
            config.dtype,
            config.device,
        )?;
        let b = map.get(
            (config.out_channels, config.rank / old.config().groups, 1, 1),
            "b.weight",
            init::ZERO,
            config.dtype,
            config.device,
        )?;

        Ok(LoraConv2d {
            old: FrozenConv2d::new_from_conv2d(old)?,
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

impl Module for LoraConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(scale) = self.scale {
            let weight = self.old.forward(input)?;
            let mut a_input = input.clone();
            if self.dropout.is_some() {
                a_input = self.dropout.as_ref().unwrap().forward(input, true)?;
            }

            let a_conv = Conv2d::new(self.a.clone(), None, *self.config());
            let b_conv = Conv2d::new(
                self.b.clone(),
                None,
                Conv2dConfig {
                    stride: 1,
                    ..*self.config()
                },
            );

            let tmp = b_conv.forward(&a_conv.forward(&a_input)?)?;

            &weight + tmp.mul(scale)?
        } else {
            self.old.forward(input)
        }
    }
}

impl Conv2dLayerLike for LoraConv2d {
    fn config(&self) -> &Conv2dConfig {
        self.old.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
}
