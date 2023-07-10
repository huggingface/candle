use candle::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
        }
    }
}

#[derive(Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(&self.weight, self.config.padding, self.config.stride)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.shape().r1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}
