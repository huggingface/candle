use candle::Tensor;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Gelu,
    Relu,
    Elu(f64),
}

impl super::Module for Activation {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
            &Self::Elu(alpha) => xs.elu(alpha),
        }
    }
}
