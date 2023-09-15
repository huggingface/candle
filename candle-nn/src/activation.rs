use candle::Tensor;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Gelu,
    #[serde(rename = "gated-gelu")]
    NewGelu,
    Relu,
    Elu(f64),
    LeakyRelu(f64),
}

impl super::Module for Activation {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            // TODO: This is "gelu_new", not the original "gelu".
            // There's some small numerical difference:
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => crate::ops::leaky_relu(xs, negative_slope),
        }
    }
}
