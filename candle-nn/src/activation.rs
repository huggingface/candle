use candle::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
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
