//! Variable initialization.
// This is based on:
// https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/nn/init.py#
use candle::{DType, Device, Result, Shape, Tensor, Var};

/// Number of features as input or output of a layer.
/// In Kaiming initialization, choosing `FanIn` preserves
/// the magnitude of the variance of the weights in the
/// forward pass, choosing `FanOut` preserves this
/// magnitude in the backward pass.
#[derive(Debug, Copy, Clone)]
pub enum FanInOut {
    FanIn,
    FanOut,
}

impl FanInOut {
    /// Compute the fan-in or fan-out value for a weight tensor of
    /// the specified dimensions.
    /// <https://github.com/pytorch/pytorch/blob/dbeacf11820e336e803bb719b7aaaf2125ae4d9c/torch/nn/init.py#L284>
    pub fn for_shape(&self, shape: &Shape) -> usize {
        let dims = shape.dims();
        let receptive_field_size: usize = dims.iter().skip(2).product();
        match &self {
            FanInOut::FanIn => {
                if dims.len() < 2 {
                    1
                } else {
                    dims[1] * receptive_field_size
                }
            }
            FanInOut::FanOut => {
                if dims.is_empty() {
                    1
                } else {
                    dims[0] * receptive_field_size
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NormalOrUniform {
    Normal,
    Uniform,
}

/// The non-linear function that follows this layer. ReLU is the
/// recommended value.
#[derive(Debug, Copy, Clone)]
pub enum NonLinearity {
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain(f64),
}

impl NonLinearity {
    // https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/nn/init.py#L67
    pub fn gain(&self) -> f64 {
        match *self {
            NonLinearity::ReLU => 2f64.sqrt(),
            NonLinearity::Tanh => 5. / 3.,
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.,
            NonLinearity::SELU => 0.75,
            NonLinearity::ExplicitGain(g) => g,
        }
    }
}

/// Variable initializations.
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f64),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f64, stdev: f64 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f64, up: f64 },

    /// Kaiming uniform initialization.
    /// See "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    /// He, K. et al. (2015). This uses a uniform distribution.
    Kaiming {
        dist: NormalOrUniform,
        fan: FanInOut,
        non_linearity: NonLinearity,
    },
}

pub const ZERO: Init = Init::Const(0.);
pub const ONE: Init = Init::Const(1.);

pub const DEFAULT_KAIMING_UNIFORM: Init = Init::Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

pub const DEFAULT_KAIMING_NORMAL: Init = Init::Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

impl Init {
    /// Creates a new tensor with the specified shape, device, and initialization.
    pub fn var<S: Into<Shape>>(&self, s: S, dtype: DType, device: &Device) -> Result<Var> {
        match self {
            Self::Const(v) if *v == 0. => Var::zeros(s, dtype, device),
            Self::Const(v) if *v == 1. => Var::ones(s, dtype, device),
            Self::Const(cst) => {
                Var::from_tensor(&Tensor::ones(s, dtype, device)?.affine(*cst, 0.)?)
            }
            Self::Uniform { lo, up } => Var::rand_f64(*lo, *up, s, dtype, device),
            Self::Randn { mean, stdev } => Var::randn_f64(*mean, *stdev, s, dtype, device),
            Self::Kaiming {
                dist,
                fan,
                non_linearity,
            } => {
                let s = s.into();
                let fan = fan.for_shape(&s);
                let gain = non_linearity.gain();
                let std = gain / (fan as f64).sqrt();
                match dist {
                    NormalOrUniform::Uniform => {
                        let bound = 3f64.sqrt() * std;
                        Var::rand_f64(-bound, bound, s, dtype, device)
                    }
                    NormalOrUniform::Normal => Var::randn_f64(0., std, s, dtype, device),
                }
            }
        }
    }
}

impl Default for Init {
    fn default() -> Self {
        Self::Const(0.)
    }
}
