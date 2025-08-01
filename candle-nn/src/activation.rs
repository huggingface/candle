//! Activation Functions
//!
use candle::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,

    // New GLU variants
    /// Gated Linear Unit - splits input in half, applies sigmoid to one half,
    /// multiplies with the other half. Commonly used in transformer FFNs.
    #[serde(alias = "glu")]
    Glu,
    /// GeGLU - GLU variant using GELU instead of sigmoid
    #[serde(alias = "geglu")]
    GeGlu,
    /// ReGLU - GLU variant using ReLU instead of sigmoid  
    #[serde(alias = "reglu")]
    ReGlu,
}

impl super::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => crate::ops::sigmoid(xs),
            Self::HardSigmoid => crate::ops::hard_sigmoid(xs),
            Self::Swiglu => crate::ops::swiglu(xs),
            Self::Swish => xs * crate::ops::sigmoid(xs)?,
            Self::HardSwish => xs * crate::ops::hard_sigmoid(xs)?,
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => crate::ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
            Self::Glu => xs.glu(),
            Self::GeGlu => xs.geglu(),
            Self::ReGlu => xs.reglu(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PReLU {
    weight: Tensor,
    is_scalar: bool,
}

impl PReLU {
    pub fn new(weight: Tensor, is_scalar: bool) -> Self {
        Self { weight, is_scalar }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}

impl candle::Module for PReLU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = if self.is_scalar {
            self.weight.reshape(())?
        } else if xs.shape() == self.weight.shape() {
            self.weight.clone()
        } else if xs.rank() >= 2 {
            let num_channels = xs.dim(1)?;
            let num_weights = self.weight.elem_count();
            if num_weights != num_channels {
                candle::bail!("error in prelu: unexpected number of channels for the input, got {num_channels}, weight dim is {num_weights}")
            }
            let mut s = vec![1; xs.rank()];
            s[1] = num_weights;
            self.weight.reshape(s)?
        } else {
            self.weight.clone()
        };
        let zeros = xs.zeros_like()?;
        xs.maximum(&zeros)? + xs.minimum(&zeros)?.broadcast_mul(&weight)?
    }
}

/// Create or initialize a new PReLU layer.
///
/// This uses some default name for weights, namely `"weight"`.
/// # Arguments
///
/// * `num_channels` - The number of channels. Use `None` to have as single trainable value and
///   `Some` for a 1D vector with the appropriate number of channels. When applying the `forward`
///   function, the input tensor shape `s` should either be one dimension with this number of
///   channels or if `s.len() >= 2` it should have `s[1]` equal to this number.
pub fn prelu(num_channels: Option<usize>, vs: crate::VarBuilder) -> Result<PReLU> {
    let init_ws = crate::init::Init::Const(0.25);
    // When using a scalar weight, the PyTorch encoding is to use a 1d vector of length 1.
    let ws = vs.get_with_hints((num_channels.unwrap_or(1),), "weight", init_ws)?;
    Ok(PReLU::new(ws, num_channels.is_none()))
}
