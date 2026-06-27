//! A single type spanning the GPTQ/AWQ quantized-linear formats, analogous to
//! [`candle_core::quantized::QMatMul`] for the GGUF path: callers pick a [`QuantMethod`] and call
//! [`QuantizedLinear::load`] once, instead of matching on the format themselves and calling one of
//! `gptq_linear`/`awq_linear` or the four `*Cuda`/`*Metal` structs across
//! [`crate::quantized_gptq`] and [`crate::quantized_awq`] directly.
//!
//! [`QuantizedLinear::load`] picks the fused dequantize+GEMM kernel for the checkpoint's format
//! when the matching `{gptq,awq}-{cuda,metal}` feature is enabled and `vb`'s device matches;
//! otherwise it falls back to the portable path, which dequantizes once at load time and runs the
//! regular dense matmul.

use candle::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::quantized_awq::{awq_linear, AwqConfig};
use crate::quantized_gptq::{gptq_linear, GptqConfig};

/// Which quantization format a checkpoint uses, with its format-specific parameters.
#[derive(Debug, Clone, Copy)]
pub enum QuantMethod {
    Gptq(GptqConfig),
    Awq(AwqConfig),
}

/// A quantized linear layer, dispatching to whichever of GPTQ/AWQ the checkpoint uses and to
/// whichever of the portable dense / fused CUDA / fused Metal paths is available, behind a single
/// [`candle_nn::Module`] impl.
#[derive(Debug, Clone)]
pub enum QuantizedLinear {
    /// Portable path: the checkpoint was dequantized into a dense weight once at load time, for
    /// either format.
    Dense(Linear),
    #[cfg(feature = "gptq-cuda")]
    GptqCuda(crate::quantized_gptq::cuda::GptqLinearCuda),
    #[cfg(feature = "gptq-metal")]
    GptqMetal(crate::quantized_gptq::metal::GptqLinearMetal),
    #[cfg(feature = "awq-cuda")]
    AwqCuda(crate::quantized_awq::cuda::AwqLinearCuda),
    #[cfg(feature = "awq-metal")]
    AwqMetal(crate::quantized_awq::metal::AwqLinearMetal),
}

impl QuantizedLinear {
    /// Load a quantized linear layer at the current `VarBuilder` path.
    ///
    /// Uses the fused dequantize+GEMM kernel for `method`'s format when the corresponding
    /// `{gptq,awq}-{cuda,metal}` feature is compiled in and `vb.device()` matches; otherwise
    /// dequantizes once at load time and falls back to a dense [`candle_nn::Linear`].
    pub fn load(
        in_dim: usize,
        out_dim: usize,
        method: QuantMethod,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        match method {
            QuantMethod::Gptq(cfg) => {
                #[cfg(feature = "gptq-cuda")]
                if vb.device().is_cuda() {
                    return Ok(Self::GptqCuda(
                        crate::quantized_gptq::cuda::GptqLinearCuda::new(
                            in_dim, out_dim, cfg, bias, vb,
                        )?,
                    ));
                }
                #[cfg(feature = "gptq-metal")]
                if vb.device().is_metal() {
                    return Ok(Self::GptqMetal(
                        crate::quantized_gptq::metal::GptqLinearMetal::new(
                            in_dim, out_dim, cfg, bias, vb,
                        )?,
                    ));
                }
                Ok(Self::Dense(gptq_linear(in_dim, out_dim, cfg, bias, vb)?))
            }
            QuantMethod::Awq(cfg) => {
                #[cfg(feature = "awq-cuda")]
                if vb.device().is_cuda() {
                    return Ok(Self::AwqCuda(
                        crate::quantized_awq::cuda::AwqLinearCuda::new(
                            in_dim, out_dim, cfg, bias, vb,
                        )?,
                    ));
                }
                #[cfg(feature = "awq-metal")]
                if vb.device().is_metal() {
                    return Ok(Self::AwqMetal(
                        crate::quantized_awq::metal::AwqLinearMetal::new(
                            in_dim, out_dim, cfg, bias, vb,
                        )?,
                    ));
                }
                Ok(Self::Dense(awq_linear(in_dim, out_dim, cfg, bias, vb)?))
            }
        }
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(xs),
            #[cfg(feature = "gptq-cuda")]
            Self::GptqCuda(l) => l.forward(xs),
            #[cfg(feature = "gptq-metal")]
            Self::GptqMetal(l) => l.forward(xs),
            #[cfg(feature = "awq-cuda")]
            Self::AwqCuda(l) => l.forward(xs),
            #[cfg(feature = "awq-metal")]
            Self::AwqMetal(l) => l.forward(xs),
        }
    }
}
