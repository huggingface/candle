//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu, test_utils::to_vec3_round};
//! use candle_nn::{LayerNorm, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &Cpu)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &Cpu)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(
//!     to_vec3_round(&ys, 4)?,
//!     &[[[-1.2247, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]]);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450

use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use candle::cuda_backend::{
    cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig},
    kernel_name, kernels, CudaDType, WrapErr,
};

#[cfg(feature = "cuda")]
use candle::{
    backend::BackendStorage, from_storage_no_op, CudaDevice, CudaStorage, Device, Storage,
    WithDType,
};

use candle::{DType, Module, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f64,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    pub remove_mean: bool,
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias,
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight: weight.clone(),
            bias: Tensor::zeros_like(&weight).unwrap(),
            remove_mean: true,
            eps,
        }
    }

    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self {
            weight: weight.clone(),
            bias: Tensor::zeros_like(&weight).unwrap(),
            remove_mean: false,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && self.remove_mean {
            return crate::ops::layer_norm(x, &self.weight, &self.bias, self.eps as f32);
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::VarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get_with_hints(size, "weight", crate::Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", crate::Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm {
        weight: weight.clone(),
        bias: bias.unwrap_or(Tensor::zeros_like(&weight)?),
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

// This whole non quantized/quantized RmsNorm is a hack. It seems like quantized works without this impl, but it is slower.
#[derive(Clone, Debug)]
pub struct RmsNormQuantized;
#[derive(Clone, Debug)]
pub struct RmsNormNonQuantized;

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm<T> {
    inner: LayerNorm,
    _ghost: PhantomData<T>,
}

impl RmsNorm<RmsNormNonQuantized> {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            inner: LayerNorm::rms_norm(weight, eps),
            _ghost: PhantomData,
        }
    }
}

impl RmsNorm<RmsNormQuantized> {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            inner: LayerNorm::rms_norm(weight, eps),
            _ghost: PhantomData,
        }
    }

    #[cfg(feature = "cuda")]
    fn dtype_execute_rmsnorm<T: CudaDType + DeviceRepr + WithDType, F>(
        &self,
        dev: &CudaDevice,
        eps_converter: F,
        x_storage: &CudaStorage,
        weight_storage: &CudaStorage,
        x: &Tensor,
    ) -> Result<Tensor>
    where
        F: FnOnce(f64) -> T,
    {
        assert!(x.layout().is_contiguous());
        let hidden_size = *x.dims().last().unwrap();
        let elem_count = x.elem_count();
        let num_tokens = elem_count / hidden_size;
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (u32::min(hidden_size as u32, 1024), 1, 1),
            shared_mem_bytes: 0,
        };

        let func = dev.get_or_load_func(&kernel_name::<T>("rms_norm"), kernels::FUSED_RMS_NORM)?;

        let params = (
            &out,
            x_storage.as_cuda_slice::<T>()?,
            weight_storage.as_cuda_slice::<T>()?,
            eps_converter(self.inner.eps),
            num_tokens as i32,
            hidden_size as i32,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(from_storage_no_op(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            x.shape(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn fused_rmsnorm(&self, x: &Tensor, dev: &CudaDevice) -> Result<Tensor> {
        match (
            &*x.storage_and_layout().0,
            &*self.inner.weight().storage_and_layout().0,
        ) {
            (Storage::Cuda(x_storage), Storage::Cuda(weight_storage)) => {
                match (x_storage.dtype(), weight_storage.dtype()) {
                    (DType::BF16, DType::BF16) => self.dtype_execute_rmsnorm::<half::bf16, _>(
                        dev,
                        |x| half::bf16::from_f64(x),
                        &x_storage,
                        &weight_storage,
                        x,
                    ),
                    (DType::F16, DType::F16) => self.dtype_execute_rmsnorm::<half::f16, _>(
                        dev,
                        |x| half::f16::from_f64(x),
                        &x_storage,
                        &weight_storage,
                        x,
                    ),
                    (DType::F32, DType::F32) => self.dtype_execute_rmsnorm::<f32, _>(
                        dev,
                        |x| x as f32,
                        &x_storage,
                        &weight_storage,
                        x,
                    ),
                    _ => candle::bail!("DType mismatch in fused rmsnorm."),
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<T> RmsNorm<T> {
    pub fn into_inner(self) -> LayerNorm {
        self.inner
    }
    pub fn inner(&self) -> &LayerNorm {
        &self.inner
    }
}

impl Module for RmsNorm<RmsNormNonQuantized> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

impl Module for RmsNorm<RmsNormQuantized> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        match (xs.dtype(), xs.device()) {
            (DType::BF16, Device::Cuda(dev))
            | (DType::F32, Device::Cuda(dev))
            | (DType::F16, Device::Cuda(dev)) => return self.fused_rmsnorm(xs, &dev),
            _ => return self.inner.forward(xs),
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.inner.forward(xs)
        }
    }
}

pub fn rms_norm_non_quant(
    size: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<RmsNorm<RmsNormNonQuantized>> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm {
        inner: layer_norm(size, config, vb)?,
        _ghost: PhantomData,
    })
}

pub fn rms_norm_quant(
    size: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<RmsNorm<RmsNormQuantized>> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm {
        inner: layer_norm(size, config, vb)?,
        _ghost: PhantomData,
    })
}
