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
//! let w = Tensor::new(1f32, &Cpu)?;
//! let b = Tensor::new(0f32, &Cpu)?;
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
use std::{mem, time::{SystemTime, UNIX_EPOCH}};

use candle::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{sys, DeviceRepr, LaunchAsync, LaunchConfig},
        kernel_name, kernels, CudaDType, WrapErr,
    },
    from_storage_no_op, CudaDevice, CudaStorage, DType, Device, Result, Storage, Tensor, WithDType,
    D,
};

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

    #[cfg(feature = "cuda")]
    fn dtype_execute_layernorm<T: CudaDType + DeviceRepr + WithDType, F>(
        &self,
        dev: &CudaDevice,
        elem_count: usize,
        n_rows: usize,
        n_cols: usize,
        max_grid_y: u32,
        eps_converter: F,
        x_storage: &CudaStorage,
        weight_storage: &CudaStorage,
        bias_storage: &CudaStorage,
        x: &Tensor,
    ) -> Result<Tensor>
    where
        F: FnOnce(f64) -> T,
    {
        const BLOCK_DIM_Y: u32 = 4;
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let func =
            dev.get_or_load_func(&kernel_name::<T>("layernorm"), kernels::FUSED_LAYER_NORM)?;
        // 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
        let cfg = LaunchConfig {
            grid_dim: (1, max_grid_y.max(n_rows as u32), max_grid_y),
            block_dim: (32, BLOCK_DIM_Y, 1),
            shared_mem_bytes: 2 * BLOCK_DIM_Y * mem::size_of::<T>() as u32
                + BLOCK_DIM_Y * mem::size_of::<T>() as u32,
        };
        let mean = unsafe { dev.alloc::<T>(n_rows) }.w()?;
        let invvar = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let params = (
            &out,
            &mean,
            &invvar,
            x_storage.as_cuda_slice::<T>()?,
            n_rows,
            n_cols,
            eps_converter(self.eps),
            weight_storage.as_cuda_slice::<T>()?,
            bias_storage.as_cuda_slice::<T>()?,
        );
        unsafe { func.launch(cfg, params) }.w()?;

        Ok(from_storage_no_op(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            x.shape(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn fused_layernorm(&self, x: &Tensor, dev: &CudaDevice) -> Result<Tensor> {
        let elem_count = x.layout().shape().elem_count();
        let dims = x.layout().shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let (n_rows, n_cols) = (elem_count / dim_m1, dim_m1);

        let mut devprop = sys::CUdevprop::default();
        let res = unsafe { sys::cuDeviceGetProperties(&mut devprop as *mut _, *dev.cu_device()) };
        if res != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("{res:?}");
        }
        let max_grid_y: u32 = devprop.maxGridSize[1] as u32;
        dbg!(max_grid_y);
        
        match (
            &*x.storage_and_layout().0,
            &*self.weight().storage_and_layout().0,
            &*self.bias().storage_and_layout().0,
        ) {
            (
                Storage::Cuda(x_storage),
                Storage::Cuda(weight_storage),
                Storage::Cuda(bias_storage),
            ) => {
                match (
                    x_storage.dtype(),
                    weight_storage.dtype(),
                    bias_storage.dtype(),
                ) {
                    (DType::BF16, DType::BF16, DType::BF16) => self
                        .dtype_execute_layernorm::<half::bf16, _>(
                            dev,
                            elem_count,
                            n_rows,
                            n_cols,
                            max_grid_y,
                            |x| half::bf16::from_f64(x),
                            x_storage,
                            weight_storage,
                            &*bias_storage,
                            x,
                        ),
                    (DType::F16, DType::F16, DType::F16) => self
                        .dtype_execute_layernorm::<half::f16, _>(
                            dev,
                            elem_count,
                            n_rows,
                            n_cols,
                            max_grid_y,
                            |x| half::f16::from_f64(x),
                            x_storage,
                            weight_storage,
                            &*bias_storage,
                            x,
                        ),
                    (DType::F32, DType::F32, DType::F32) => self
                        .dtype_execute_layernorm::<f32, _>(
                            dev,
                            elem_count,
                            n_rows,
                            n_cols,
                            max_grid_y,
                            |x| x as f32,
                            x_storage,
                            weight_storage,
                            &*bias_storage,
                            x,
                        ),
                    _ => candle::bail!("DType mismatch in fused layernorm."),
                }
            }
            _ => unreachable!(),
        }
    }
}

impl crate::Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        match (x.dtype(), x.device()) {
            (DType::BF16, Device::Cuda(dev))
            | (DType::F32, Device::Cuda(dev))
            | (DType::F16, Device::Cuda(dev)) => {
                let start = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time travel has occurred!")
                    .as_micros();
                println!("starting...");
                let res = self.fused_layernorm(x, dev);
                let end = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time travel has occurred!")
                    .as_micros();
                println!("{}us", end - start);
                dbg!(&res);
                dbg!(&res?.mean_all()?);
                return res;//self.fused_layernorm(x, dev);
            }
            _ => {}
        };

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

/// RmsNorm is a specialized version of the LayerNorm module.
#[derive(Clone, Debug)]
pub struct RmsNorm(LayerNorm);

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self(LayerNorm::rms_norm(weight, eps))
    }

    pub fn into_inner(self) -> LayerNorm {
        self.0
    }
}

impl crate::Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<RmsNorm> {
    let config = LayerNormConfig {
        eps,
        remove_mean: false,
        affine: false,
    };
    Ok(RmsNorm(layer_norm(size, config, vb)?))
}
