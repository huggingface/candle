//! This module inspired from:
//! 
//! https://github.com/huggingface/text-embeddings-inference/blob/cc1c510e8d8af8447c01e6b14c417473cf2dfda9/backends/candle/src/layers/cublaslt.rs

#![allow(unused_variables, unused_imports, dead_code)]

use crate::Activation as CandleActivation;
use candle::{Device, Result, Tensor};
use std::sync::{LazyLock, Mutex, Once};

#[cfg(feature = "cuda")]
mod api;

#[cfg(feature = "cuda")]
use api::{fused_batch_matmul, fused_matmul, Activation, CublasLt};

static INIT: Once = Once::new();
static mut CUBLASLT: Option<CublasLtWrapper> = None;
pub(crate) static CUBLASLT_HANDLE: LazyLock<Mutex<Option<&'static CublasLtWrapper>>> =
    LazyLock::new(|| Mutex::new(None));

/// Internal function to initialize the cublaslt handle wrapper, behind a LazyLock so initialization occurs
/// only once.
pub(crate) fn setup_cublas_lt_wrapper() {
    unsafe {
        INIT.call_once(|| {
            #[cfg(not(feature = "cuda"))]
            {
                CUBLASLT = None;
            }

            #[cfg(feature = "cuda")]
            {
                // Check if we can call the driver
                // Then check if we can create a device
                // Then check that the device is CUDA
                use candle::cuda_backend::cudarc::driver;
                CUBLASLT = driver::result::init()
                    .ok()
                    .and_then(|_| Device::cuda_if_available(0).ok())
                    .and_then(|device| match device {
                        Device::Cuda(_) => Some(CublasLtWrapper {
                            cublaslt: CublasLt::new(&device).unwrap(),
                        }),
                        _ => None,
                    });
            }
        });
        let cublaslt: Option<&'static CublasLtWrapper> = CUBLASLT.as_ref();
        *CUBLASLT_HANDLE.lock().unwrap() = cublaslt;
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<CandleActivation>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = act.map(|a| match a {
                CandleActivation::Relu => Activation::Relu,
                CandleActivation::Gelu => Activation::Gelu,
                _ => unreachable!("Unsupported activation in cublaslt matmul"),
            });
            let mut result = fused_batch_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(CandleActivation::Swiglu) == act {
                result = crate::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    }
}
