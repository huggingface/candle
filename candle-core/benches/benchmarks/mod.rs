pub(crate) mod affine;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod where_cond;

use candle_core::{BackendDevice, BackendStorage, CpuDevice, CpuStorage};
#[cfg(feature = "metal")]
use candle_core::{MetalDevice, MetalStorage};

pub(crate) trait BenchDevice<B>: BackendDevice<B>
where
    B: BackendStorage,
{
    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice<CpuStorage> for CpuDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        let cpu_type = if cfg!(feature = "accelerate") {
            "accelerate"
        } else if cfg!(feature = "mkl") {
            "mkl"
        } else {
            "cpu"
        };
        format!("{}_{}", cpu_type, name.into())
    }
}

#[cfg(feature = "metal")]
impl BenchDevice<MetalStorage> for MetalDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        format!("metal_{}", name.into())
    }
}

impl BenchDevice<candle_core::CudaStorage> for candle_core::CudaDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        format!("cuda_{}", name.into())
    }
}

#[cfg(feature = "metal")]
fn bench_device() -> MetalDevice {
    MetalDevice::new(0).unwrap()
}

#[cfg(feature = "cuda")]
fn bench_device() -> candle_core::CudaDevice {
    candle_core::CudaDevice::new(0).unwrap()
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn bench_device() -> CpuDevice {
    CpuDevice
}
