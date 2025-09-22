pub(crate) mod conv;
pub(crate) mod layer_norm;
pub(crate) mod softmax;

use candle::{BackendDevice, BackendStorage, CpuDevice, CpuStorage};

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
impl BenchDevice<candle::MetalStorage> for candle::MetalDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        format!("metal_{}", name.into())
    }
}

#[cfg(feature = "cuda")]
impl BenchDevice<candle::CudaStorage> for candle::CudaDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        format!("cuda_{}", name.into())
    }
}

#[cfg(feature = "metal")]
fn bench_device() -> candle::MetalDevice {
    candle::MetalDevice::new(0).unwrap()
}

#[cfg(feature = "cuda")]
fn bench_device() -> candle::CudaDevice {
    candle::CudaDevice::new(0).unwrap()
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn bench_device() -> CpuDevice {
    CpuDevice
}
