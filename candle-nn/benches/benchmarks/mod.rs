pub(crate) mod conv;
pub(crate) mod norm;
pub(crate) mod softmax;

use candle::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                {
                    use candle::backend::BackendDevice;
                    return Ok(device.synchronize()?);
                }
                #[cfg(not(feature = "cuda"))]
                panic!("Cuda device without cuda feature enabled: {device:?}")
            }
            Device::Metal(device) => {
                #[cfg(feature = "metal")]
                return device.wait_until_completed();
                #[cfg(not(feature = "metal"))]
                panic!("Metal device without metal feature enabled: {device:?}")
            }
            #[cfg(feature = "rocm")]
            // `RocmDevice::synchronize` is inherent, so unlike the CUDA arm
            // this needs no `BackendDevice` import.
            Device::Rocm(device) => Ok(device.synchronize()?),
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if cfg!(feature = "accelerate") {
                    "accelerate"
                } else if cfg!(feature = "mkl") {
                    "mkl"
                } else {
                    "cpu"
                };
                format!("{}_{}", cpu_type, name.into())
            }
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => format!("rocm_{}", name.into()),
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        // Exactly one accelerator is benchmarked per build and the features are
        // mutually exclusive by priority, so the choice is a cfg'd binding
        // rather than a sequence of cfg'd pushes — which clippy reads as
        // `vec_init_then_push`.
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        let device = Device::new_cuda(0)?;
        #[cfg(all(feature = "rocm", not(feature = "metal"), not(feature = "cuda")))]
        let device = Device::new_rocm(0)?;
        #[cfg(not(any(feature = "metal", feature = "cuda", feature = "rocm")))]
        let device = Device::Cpu;
        Ok(Self {
            devices: vec![device],
        })
    }
}
