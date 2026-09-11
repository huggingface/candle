pub(crate) mod affine;
pub(crate) mod binary;
pub(crate) mod broadcast;
pub(crate) mod cat;
pub(crate) mod contiguous;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod vec_dot;
pub(crate) mod where_cond;

use candle_core::{Device, Result};

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
                    use candle_core::backend::BackendDevice;
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
        // `Device::new_metal`/`new_cuda` exist unconditionally and error at
        // runtime, so those two can stay behind `cfg!`; `new_rocm` is compiled
        // only under the feature, so it needs an attribute cfg.
        #[cfg(not(feature = "rocm"))]
        let device = if cfg!(feature = "metal") {
            Device::new_metal(0)?
        } else if cfg!(feature = "cuda") {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        #[cfg(feature = "rocm")]
        let device = Device::new_rocm(0)?;
        Ok(Self {
            devices: vec![device],
        })
    }
}
