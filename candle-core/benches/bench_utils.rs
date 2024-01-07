use candle_core::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                return Ok(device.synchronize()?);
                #[cfg(not(feature = "cuda"))]
                panic!("Cuda device without cuda feature enabled: {:?}", device)
            }
            Device::Metal(device) => {
                #[cfg(feature = "metal")]
                return Ok(device.wait_until_completed()?);
                #[cfg(not(feature = "metal"))]
                panic!("Metal device without metal feature enabled: {:?}", device)
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn device() -> Result<Device> {
    return if cfg!(feature = "metal") {
        Device::new_metal(0)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0)
    } else {
        Ok(Device::Cpu)
    };
}

#[allow(dead_code)]
pub(crate) fn bench_name<S: Into<String>>(name: S) -> String {
    format!("{}_{}", device_variant(), name.into())
}

#[allow(dead_code)]
const fn device_variant() -> &'static str {
    return if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "accelerate") {
        "accelerate"
    } else if cfg!(feature = "mkl") {
        "mkl"
    } else {
        "cpu"
    };
}
