//! Device management utilities.

use candle::{Device, Result};

/// Helper function to get a single candle Device. This will not perform any multi device mapping.
/// Will prioritize Metal or CUDA (if Candle is compiled with those features) and fallback to CPU.
///
/// # Arguments
/// * `use_cpu` - Force CPU usage even if GPU is available
/// * `quiet` - Suppress informational messages about GPU availability
///
/// # Example  
/// ```
/// let device = candle_utils::device::get(true, false).unwrap();
/// ```
pub fn get_device(use_cpu: bool, quiet: bool) -> Result<Device> {
    if use_cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        if !quiet {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!("Running on CPU, to run on GPU (metal), build with `--features metal`");
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!("Running on CPU, to run on GPU, build with `--features cuda`");
            }
        }

        Ok(Device::Cpu)
    }
}
