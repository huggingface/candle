//! GPU architecture and ROCm toolchain detection.

use crate::error::KernelError;
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Resolve the target architecture for device `ordinal`, e.g. "gfx1101".
///
/// Queried per device rather than once per machine: on a heterogeneous box —
/// an iGPU next to a dGPU, or two different dGPUs — a global answer would build
/// device 1's code objects for device 0's architecture, which only surfaces at
/// the first launch as an opaque "invalid device function". `gcnArchName` also
/// carries the target features (`gfx1101:xnack-`) that `--offload-arch` wants.
///
/// Guessing here is worse than failing, so a device that reports something
/// other than a `gfx*` target is an error rather than a fallback.
/// Memoised per ordinal: `RocmDevice::new` runs on every test and short-lived
/// process, and the answer cannot change while the process lives.
pub(crate) fn gpu_arch(ordinal: usize) -> Result<String, KernelError> {
    if let Ok(arch) = std::env::var("CANDLE_ROCM_ARCH") {
        return Ok(arch);
    }

    static ARCHES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
    let arches = ARCHES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cached) = arches.lock() {
        if let Some(arch) = cached.get(&ordinal) {
            return Ok(arch.clone());
        }
    }

    let arch = gcn_arch_name(ordinal)?;
    if !arch.starts_with("gfx") {
        return Err(KernelError::Compilation(format!(
            "device {ordinal} reports architecture `{arch}`, which is not an AMD GPU target; \
             set CANDLE_ROCM_ARCH to build kernels anyway"
        )));
    }
    if let Ok(mut cached) = arches.lock() {
        cached.insert(ordinal, arch.clone());
    }
    Ok(arch)
}

fn gcn_arch_name(ordinal: usize) -> Result<String, KernelError> {
    use rocm_rs::hip::ffi;

    let device = i32::try_from(ordinal)
        .map_err(|_| KernelError::Rocm(format!("device ordinal {ordinal} is out of range")))?;

    // SAFETY: `hipDeviceProp_tR0600` is a plain-data struct, so an all-zero
    // value is a valid one to hand to the driver. The call only writes into it,
    // and its status is checked before any field is read.
    let (status, props) = unsafe {
        let mut props = std::mem::zeroed::<ffi::hipDeviceProp_tR0600>();
        let status = ffi::hipGetDevicePropertiesR0600(&mut props, device);
        (status, props)
    };

    if status != ffi::hipError_t_hipSuccess {
        return Err(KernelError::Rocm(format!(
            "hipGetDeviceProperties failed for device {ordinal} (HIP error {status}); \
             set CANDLE_ROCM_ARCH to build kernels anyway"
        )));
    }

    // The driver is not required to NUL-terminate a full 256-byte name, so stop
    // at the first NUL rather than handing the array to `CStr`.
    let name: Vec<u8> = props
        .gcnArchName
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    Ok(String::from_utf8_lossy(&name).trim().to_string())
}

/// The ROCm toolchain identity, as `(directory tag, full version string)`.
///
/// The tag stays short so cache directories remain readable. The full
/// `hipcc --version` output is what goes into the cache key: a patch-level LLVM
/// bump inside one minor release can change the code-object ABI, so keying on
/// "7.2" alone would reuse objects the new loader may reject.
///
/// Memoised, because the answer costs a process spawn — around 80 ms of the
/// ~90 ms every `RocmDevice::new` used to take — and cannot change under a
/// running process.
pub(crate) fn rocm_version() -> Result<(String, String), KernelError> {
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok((version.clone(), version));
    }

    static VERSION: OnceLock<Result<(String, String), String>> = OnceLock::new();
    VERSION
        .get_or_init(|| probe_rocm_version().map_err(|e| e.to_string()))
        .clone()
        .map_err(KernelError::Compilation)
}

fn probe_rocm_version() -> Result<(String, String), KernelError> {
    let output = Command::new("hipcc")
        .arg("--version")
        .output()
        .map_err(|e| KernelError::Compilation(format!("could not run hipcc: {e}")))?;

    let full = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let tag = full.lines().find_map(short_version).ok_or_else(|| {
        KernelError::Compilation("could not parse the HIP version from hipcc --version".to_string())
    })?;
    Ok((tag, full))
}

/// "HIP version: 7.2.4-53211" -> "7.2".
fn short_version(line: &str) -> Option<String> {
    let version = line.strip_prefix("HIP version:")?.trim();
    let short = version.split('.').take(2).collect::<Vec<_>>().join(".");
    Some(short.split('-').next().unwrap_or(&short).to_string())
}

#[cfg(test)]
mod tests {
    use super::short_version;

    #[test]
    fn parses_the_hip_version_line() {
        assert_eq!(
            short_version("HIP version: 7.2.4-53211"),
            Some("7.2".to_string())
        );
        assert_eq!(short_version("HIP version: 6.2"), Some("6.2".to_string()));
        assert_eq!(short_version("clang version 20.0.0"), None);
    }
}
