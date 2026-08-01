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

/// Which set of MMQ tile constants `quantized.cu` is compiled with.
///
/// The kernel carries one geometry per architecture and picks between them on
/// the `RDNA2`/`RDNA3` defines this module supplies. The host has to launch the
/// *same* geometry — the grid and block are computed from `mmq_x`, `mmq_y` and
/// `nwarps` — so the choice cannot stay inside the compiler flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmqTiles {
    /// `MMQ_*_AMPERE`, `nwarps = 4`. Every non-RDNA target.
    Ampere,
    /// `MMQ_*_RDNA2`, `nwarps = 8`. RDNA2 and later.
    Rdna2,
}

/// The `RDNA*` define that selects `arch`'s MMQ geometry, or `None` where the
/// kernel's default (Ampere) set stands.
///
/// gfx12xx (RDNA4) takes the RDNA3 define: `quantized.cu` maps RDNA3 onto the
/// RDNA2 tiles, which is the geometry RDNA4 wants as well, and adding a fourth
/// spelling would select nothing new. gfx101x (RDNA1) is deliberately absent —
/// its tile set is a third variant, and no candle CI or developer machine has
/// the hardware to check it.
pub(crate) fn rdna_define(arch: &str) -> Option<&'static str> {
    match arch_number(arch)? {
        1030..=1039 => Some("RDNA2"),
        1100..=1299 => Some("RDNA3"),
        _ => None,
    }
}

/// [`MmqTiles`] for `arch`, in lockstep with [`rdna_define`].
pub(crate) fn mmq_tiles(arch: &str) -> MmqTiles {
    match rdna_define(arch) {
        Some(_) => MmqTiles::Rdna2,
        None => MmqTiles::Ampere,
    }
}

/// The numeric part of a `gfx` target, e.g. `gfx1101:xnack-` -> 1101.
///
/// `None` for anything that is not purely numeric: `gfx90a` must not read as
/// "90" and land in a range it was never meant to match.
fn arch_number(arch: &str) -> Option<u32> {
    arch.split(':').next()?.strip_prefix("gfx")?.parse().ok()
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
    use super::{mmq_tiles, rdna_define, short_version, MmqTiles};

    #[test]
    fn selects_the_rdna_mmq_geometry_per_architecture() {
        assert_eq!(rdna_define("gfx1101"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1101:xnack-"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1151"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1201"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1030"), Some("RDNA2"));
        // RDNA1 and the CDNA/Vega parts keep the kernel's default set. `gfx90a`
        // must not be read as architecture 90.
        assert_eq!(rdna_define("gfx1010"), None);
        assert_eq!(rdna_define("gfx90a:sramecc+:xnack-"), None);
        assert_eq!(rdna_define("gfx942"), None);
        assert_eq!(rdna_define("not-a-gfx-target"), None);
    }

    #[test]
    fn the_tile_set_follows_the_define() {
        assert_eq!(mmq_tiles("gfx1101"), MmqTiles::Rdna2);
        assert_eq!(mmq_tiles("gfx1030"), MmqTiles::Rdna2);
        assert_eq!(mmq_tiles("gfx90a"), MmqTiles::Ampere);
    }

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
