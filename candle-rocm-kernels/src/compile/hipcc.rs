//! Driving `hipcc` and its bundler, and staging the sources they read.

use crate::error::KernelError;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use super::cache;

/// hipcc the source into a bundled code object, then unbundle it into the
/// single-architecture ELF that `hipModuleLoadData` expects.
///
/// `entry` is the cache entry's file stem, i.e. `<name>_<key>`; `name` is the
/// module name and is only there for the error message. Both intermediates are
/// named per entry *and* per process. The lock in
/// [`super::KernelCache::code_object`] is advisory, so nothing outside this
/// crate is obliged to respect it.
pub(crate) fn compile(
    src_dir: &Path,
    cache_dir: &Path,
    arch: &str,
    name: &str,
    entry: &str,
    source: &str,
) -> Result<Vec<u8>, KernelError> {
    let src_file = src_dir.join(format!("{entry}.cu"));
    cache::write_atomic(&src_file, source.as_bytes())?;

    let shim_dir = src_dir.join("hip_shim");
    let bundle = cache_dir.join(format!("{entry}.{}.bundle", std::process::id()));

    let output = Command::new("hipcc")
        .args(cache::COMPILE_FLAGS)
        .arg(format!("--offload-arch={arch}"))
        .arg("-include")
        .arg(shim_dir.join("hip_compat.h"))
        // Shim first: its cuda_*.h shadow the CUDA toolkit headers.
        .arg("-I")
        .arg(&shim_dir)
        .arg("-I")
        .arg(src_dir)
        .arg("-o")
        .arg(&bundle)
        .arg(&src_file)
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!("could not run hipcc: {e}. Is ROCm installed?"))
        })?;

    if !output.status.success() {
        return Err(KernelError::Compilation(format!(
            "hipcc failed for `{}` ({}):\n{}",
            name,
            arch,
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let binary = unbundle(&bundle, arch);
    let _ = fs::remove_file(&bundle);
    binary
}

/// Extract the single-arch code object from a clang offload bundle.
fn unbundle(bundle: &Path, arch: &str) -> Result<Vec<u8>, KernelError> {
    // `bundle` is already process-unique, so this derived name is too.
    let unbundled = bundle.with_extension("hsaco.tmp");
    let output = Command::new(offload_bundler())
        .arg("--unbundle")
        .arg("--type=o")
        .arg(format!("--targets=hipv4-amdgcn-amd-amdhsa--{arch}"))
        .arg(format!("--input={}", bundle.display()))
        .arg(format!("--output={}", unbundled.display()))
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!("could not run clang-offload-bundler: {e}"))
        })?;

    if !output.status.success() {
        return Err(KernelError::Compilation(format!(
            "clang-offload-bundler failed for {arch}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let binary = fs::read(&unbundled)
        .map_err(|e| KernelError::Io(format!("could not read unbundled code object: {e}")));
    let _ = fs::remove_file(&unbundled);
    binary
}

/// Prefer ROCm's own bundler: a system clang's copy can be a different LLVM
/// version than the hipcc that produced the bundle.
fn offload_bundler() -> PathBuf {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    for candidate in [
        PathBuf::from(&rocm).join("llvm/bin/clang-offload-bundler"),
        PathBuf::from(&rocm).join("bin/clang-offload-bundler"),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("clang-offload-bundler")
}

/// Directories this process has already staged, so a second `RocmDevice::new`
/// does not re-walk the header set.
fn staged_dirs() -> &'static Mutex<HashSet<PathBuf>> {
    static STAGED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    STAGED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Write the staged sources the compiler needs to `#include`.
///
/// `src_dir` is named after the digest of this exact header set, so a directory
/// carrying the completion marker already holds the right contents. The marker
/// is written last and atomically: a run that crashed mid-staging leaves no
/// marker, and the next one rewrites everything rather than trusting a file it
/// may have truncated.
pub(crate) fn stage_headers(src_dir: &Path) -> Result<(), KernelError> {
    let staged = staged_dirs();
    if let Ok(seen) = staged.lock() {
        if seen.contains(src_dir) {
            return Ok(());
        }
    }

    let marker = src_dir.join(".staged");
    if !marker.is_file() {
        for (rel_path, contents) in crate::HEADERS {
            let path = src_dir.join(rel_path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    KernelError::Io(format!("could not create {}: {e}", parent.display()))
                })?;
            }
            cache::write_atomic(&path, contents.as_bytes())?;
        }
        cache::write_atomic(&marker, b"")?;
    }

    if let Ok(mut seen) = staged.lock() {
        seen.insert(src_dir.to_path_buf());
    }
    Ok(())
}
