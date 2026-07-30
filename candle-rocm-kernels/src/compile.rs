//! Runtime compilation of the shared kernel sources with `hipcc`.
//!
//! A module is compiled once per (source, GPU architecture, ROCm version) and
//! cached on disk, so the cost is paid on first use only.

use crate::error::KernelError;
use crate::wrappers::SendSyncModule;
use crate::Module;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Compiled-module cache: in memory for the process, on disk across runs.
pub struct KernelCache {
    cache_dir: PathBuf,
    src_dir: PathBuf,
    arch: String,
    rocm_version: String,
    /// Keyed by module name, i.e. one entry per translation unit. Keying this
    /// by *kernel* name would recompile the whole unit once per kernel.
    modules: Mutex<HashMap<&'static str, Arc<SendSyncModule>>>,
}

impl KernelCache {
    pub fn new(arch: Option<String>) -> Result<Self, KernelError> {
        let arch = match arch {
            Some(arch) => arch,
            None => detect_gpu_arch()?,
        };
        let rocm_version = detect_rocm_version()?;
        let cache_dir = base_cache_dir()?.join(format!("{arch}-{rocm_version}"));
        let src_dir = cache_dir.join("src");

        stage_headers(&src_dir)?;

        Ok(Self {
            cache_dir,
            src_dir,
            arch,
            rocm_version,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// Return the loaded module, compiling it if this is the first use.
    pub fn get_or_load(&self, module: &Module) -> Result<Arc<SendSyncModule>, KernelError> {
        {
            let modules = self.lock_modules()?;
            if let Some(loaded) = modules.get(module.name()) {
                return Ok(loaded.clone());
            }
        }

        let hash = source_hash(module.source());
        let cache_file = self
            .cache_dir
            .join(format!("{}_{}.hsaco", module.name(), hash));

        let binary = match fs::read(&cache_file) {
            Ok(binary) => binary,
            Err(_) => {
                let binary = self.compile(module, &hash)?;
                write_atomic(&cache_file, &binary)?;
                binary
            }
        };

        let loaded = Arc::new(SendSyncModule::load_data(&binary).map_err(|e| {
            KernelError::Compilation(format!(
                "failed to load module `{}` for {}: {e}",
                module.name(),
                self.arch
            ))
        })?);

        self.lock_modules()?.insert(module.name(), loaded.clone());
        Ok(loaded)
    }

    fn lock_modules(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<&'static str, Arc<SendSyncModule>>>, KernelError>
    {
        self.modules
            .lock()
            .map_err(|_| KernelError::Internal("kernel module cache mutex is poisoned".to_string()))
    }

    /// hipcc the source into a bundled code object, then unbundle it into the
    /// single-architecture ELF that `hipModuleLoadData` expects.
    fn compile(&self, module: &Module, hash: &str) -> Result<Vec<u8>, KernelError> {
        let src_file = self.src_dir.join(format!("{}.cu", module.name()));
        write_atomic(&src_file, module.source().as_bytes())?;

        let shim_dir = self.src_dir.join("hip_shim");
        let bundle = self
            .cache_dir
            .join(format!("{}_{}.bundle", module.name(), hash));

        let output = Command::new("hipcc")
            .arg("--genco")
            .arg(format!("--offload-arch={}", self.arch))
            .args(["-O3", "-std=c++17"])
            // The shared sources gate f16 on >= 530 and bf16 on >= 800. fp8
            // shares the 800 guard, and maps onto HIP's OCP fp8 type.
            .arg("-D__CUDA_ARCH__=800")
            .arg("-include")
            .arg(shim_dir.join("hip_compat.h"))
            // Shim first: its cuda_*.h shadow the CUDA toolkit headers.
            .arg("-I")
            .arg(&shim_dir)
            .arg("-I")
            .arg(&self.src_dir)
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
                module.name(),
                self.arch,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let binary = unbundle(&bundle, &self.arch);
        let _ = fs::remove_file(&bundle);
        binary
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn rocm_version(&self) -> &str {
        &self.rocm_version
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

/// Extract the single-arch code object from a clang offload bundle.
fn unbundle(bundle: &Path, arch: &str) -> Result<Vec<u8>, KernelError> {
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

/// Write the staged sources the compiler needs to `#include`.
fn stage_headers(src_dir: &Path) -> Result<(), KernelError> {
    for (rel_path, contents) in crate::HEADERS {
        let path = src_dir.join(rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                KernelError::Io(format!("could not create {}: {e}", parent.display()))
            })?;
        }
        // Staged files are content-addressed by the enclosing directory's arch
        // and ROCm version, but the sources themselves change with the crate,
        // so always rewrite rather than trusting an existing file.
        write_atomic(&path, contents.as_bytes())?;
    }
    Ok(())
}

/// Write via a unique temporary in the destination directory, then rename.
/// Concurrent compilers (parallel test threads, several processes) would
/// otherwise observe a half-written file.
fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), KernelError> {
    let parent = path
        .parent()
        .ok_or_else(|| KernelError::Io(format!("{} has no parent directory", path.display())))?;
    fs::create_dir_all(parent)
        .map_err(|e| KernelError::Io(format!("could not create {}: {e}", parent.display())))?;

    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("staged"),
        std::process::id()
    ));
    fs::write(&tmp, contents)
        .map_err(|e| KernelError::Io(format!("could not write {}: {e}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        KernelError::Io(format!("could not place {}: {e}", path.display()))
    })
}

/// Resolve the target architecture, e.g. "gfx1101".
///
/// Guessing here is worse than failing: a code object built for the wrong
/// architecture loads with an opaque "invalid device function" much later.
fn detect_gpu_arch() -> Result<String, KernelError> {
    if let Ok(arch) = std::env::var("CANDLE_ROCM_ARCH") {
        return Ok(arch);
    }

    let output = Command::new("rocm_agent_enumerator")
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!(
                "could not run rocm_agent_enumerator: {e}. Install ROCm or set CANDLE_ROCM_ARCH"
            ))
        })?;

    // The host agent reports itself as gfx000.
    let arch = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("gfx") && *line != "gfx000")
        .map(str::to_string);

    arch.ok_or_else(|| {
        KernelError::Compilation(
            "no AMD GPU architecture detected; set CANDLE_ROCM_ARCH to build kernels anyway"
                .to_string(),
        )
    })
}

fn detect_rocm_version() -> Result<String, KernelError> {
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok(version);
    }

    let output = Command::new("hipcc")
        .arg("--version")
        .output()
        .map_err(|e| KernelError::Compilation(format!("could not run hipcc: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version = stdout
        .lines()
        .find_map(|line| line.strip_prefix("HIP version:"))
        .map(|v| {
            v.trim()
                .split('.')
                .take(2)
                .collect::<Vec<_>>()
                .join(".")
                // Drop any build suffix, e.g. "7.2-53211".
                .split('-')
                .next()
                .unwrap_or("unknown")
                .to_string()
        });

    version.ok_or_else(|| {
        KernelError::Compilation("could not parse the HIP version from hipcc --version".to_string())
    })
}

fn base_cache_dir() -> Result<PathBuf, KernelError> {
    dirs::cache_dir()
        .map(|dir| dir.join("candle-rocm"))
        .ok_or_else(|| KernelError::Internal("could not determine a cache directory".to_string()))
}

fn source_hash(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}
