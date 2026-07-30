//! Runtime compilation of the shared kernel sources with `hipcc`.
//!
//! A module is compiled once per cache key — source, staged headers, compile
//! flags, GPU architecture, toolchain version and crate version — and cached on
//! disk, so the cost is paid on first use only.

mod cache;
mod detect;

use crate::error::KernelError;
use crate::wrappers::SendSyncModule;
use crate::Module;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Compiled-module cache: in memory for the process, on disk across runs.
pub struct KernelCache {
    cache_dir: PathBuf,
    src_dir: PathBuf,
    arch: String,
    /// Full `hipcc --version` output; part of the cache key.
    toolchain: String,
    /// Keyed by module name, i.e. one entry per translation unit. Keying this
    /// by *kernel* name would recompile the whole unit once per kernel.
    modules: Mutex<HashMap<&'static str, Arc<SendSyncModule>>>,
}

impl KernelCache {
    /// Build a cache for the given device ordinal. The architecture is read
    /// from that specific device, so a second GPU of a different generation
    /// gets its own code objects.
    pub fn new(ordinal: usize) -> Result<Self, KernelError> {
        let arch = detect::gpu_arch(ordinal)?;
        let (version_tag, toolchain) = detect::rocm_version()?;
        let cache_dir =
            cache::base_cache_dir()?.join(format!("{}-{version_tag}", cache::arch_dir_name(&arch)));
        // Staging is content-addressed by the header set: two builds whose
        // headers differ never overwrite each other's staged copies.
        let src_dir = cache_dir.join(format!("src-{}", cache::headers_key(crate::HEADERS)));

        stage_headers(&src_dir)?;

        Ok(Self {
            cache_dir,
            src_dir,
            arch,
            toolchain,
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

        let binary = self.code_object(module)?;
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

    /// Read the module's code object from the disk cache, compiling it first if
    /// it is not there.
    fn code_object(&self, module: &Module) -> Result<Vec<u8>, KernelError> {
        let key = cache::module_key(module.source(), crate::HEADERS, &self.arch, &self.toolchain);
        let cache_file = self
            .cache_dir
            .join(format!("{}_{}.hsaco", module.name(), key));
        let forced = cache::force_recompile();

        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        // Hold the lock across the re-check, the compile and the write: the
        // intermediates below are written non-atomically, so two compilers of
        // the same entry could otherwise persist a corrupt code object.
        let _lock = cache::lock_entry(&self.cache_dir, module.name(), &key)?;
        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        let binary = self.compile(module, &key)?;
        cache::write_atomic(&cache_file, &binary)?;
        Ok(binary)
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
    ///
    /// Both intermediates are named per key *and* per process. The lock in
    /// [`Self::code_object`] is advisory, so nothing outside this crate is
    /// obliged to respect it.
    fn compile(&self, module: &Module, key: &str) -> Result<Vec<u8>, KernelError> {
        let src_file = self.src_dir.join(format!("{}_{key}.cu", module.name()));
        cache::write_atomic(&src_file, module.source().as_bytes())?;

        let shim_dir = self.src_dir.join("hip_shim");
        let bundle = self.cache_dir.join(format!(
            "{}_{key}.{}.bundle",
            module.name(),
            std::process::id()
        ));

        let output = Command::new("hipcc")
            .args(cache::COMPILE_FLAGS)
            .arg(format!("--offload-arch={}", self.arch))
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
}

/// Extract the single-arch code object from a clang offload bundle.
fn unbundle(bundle: &std::path::Path, arch: &str) -> Result<Vec<u8>, KernelError> {
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

/// Write the staged sources the compiler needs to `#include`.
fn stage_headers(src_dir: &std::path::Path) -> Result<(), KernelError> {
    for (rel_path, contents) in crate::HEADERS {
        let path = src_dir.join(rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                KernelError::Io(format!("could not create {}: {e}", parent.display()))
            })?;
        }
        // The directory is named after the digest of this exact header set, so
        // rewriting is a no-op in content; do it anyway rather than trusting a
        // file some earlier crash may have truncated.
        cache::write_atomic(&path, contents.as_bytes())?;
    }
    Ok(())
}
