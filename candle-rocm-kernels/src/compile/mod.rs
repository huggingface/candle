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
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// Compiled-module cache: in memory for the process, on disk across runs.
pub struct KernelCache {
    cache_dir: PathBuf,
    src_dir: PathBuf,
    arch: String,
    /// Full `hipcc --version` output; part of the cache key.
    toolchain: String,
    /// Keyed by module name, i.e. one entry per translation unit. Keying this
    /// by *kernel* name would recompile the whole unit once per kernel.
    ///
    /// An `RwLock` rather than a `Mutex`: after the first launch every lookup
    /// is a hit, so the hot path only needs a shared read.
    modules: RwLock<HashMap<&'static str, Arc<SendSyncModule>>>,
    /// One lock per translation unit, taken only on the compile-and-load path.
    ///
    /// The disk lock in [`cache::lock_entry`] is a *file* lock, which excludes
    /// other processes but not other threads of this one — and the compiler's
    /// intermediates are named per process, so two threads building the same
    /// module would write the same bundle concurrently and load the wreckage.
    /// This serialises that path per module while leaving different modules,
    /// and every cache hit, free to proceed in parallel.
    compiling: Vec<Mutex<()>>,
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
            modules: RwLock::new(HashMap::new()),
            compiling: (0..crate::ALL_IDS.len()).map(|_| Mutex::new(())).collect(),
        })
    }

    /// Return the loaded module, compiling it if this is the first use.
    pub fn get_or_load(&self, module: &Module) -> Result<Arc<SendSyncModule>, KernelError> {
        {
            let modules = self.read_modules()?;
            if let Some(loaded) = modules.get(module.name()) {
                return Ok(loaded.clone());
            }
        }

        let _compiling = self
            .compiling
            .get(module.index())
            .ok_or_else(|| {
                KernelError::Internal(format!("module index {} is out of range", module.index()))
            })?
            .lock()
            .map_err(|_| KernelError::Internal("kernel compile lock is poisoned".to_string()))?;
        // Re-check: another thread may have finished while this one waited.
        {
            let modules = self.read_modules()?;
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

        // Return whichever load *landed*, not this thread's. Without the outer
        // mutex the lookup above no longer excludes a concurrent loader, so two
        // threads can both reach here; a plain `insert` would hand the loser a
        // module whose `Arc` is unique and therefore `hipModuleUnload`s the
        // moment the caller drops it — leaving the resolved `hipFunction_t`
        // dangling and faulting the GPU at the next launch. The loser's own
        // module unloads harmlessly instead.
        let mut modules = self.write_modules()?;
        Ok(modules.entry(module.name()).or_insert(loaded).clone())
    }

    /// Resolve `name` inside `module`.
    ///
    /// The returned [`rocm_rs::hip::Function`] borrows nothing: it is a plain
    /// `hipFunction_t` with no `Drop`, so the caller holds no lock across the
    /// launch. That is the point — the previous shape kept a `Mutex<KernelCache>`
    /// guard alive for the whole launch, FFI call included, and so serialised
    /// every kernel on the device.
    ///
    /// The resolved handles are deliberately *not* memoised. Caching them in a
    /// per-module `RwLock<HashMap<String, _>>` was implemented and measured, and
    /// `hipModuleGetFunction` turned out to be free on this driver: 5000 launches
    /// took 0.155-0.160 ms with the cache and 0.156-0.182 ms without, and eight
    /// threads launching concurrently showed the same 1.42-1.51 ms either way.
    /// Not worth the extra raw-handle plumbing.
    pub fn function(
        &self,
        module: &Module,
        name: &str,
    ) -> Result<rocm_rs::hip::Function, KernelError> {
        let loaded = self.get_or_load(module)?;
        loaded.get_function(name).map_err(|e| {
            KernelError::Rocm(format!(
                "kernel `{name}` not found in module `{}`: {e}",
                module.name()
            ))
        })
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

    fn read_modules(
        &self,
    ) -> Result<
        std::sync::RwLockReadGuard<'_, HashMap<&'static str, Arc<SendSyncModule>>>,
        KernelError,
    > {
        self.modules
            .read()
            .map_err(|_| KernelError::Internal("kernel module cache lock is poisoned".to_string()))
    }

    fn write_modules(
        &self,
    ) -> Result<
        std::sync::RwLockWriteGuard<'_, HashMap<&'static str, Arc<SendSyncModule>>>,
        KernelError,
    > {
        self.modules
            .write()
            .map_err(|_| KernelError::Internal("kernel module cache lock is poisoned".to_string()))
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
fn stage_headers(src_dir: &Path) -> Result<(), KernelError> {
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

#[cfg(test)]
mod tests {
    use super::KernelCache;
    use std::sync::Arc;

    /// Dropping the outer `Mutex<KernelCache>` the ROCm backend used to hold
    /// exposed two races on the compile-and-load path, both of which faulted the
    /// GPU rather than failing cleanly: two threads writing the same
    /// process-named hipcc intermediates, and the loser of an insert race
    /// `hipModuleUnload`ing a module whose functions the caller had already
    /// resolved. Ten threads racing on one module is what reproduced them.
    #[test]
    fn concurrent_first_use_of_a_module_is_safe() {
        let cache = match KernelCache::new(0) {
            Ok(cache) => Arc::new(cache),
            // No ROCm device on this machine.
            Err(_) => return,
        };
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    cache
                        .function(&crate::UNARY, "ucopy_f32")
                        .map(|f| f.as_raw() as usize)
                })
            })
            .collect();

        let resolved: Vec<usize> = handles
            .into_iter()
            .map(|h| h.join().expect("thread panicked").expect("resolve failed"))
            .collect();
        // One module, one symbol: every thread must have landed on the same
        // handle, which is only true if exactly one load won.
        assert!(resolved.windows(2).all(|w| w[0] == w[1]), "{resolved:?}");
    }
}
