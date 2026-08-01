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

/// Identifies one translation unit in the cache.
///
/// Built-in and caller-supplied modules share the map but not the namespace: a
/// custom module named `unary` must never be handed candle's own `unary`, which
/// a bare string key would do.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ModuleName {
    BuiltIn(&'static str),
    Custom(String),
}

impl ModuleName {
    fn as_str(&self) -> &str {
        match self {
            ModuleName::BuiltIn(name) => name,
            ModuleName::Custom(name) => name.as_str(),
        }
    }

    /// File-name stem for this module's cache entries. Custom names are
    /// arbitrary caller text, so they are both sanitised and prefixed — the
    /// prefix keeps a stray `unary` out of the built-in's entry even when the
    /// cache key would have separated them anyway.
    fn file_stem(&self) -> String {
        match self {
            ModuleName::BuiltIn(name) => cache::path_component(name),
            ModuleName::Custom(name) => format!("custom-{}", cache::path_component(name)),
        }
    }
}

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
    modules: RwLock<HashMap<ModuleName, Arc<SendSyncModule>>>,
    /// One lock per translation unit, created on first compile of that unit and
    /// taken only on the compile-and-load path.
    ///
    /// The disk lock in [`cache::lock_entry`] is a *file* lock, which excludes
    /// other processes but not other threads of this one — and the compiler's
    /// intermediates are named per process, so two threads building the same
    /// module would write the same bundle concurrently and load the wreckage.
    /// This serialises that path per module while leaving different modules,
    /// and every cache hit, free to proceed in parallel.
    ///
    /// A map rather than an array indexed by [`crate::Module::index`]: custom
    /// modules have no slot in [`crate::ALL_IDS`], and the outer lock is only
    /// ever taken on a miss, so the extra hash costs nothing that matters.
    compiling: Mutex<HashMap<ModuleName, Arc<Mutex<()>>>>,
}

impl KernelCache {
    /// Build a cache for the given device ordinal. The architecture is read
    /// from that specific device, so a second GPU of a different generation
    /// gets its own code objects.
    pub fn new(ordinal: usize) -> Result<Self, KernelError> {
        let arch = detect::gpu_arch(ordinal)?;
        let (version_tag, toolchain) = detect::rocm_version()?;
        let cache_dir = cache::base_cache_dir()?
            .join(format!("{}-{version_tag}", cache::path_component(&arch)));
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
            compiling: Mutex::new(HashMap::new()),
        })
    }

    /// Return the loaded module, compiling it if this is the first use.
    pub fn get_or_load(&self, module: &Module) -> Result<Arc<SendSyncModule>, KernelError> {
        self.load(ModuleName::BuiltIn(module.name()), module.source())
    }

    /// [`Self::get_or_load`] for a translation unit this crate does not own.
    ///
    /// `source` is CUDA-syntax HIP compiled exactly as the built-in modules
    /// are: the shim is force-included and candle's own headers
    /// (`cuda_utils.cuh`, `compatibility.cuh`, `binary_op_macros.cuh`) are on
    /// the include path, so a downstream kernel can be written against the same
    /// dialect. `module_name` names the cache entry and must be unique within
    /// the process; it lives in its own namespace, so it may repeat a built-in
    /// name without colliding.
    pub fn get_or_load_custom(
        &self,
        module_name: &str,
        source: &str,
    ) -> Result<Arc<SendSyncModule>, KernelError> {
        self.load(ModuleName::Custom(module_name.to_string()), source)
    }

    fn load(&self, name: ModuleName, source: &str) -> Result<Arc<SendSyncModule>, KernelError> {
        {
            let modules = self.read_modules()?;
            if let Some(loaded) = modules.get(&name) {
                return Ok(loaded.clone());
            }
        }

        let gate = self.compile_gate(&name)?;
        let _compiling = gate
            .lock()
            .map_err(|_| KernelError::Internal("kernel compile lock is poisoned".to_string()))?;
        // Re-check: another thread may have finished while this one waited.
        {
            let modules = self.read_modules()?;
            if let Some(loaded) = modules.get(&name) {
                return Ok(loaded.clone());
            }
        }

        let binary = self.code_object(&name, source)?;
        let loaded = Arc::new(SendSyncModule::load_data(&binary).map_err(|e| {
            KernelError::Compilation(format!(
                "failed to load module `{}` for {}: {e}",
                name.as_str(),
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
        Ok(modules.entry(name).or_insert(loaded).clone())
    }

    /// The per-module compile lock, created on first use.
    fn compile_gate(&self, name: &ModuleName) -> Result<Arc<Mutex<()>>, KernelError> {
        let mut gates = self.compiling.lock().map_err(|_| {
            KernelError::Internal("kernel compile gate map is poisoned".to_string())
        })?;
        Ok(gates.entry(name.clone()).or_default().clone())
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
        resolve(&loaded, module.name(), name)
    }

    /// [`Self::function`] for a translation unit this crate does not own; see
    /// [`Self::get_or_load_custom`].
    pub fn custom_function(
        &self,
        module_name: &str,
        source: &str,
        name: &str,
    ) -> Result<rocm_rs::hip::Function, KernelError> {
        let loaded = self.get_or_load_custom(module_name, source)?;
        resolve(&loaded, module_name, name)
    }

    /// Read the module's code object from the disk cache, compiling it first if
    /// it is not there.
    fn code_object(&self, name: &ModuleName, source: &str) -> Result<Vec<u8>, KernelError> {
        let stem = name.file_stem();
        let key = cache::module_key(source, crate::HEADERS, &self.arch, &self.toolchain);
        let cache_file = self.cache_dir.join(format!("{stem}_{key}.hsaco"));
        let forced = cache::force_recompile();

        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        // Hold the lock across the re-check, the compile and the write: the
        // intermediates below are written non-atomically, so two compilers of
        // the same entry could otherwise persist a corrupt code object.
        let _lock = cache::lock_entry(&self.cache_dir, &stem, &key)?;
        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        let binary = self.compile(name.as_str(), &stem, source, &key)?;
        cache::write_atomic(&cache_file, &binary)?;
        Ok(binary)
    }

    fn read_modules(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<ModuleName, Arc<SendSyncModule>>>, KernelError>
    {
        self.modules
            .read()
            .map_err(|_| KernelError::Internal("kernel module cache lock is poisoned".to_string()))
    }

    fn write_modules(
        &self,
    ) -> Result<
        std::sync::RwLockWriteGuard<'_, HashMap<ModuleName, Arc<SendSyncModule>>>,
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
    fn compile(
        &self,
        name: &str,
        stem: &str,
        source: &str,
        key: &str,
    ) -> Result<Vec<u8>, KernelError> {
        let src_file = self.src_dir.join(format!("{stem}_{key}.cu"));
        cache::write_atomic(&src_file, source.as_bytes())?;

        let shim_dir = self.src_dir.join("hip_shim");
        let bundle = self
            .cache_dir
            .join(format!("{stem}_{key}.{}.bundle", std::process::id()));

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
                name,
                self.arch,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let binary = unbundle(&bundle, &self.arch);
        let _ = fs::remove_file(&bundle);
        binary
    }
}

/// Look `name` up in an already-loaded module.
fn resolve(
    loaded: &SendSyncModule,
    module_name: &str,
    name: &str,
) -> Result<rocm_rs::hip::Function, KernelError> {
    loaded.get_function(name).map_err(|e| {
        KernelError::Rocm(format!(
            "kernel `{name}` not found in module `{module_name}`: {e}"
        ))
    })
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

    /// A source this crate does not own, written in the same dialect as
    /// `candle-kernels/src/*.cu`. The `#include` is the point of the test as
    /// much as the kernel is: a downstream module has to reach candle's staged
    /// headers, not just the shim.
    const CUSTOM: &str = r#"
#include "cuda_utils.cuh"

extern "C" __global__ void custom_scale_f32(const float *x, float *y, const size_t n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * 3.0f;
    }
}
"#;

    /// A caller-supplied source has to compile and resolve through the same
    /// cache the built-ins use — the whole point of the custom entry point.
    #[test]
    fn a_custom_source_compiles_and_resolves() {
        let cache = match KernelCache::new(0) {
            Ok(cache) => cache,
            // No ROCm device on this machine.
            Err(_) => return,
        };
        cache
            .custom_function("candle_rocm_kernels_test", CUSTOM, "custom_scale_f32")
            .expect("custom module failed to compile");
        // Second call is the in-memory hit.
        cache
            .custom_function("candle_rocm_kernels_test", CUSTOM, "custom_scale_f32")
            .expect("custom module failed to resolve from cache");
    }

    /// Custom modules live in their own namespace. Keying both kinds by a bare
    /// string would hand a downstream crate that named its module `unary`
    /// candle's `unary` instead — a wrong-kernel launch, not an error.
    #[test]
    fn a_custom_module_does_not_shadow_a_built_in_of_the_same_name() {
        let cache = match KernelCache::new(0) {
            Ok(cache) => cache,
            Err(_) => return,
        };
        // Load the custom one first, so a shared namespace would poison the
        // built-in lookup rather than the other way round.
        cache
            .custom_function("unary", CUSTOM, "custom_scale_f32")
            .expect("custom module failed to compile");
        cache
            .function(&crate::UNARY, "ucopy_f32")
            .expect("built-in `unary` was shadowed by the custom module");
        // And the reverse: the built-in must not satisfy the custom lookup.
        assert!(cache.custom_function("unary", CUSTOM, "ucopy_f32").is_err());
    }

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
