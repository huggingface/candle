//! Runtime compilation of the shared kernel sources with `hipcc`.
//!
//! A module is compiled once per cache key — source, staged headers, compile
//! flags, GPU architecture, toolchain version and crate version — and cached on
//! disk, so the cost is paid on first use only.

mod cache;
mod detect;
mod hipcc;
#[cfg(test)]
mod tests;

pub use detect::MmqTiles;

use crate::error::KernelError;
use crate::wrappers::SendSyncModule;
use crate::Module;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

/// Identifies one translation unit in the cache.
///
/// Built-in and caller-supplied modules share the map but not the namespace: a
/// custom module named `unary` must never be handed candle's own `unary`, which
/// a bare string key would do.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ModuleName {
    BuiltIn(&'static str),
    /// Name *and* source digest. A caller may revise its source and keep the
    /// name — an edit between runs, a kernel specialised per shape — and the
    /// revision must not resolve against the module it replaces. The disk cache
    /// has always keyed on the source; without the digest here the in-memory
    /// map shadowed it and silently served the older code object.
    Custom {
        name: String,
        source: [u8; 32],
    },
}

impl ModuleName {
    fn as_str(&self) -> &str {
        match self {
            ModuleName::BuiltIn(name) => name,
            ModuleName::Custom { name, .. } => name.as_str(),
        }
    }

    /// File-name stem for this module's cache entries. Custom names are
    /// arbitrary caller text, so they are both sanitised and prefixed — the
    /// prefix keeps a stray `unary` out of the built-in's entry even when the
    /// cache key would have separated them anyway.
    fn file_stem(&self) -> String {
        match self {
            ModuleName::BuiltIn(name) => cache::path_component(name),
            ModuleName::Custom { name, .. } => format!("custom-{}", cache::path_component(name)),
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
    /// Keyed by module identity, i.e. one entry per translation unit. Keying
    /// this by *kernel* name would recompile the whole unit once per kernel.
    ///
    /// An `RwLock` rather than a `Mutex`: after the first launch every lookup
    /// is a hit, so the hot path only needs a shared read.
    ///
    /// Nothing is ever evicted, and that is load-bearing rather than an
    /// oversight: [`Self::function`] hands out a bare `hipFunction_t` that
    /// borrows nothing, so unloading a module would dangle every handle already
    /// resolved from it and fault the GPU at the next launch. Built-ins are
    /// finite; a caller that generates a module per shape holds the memory it
    /// asks for, which is the only safe reading of the API it used.
    modules: RwLock<HashMap<ModuleName, Arc<SendSyncModule>>>,
    /// One lock per translation unit, created on first compile of that unit,
    /// taken only on the compile-and-load path and dropped once the module is
    /// resident.
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

        hipcc::stage_headers(&src_dir)?;

        Ok(Self {
            cache_dir,
            src_dir,
            arch,
            toolchain,
            modules: RwLock::new(HashMap::new()),
            compiling: Mutex::new(HashMap::new()),
        })
    }

    /// Which MMQ tile geometry `quantized.cu` was compiled with for this
    /// device. A `mul_mat_q*` launch has to derive its grid and block from the
    /// same set, so the host cannot assume one.
    pub fn mmq_tiles(&self) -> MmqTiles {
        detect::mmq_tiles(&self.arch)
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
    /// dialect. `module_name` names the cache entry; it lives in its own
    /// namespace, so it may repeat a built-in name without colliding.
    ///
    /// The entry is keyed on the source as well as the name, so revising a
    /// source compiles the revision rather than returning the module it
    /// replaces. That costs one SHA-256 pass over `source` per call — a few
    /// tens of microseconds for a large kernel, which is real next to a launch.
    /// A caller launching in a loop should hold the [`rocm_rs::hip::Function`]
    /// (it borrows nothing and stays valid, since modules are never unloaded)
    /// instead of resolving it per launch. Built-in lookups pay none of this.
    ///
    /// Every distinct source stays resident for the life of the process:
    /// handing out unborrowed function handles rules out ever unloading one.
    /// Generating a module per shape therefore grows memory without bound; if
    /// that is the intent, bound the set of sources you ask for.
    pub fn get_or_load_custom(
        &self,
        module_name: &str,
        source: &str,
    ) -> Result<Arc<SendSyncModule>, KernelError> {
        self.load(
            ModuleName::Custom {
                name: module_name.to_string(),
                source: cache::source_digest(source),
            },
            source,
        )
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
        let landed = {
            let mut modules = self.write_modules()?;
            modules.entry(name.clone()).or_insert(loaded).clone()
        };
        self.release_compile_gate(&name)?;
        Ok(landed)
    }

    /// The per-module compile lock, created on first use.
    fn compile_gate(&self, name: &ModuleName) -> Result<Arc<Mutex<()>>, KernelError> {
        let mut gates = self.compiling.lock().map_err(|_| {
            KernelError::Internal("kernel compile gate map is poisoned".to_string())
        })?;
        Ok(gates.entry(name.clone()).or_default().clone())
    }

    /// Forget a unit's gate now that its module is resident, so the map does not
    /// retain one lock per key for the life of the process.
    ///
    /// Safe because the module is written first: a thread still queued on this
    /// gate holds it through its own `Arc` and finds the module on its re-check,
    /// and a thread that arrives afterwards and creates a fresh gate never gets
    /// as far as compiling — the lookup that precedes the gate is already a hit.
    fn release_compile_gate(&self, name: &ModuleName) -> Result<(), KernelError> {
        let mut gates = self.compiling.lock().map_err(|_| {
            KernelError::Internal("kernel compile gate map is poisoned".to_string())
        })?;
        gates.remove(name);
        Ok(())
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
        let key = cache::module_key(source, crate::HEADERS, &self.arch, &self.toolchain);
        let entry = format!("{}_{key}", name.file_stem());
        let cache_file = self.cache_dir.join(format!("{entry}.hsaco"));
        let forced = cache::force_recompile();

        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        // Hold the lock across the re-check, the compile and the write: the
        // intermediates below are written non-atomically, so two compilers of
        // the same entry could otherwise persist a corrupt code object.
        let _lock = cache::lock_entry(&self.cache_dir, &entry)?;
        if !forced {
            if let Ok(binary) = fs::read(&cache_file) {
                return Ok(binary);
            }
        }

        let binary = hipcc::compile(
            &self.src_dir,
            &self.cache_dir,
            &self.arch,
            name.as_str(),
            &entry,
            source,
        )?;
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
