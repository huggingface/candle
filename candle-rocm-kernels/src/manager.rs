use crate::cache::CacheManager;
use crate::error::RocmKernelError;
use crate::source::Source;
use rocm_rs::hip::{Device, Module};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Manages kernel compilation and caching using AOT cache
/// Modules are stored as Arc<Module> to allow cloning
pub struct KernelManager {
    cache: CacheManager,
    modules: Mutex<HashMap<String, Arc<Module>>>,
}

impl KernelManager {
    /// Create a new KernelManager for the given device
    pub fn new(device: &Device) -> Result<Self, RocmKernelError> {
        let cache = CacheManager::new(device)?;
        Ok(Self {
            cache,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// Get or compile a module for the given source
    pub fn get_or_compile_module(&self, source: Source) -> Result<Arc<Module>, RocmKernelError> {
        let name = source.name();

        // Check if already loaded
        {
            let modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules mutex".to_string())
            })?;
            if let Some(module) = modules.get(name) {
                return Ok(module.clone());
            }
        }

        // Get or compile the binary
        let binary = self.cache.get_or_compile(name, source.code())?;

        // Load the module from binary
        let module = Module::load_data(&binary).map_err(|e| {
            RocmKernelError::Compilation(format!(
                "Failed to load module {} from compiled binary: {}",
                name, e
            ))
        })?;

        let module = Arc::new(module);

        // Store in cache
        {
            let mut modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules mutex".to_string())
            })?;
            modules.insert(name.to_string(), module.clone());
        }

        Ok(module)
    }

    /// Get the GPU architecture
    pub fn arch(&self) -> &str {
        self.cache.arch()
    }

    /// Get the ROCm version
    pub fn rocm_version(&self) -> &str {
        self.cache.rocm_version()
    }
}
