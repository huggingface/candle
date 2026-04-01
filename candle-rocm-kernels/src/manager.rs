use crate::cache::CacheManager;
use crate::error::RocmKernelError;
use crate::source::Source;
use crate::wrappers::SendSyncModule;
use rocm_rs::hip::Device;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct KernelManager {
    cache: CacheManager,
    modules: Mutex<HashMap<String, Arc<SendSyncModule>>>,
}

impl KernelManager {
    pub fn new(device: &Device) -> Result<Self, RocmKernelError> {
        let cache = CacheManager::new(device)?;
        Ok(Self {
            cache,
            modules: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_or_compile_module(
        &self,
        source: Source,
    ) -> Result<Arc<SendSyncModule>, RocmKernelError> {
        let name = source.name();

        {
            let modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules mutex".to_string())
            })?;
            if let Some(module) = modules.get(name) {
                return Ok(module.clone());
            }
        }

        let binary = self.cache.get_or_compile(name, source.code())?;

        let module = SendSyncModule::load_data(&binary).map_err(|e| {
            RocmKernelError::Compilation(format!(
                "Failed to load module {} from compiled binary: {}",
                name, e
            ))
        })?;

        let module = Arc::new(module);

        {
            let mut modules = self.modules.lock().map_err(|_| {
                RocmKernelError::Internal("Failed to lock modules mutex".to_string())
            })?;
            modules.insert(name.to_string(), module.clone());
        }

        Ok(module)
    }

    pub fn arch(&self) -> &str {
        self.cache.arch()
    }

    pub fn rocm_version(&self) -> &str {
        self.cache.rocm_version()
    }
}
