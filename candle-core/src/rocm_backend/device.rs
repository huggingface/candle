//! ROCm device wrapper
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! CUDA parity verified by: TEAM-497, TEAM-498
//! Updated by: TEAM-507 (Module caching for CUDA parity)
//! 
//! Note: Like CUDA's CudaDevice, this is a thin wrapper around the underlying library.
//! CUDA exposes `cudarc` directly, we expose `rocm_rs` directly (see mod.rs).
//! Users can access rocm-rs APIs directly when needed via `hip_device()` or `rocm_rs::*`.

use super::error::{Result, RocmError};
use super::kernels_module;
use rocm_rs::hip::{Device as HipDevice, DeviceProperties as HipProps, Module as HipModule};
use rocm_rs::rocrand::PseudoRng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

// TEAM-507: Module cache for CUDA parity
// Matches cuda_backend/device.rs:244-246
struct ModuleStore {
    mdls: [Option<HipModule>; kernels_module::ALL_IDS.len()],
}

// TEAM-509: Random number generator wrapper (CUDA parity)
// Matches cuda_backend/device.rs:26-27
struct RocmRng(PseudoRng);
unsafe impl Send for RocmRng {}

// TEAM-510: ROCm function wrapper for ug-rocm (CUDA parity)
// Matches cuda_backend/device.rs:108-125
/// Wrapper for compiled ROCm kernel function
pub struct RocmFunc {
    func: rocm_rs::hip::Function,
    stream: Arc<rocm_rs::hip::Stream>,
}

impl std::ops::Deref for RocmFunc {
    type Target = rocm_rs::hip::Function;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl RocmFunc {
    pub fn into_hip_function(self) -> rocm_rs::hip::Function {
        self.func
    }
}

/// Candle wrapper for ROCm device
/// 
/// This is a thin wrapper around rocm_rs::hip::Device.
/// We don't reimplement - we just wrap the existing API.
#[derive(Clone)]
pub struct RocmDevice {
    inner: HipDevice,
    // TEAM-507: Module cache for CUDA parity
    modules: Arc<RwLock<ModuleStore>>,
    // TEAM-509: Custom module cache for runtime-compiled kernels (CUDA parity)
    // Matches cuda_backend/device.rs:38
    custom_modules: Arc<RwLock<HashMap<String, HipModule>>>,
    // TEAM-509: Random number generator (CUDA parity)
    // Matches cuda_backend/device.rs:41
    rocrand: Arc<Mutex<RocmRng>>,
    // TEAM-510: HIP stream for kernel execution (CUDA parity)
    // Matches cuda_backend/device.rs stream field
    pub(crate) stream: Arc<rocm_rs::hip::Stream>,
}

// TEAM-507: Manual Debug impl (HipModule doesn't implement Debug)
// Updated by: TEAM-509 (added custom_modules, rocrand)
// Updated by: TEAM-510 (added stream)
impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmDevice")
            .field("inner", &self.inner)
            .field("modules", &"<cached modules>")
            .field("custom_modules", &"<custom module cache>")
            .field("rocrand", &"<random generator>")
            .field("stream", &"<hip stream>")
            .finish()
    }
}

impl RocmDevice {
    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs:262-279 (BackendDevice::new)
    // Updated by: TEAM-507 | Added module cache initialization
    // Updated by: TEAM-509 | Added custom module cache and rocRAND generator
    // Updated by: TEAM-510 | Added stream initialization
    /// Create a new ROCm device
    pub fn new(id: usize) -> Result<Self> {
        let inner = HipDevice::new(id as i32)?;
        inner.set_current()?;
        
        // TEAM-507: Initialize module cache (CUDA parity)
        let module_store = ModuleStore {
            mdls: [const { None }; kernels_module::ALL_IDS.len()],
        };
        
        // TEAM-509: Initialize rocRAND generator (CUDA parity)
        // Use XORWOW as default (matches CUDA's curand default)
        let rng = PseudoRng::new(rocm_rs::rocrand::rng_type::XORWOW)
            .map_err(|e| RocmError::InternalError(&format!("Failed to create rocRAND generator: {:?}", e)))?;
        
        // TEAM-510: Initialize HIP stream (CUDA parity)
        let stream = Arc::new(rocm_rs::hip::Stream::new()?);
        
        Ok(Self { 
            inner,
            modules: Arc::new(RwLock::new(module_store)),
            custom_modules: Arc::new(RwLock::new(HashMap::new())),
            rocrand: Arc::new(Mutex::new(RocmRng(rng))),
            stream,
        })
    }

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs:188-190
    /// Get device ID
    pub fn id(&self) -> usize {
        self.inner.id() as usize
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc properties directly)
    /// Get device name
    pub fn name(&self) -> Result<String> {
        let props = self.inner.properties()?;
        Ok(props.name)
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc properties directly)
    /// Get compute capability (major, minor)
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        let props = self.inner.properties()?;
        Ok((props.major, props.minor))
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc stream sync directly)
    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()?;
        Ok(())
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc memory info directly)
    /// Get total memory in bytes
    pub fn total_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.total)
    }

    // Created by: TEAM-488 | ROCm-specific (CUDA uses cudarc memory info directly)
    /// Get free memory in bytes
    pub fn free_memory(&self) -> Result<usize> {
        let info = rocm_rs::hip::memory_info()?;
        Ok(info.free)
    }

    // Created by: TEAM-488 | ROCm-specific (no CUDA equivalent)
    /// Get underlying rocm-rs device (for kernel operations)
    /// 
    /// This allows direct access to rocm-rs APIs when needed.
    pub fn hip_device(&self) -> &HipDevice {
        &self.inner
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs:56-58
    /// Allocate device memory
    pub unsafe fn alloc<T>(&self, len: usize) -> Result<rocm_rs::hip::DeviceMemory<T>> {
        self.inner.alloc::<T>(len).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs:59-65
    /// Allocate zero-initialized device memory
    pub fn alloc_zeros<T>(&self, len: usize) -> Result<rocm_rs::hip::DeviceMemory<T>>
    where
        T: Default + Clone,
    {
        self.inner.alloc_zeros::<T>(len).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from host (stack) to device
    pub fn memcpy_stod<T: Clone>(&self, src: &[T]) -> Result<rocm_rs::hip::DeviceMemory<T>> {
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        self.inner.memcpy_htod(src, &mut dst).map_err(RocmError::from)?;
        Ok(dst)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from host to device
    pub fn memcpy_htod<T>(&self, src: &[T], dst: &mut rocm_rs::hip::DeviceMemory<T>) -> Result<()> {
        self.inner.memcpy_htod(src, dst).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (memcpy operations)
    /// Copy data from device to host (vector)
    pub fn memcpy_dtov<T: Clone>(&self, src: &rocm_rs::hip::DeviceMemory<T>) -> Result<Vec<T>> {
        self.inner.memcpy_dtoh(src).map_err(RocmError::from)
    }

    // Created by: TEAM-502 | CUDA parity: cuda_backend/device.rs (kernel loading)
    // Updated by: TEAM-507 | Now takes kernels_module::Module and caches modules
    /// Get or load a kernel function from candle-kernels Module
    /// 
    /// This matches CUDA's pattern exactly:
    /// - Takes &kernels_module::Module (not raw bytes)
    /// - Caches loaded modules for reuse
    /// - Only loads each module once
    pub fn get_or_load_func(&self, name: &str, mdl: &kernels_module::Module) -> Result<rocm_rs::hip::Function> {
        // Try to get from cache first
        let ms = self.modules.read().unwrap();
        if let Some(module) = ms.mdls[mdl.index()].as_ref() {
            let func = module.get_function(name).map_err(RocmError::from)?;
            return Ok(func);
        }
        drop(ms);
        
        // Not cached, load it
        let mut ms = self.modules.write().unwrap();
        let hip_module = self.inner.load_module(mdl.hsaco()).map_err(RocmError::from)?;
        ms.mdls[mdl.index()] = Some(hip_module.clone());
        let func = hip_module.get_function(name).map_err(RocmError::from)?;
        Ok(func)
    }
    
    // TEAM-507: Keep old signature for quantized_stub (runtime compilation)
    /// Get or load a kernel function from raw HSACO binary (for runtime compilation)
    /// 
    /// This is used by quantized_stub which uses runtime compilation.
    /// For pre-compiled kernels, use get_or_load_func() instead.
    pub fn get_or_load_func_raw(&self, name: &str, hsaco: &[u8]) -> Result<rocm_rs::hip::Function> {
        let module = self.inner.load_module(hsaco).map_err(RocmError::from)?;
        module.get_function(name).map_err(RocmError::from)
    }
    
    // TEAM-509: Custom module cache for runtime-compiled kernels (CUDA parity)
    // Matches cuda_backend/device.rs:192-220
    /// Get or load a custom kernel function (with caching)
    /// 
    /// This caches runtime-compiled modules by name to avoid reloading.
    /// Used for custom operations and quantized kernels.
    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        hsaco: &[u8],
    ) -> Result<rocm_rs::hip::Function> {
        // Try to get from cache first
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name) {
            let func = mdl.get_function(fn_name).map_err(RocmError::from)?;
            return Ok(func);
        }
        drop(ms);
        
        // Not cached, load it
        let mut ms = self.custom_modules.write().unwrap();
        // Double-check after acquiring write lock
        if let Some(mdl) = ms.get(module_name) {
            let func = mdl.get_function(fn_name).map_err(RocmError::from)?;
            return Ok(func);
        }
        
        let module = self.inner.load_module(hsaco).map_err(RocmError::from)?;
        ms.insert(module_name.to_string(), module.clone());
        let func = module.get_function(fn_name).map_err(RocmError::from)?;
        Ok(func)
    }
    
    // TEAM-510: Compile ug kernel to HIP code (CUDA parity)
    // Matches cuda_backend/device.rs:167-186
    /// Compile a ug kernel to HIP and load it
    /// 
    /// This generates HIP C++ code from a ug kernel, compiles it using hipcc,
    /// and loads the resulting HSACO module.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<RocmFunc> {
        let mut buf = vec![];
        ug_rocm::code_gen::gen(&mut buf, func_name, &kernel)?;
        let hip_code = String::from_utf8(buf)?;
        
        // Compile HIP code to HSACO using hipcc
        // TODO: This requires hipcc to be installed and in PATH
        // For now, we'll use rocm-rs's compile_and_load if available
        let module = rocm_rs::hip::module::compile_and_load(&hip_code, &[])
            .map_err(RocmError::from)?;
        let func = module.get_function(func_name).map_err(RocmError::from)?;
        
        Ok(RocmFunc {
            func,
            stream: self.stream.clone(),
        })
    }
    
    // TEAM-509: Set random seed (CUDA parity)
    // Matches cuda_backend/device.rs set_seed behavior
    /// Set the random number generator seed
    /// 
    /// This allows reproducible random number generation.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        let mut rng = self.rocrand.lock().unwrap();
        rng.0.set_seed(seed)
            .map_err(|e| RocmError::InternalError(&format!("Failed to set rocRAND seed: {:?}", e)))?;
        Ok(())
    }
    
    // TEAM-509: Access to rocRAND generator (internal use)
    /// Get reference to rocRAND generator (for internal use)
    pub(crate) fn rocrand(&self) -> &Arc<Mutex<RocmRng>> {
        &self.rocrand
    }
}

impl PartialEq for RocmDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for RocmDevice {}

impl std::hash::Hash for RocmDevice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export device_count publicly)
/// Get number of ROCm devices
pub fn device_count() -> Result<usize> {
    let count = rocm_rs::hip::device_count()?;
    Ok(count as usize)
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export is_available publicly)
/// Check if ROCm is available
pub fn is_available() -> bool {
    rocm_rs::hip::is_hip_available()
}

// Created by: TEAM-488 | ROCm-specific (CUDA doesn't export runtime_version publicly)
/// Get ROCm runtime version
pub fn runtime_version() -> Result<i32> {
    let version = rocm_rs::hip::runtime_version()?;
    Ok(version)
}
