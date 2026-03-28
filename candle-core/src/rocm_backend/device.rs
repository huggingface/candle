use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{RocmError, RocmStorage, RocmStorageSlice, WrapErr};
use rocm_rs::hip::{
    kernel::AsKernelArg, Device as HipDevice, DeviceMemory, Dim3, Function, Module, Stream,
};

/// Unique identifier for ROCm devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// Kernel module cache
pub struct ModuleCache {
    modules: HashMap<String, Arc<Module>>,
}

impl ModuleCache {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    fn get(&self, name: &str) -> Option<Arc<Module>> {
        self.modules.get(name).cloned()
    }

    fn insert(&mut self, name: String, module: Arc<Module>) {
        self.modules.insert(name, module);
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    id: DeviceId,
    device: Arc<HipDevice>,
    stream: Arc<Stream>,
    modules: Arc<Mutex<ModuleCache>>,
    // TODO: rocBLAS integration
    // blas: Arc<Mutex<rocblas::Handle>>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({:?})", self.id)
    }
}

impl RocmDevice {
    /// Create a new ROCm device
    pub fn new(device_id: usize) -> Result<Self> {
        let device = HipDevice::new(device_id as i32)
            .map_err(|e| crate::Error::Msg(format!("Failed to create ROCm device: {}", e)))?;

        device
            .set_current()
            .map_err(|e| crate::Error::Msg(format!("Failed to set ROCm device: {}", e)))?;

        let stream = Stream::new()
            .map_err(|e| crate::Error::Msg(format!("Failed to create ROCm stream: {}", e)))?;

        Ok(Self {
            id: DeviceId::new(),
            device: Arc::new(device),
            stream: Arc::new(stream),
            modules: Arc::new(Mutex::new(ModuleCache::new())),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Allocate device memory
    pub fn alloc<T: AsKernelArg>(&self, len: usize) -> Result<DeviceMemory<T>> {
        DeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))
    }

    /// Allocate zero-initialized memory
    pub fn alloc_zeros<T: AsKernelArg + Default>(&self, len: usize) -> Result<DeviceMemory<T>> {
        let mut mem = DeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        mem.memset(0)
            .map_err(|e| crate::Error::Msg(format!("Failed to memset: {}", e)))?;
        Ok(mem)
    }

    /// Copy from host to device
    pub fn memcpy_htod<T: AsKernelArg, Src: AsRef<[T]>>(
        &self,
        src: &Src,
        dst: &mut DeviceMemory<T>,
    ) -> Result<()> {
        let slice = src.as_ref();
        dst.copy_from_host(slice)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy HtoD: {}", e)))
    }

    /// Copy from host to device (async)
    pub fn memcpy_htod_async<T: AsKernelArg + Clone, Src: AsRef<[T]>>(
        &self,
        src: &Src,
        dst: &mut DeviceMemory<T>,
    ) -> Result<()> {
        let slice = src.as_ref();
        dst.copy_from_host_async(slice.to_vec(), &self.stream)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy HtoD async: {}", e)))
    }

    /// Clone from host to device (allocates new memory)
    pub fn clone_htod<T: AsKernelArg + Clone, Src: AsRef<[T]>>(
        &self,
        src: &Src,
    ) -> Result<DeviceMemory<T>> {
        let slice = src.as_ref();
        let count = slice.len();
        let mut dst = DeviceMemory::new(count)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate: {}", e)))?;
        dst.copy_from_host(slice)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy HtoD: {}", e)))?;
        Ok(dst)
    }

    /// Copy from device to host
    pub fn memcpy_dtoh<T: AsKernelArg, Src: AsKernelArg>(
        &self,
        src: &DeviceMemory<T>,
        dst: &mut [T],
    ) -> Result<()> {
        src.copy_to_host(dst)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy DtoH: {}", e)))
    }

    /// Copy device to host and return as Vec
    pub fn clone_dtoh<T: AsKernelArg + Default + Clone>(
        &self,
        src: &DeviceMemory<T>,
    ) -> Result<Vec<T>> {
        let count = src.count();
        let mut dst: Vec<T> = vec![T::default(); count];
        src.copy_to_host(&mut dst)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy DtoH: {}", e)))?;
        Ok(dst)
    }

    /// Copy device to device
    pub fn memcpy_dtod<T: AsKernelArg>(
        &self,
        src: &DeviceMemory<T>,
        dst: &mut DeviceMemory<T>,
    ) -> Result<DeviceMemory<T>> {
        // For now, allocate new memory and copy
        let count = src.count().min(dst.count());
        let slice_size = std::mem::size_of::<T>() * count;

        unsafe {
            use rocm_rs::hip::ffi::{hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToDevice};
            let result = hipMemcpy(
                dst.as_ptr(),
                src.as_ptr(),
                slice_size,
                hipMemcpyKind_hipMemcpyDeviceToDevice,
            );
            if result != rocm_rs::hip::ffi::hipError_t_hipSuccess {
                return Err(crate::Error::Msg(format!(
                    "Device to device copy failed: {}",
                    result
                )));
            }
        }

        Ok(dst.clone())?;
        unimplemented!("memcpy_dtod needs proper implementation")
    }

    /// Load or get cached kernel module
    pub fn get_or_load_kernel(&self, kernel_name: &str) -> Result<Function> {
        let module_name = kernel_name.split('_').next().unwrap_or("default");

        // Check cache
        {
            let cache = self.modules.lock().unwrap();
            if let Some(module) = cache.get(module_name) {
                return module
                    .get_function(kernel_name)
                    .map_err(|e| crate::Error::Msg(format!("Failed to get function: {}", e)));
            }
        }

        // Load module (this would need actual kernel bytes)
        // For now, this is a placeholder - you'd embed kernel binaries
        let kernel_bytes = get_kernel_bytes(module_name);
        let module = Module::load_from_memory(kernel_bytes).map_err(|e| {
            crate::Error::Msg(format!("Failed to load module {}: {}", module_name, e))
        })?;

        let module = Arc::new(module);

        // Insert into cache
        {
            let mut cache = self.modules.lock().unwrap();
            cache.insert(module_name.to_string(), module.clone());
        }

        module
            .get_function(kernel_name)
            .map_err(|e| crate::Error::Msg(format!("Failed to get function: {}", e)))
    }

    /// Get the HIP stream
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| crate::Error::Msg(format!("Synchronize failed: {}", e)))
    }
}

/// Placeholder function to get kernel bytes
/// In real implementation, these would be embedded binaries from candle-rocm-kernels
fn get_kernel_bytes(module_name: &str) -> &'static [u8] {
    // Placeholder - would be: &candle_rocm_kernels::AFFINE
    &[]
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(device_id: usize) -> Result<Self> {
        Self::new(device_id)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.device.id() as usize,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();

        let slice = match dtype {
            DType::F32 => {
                let mem = self.alloc_zeros::<f32>(elem_count)?;
                RocmStorageSlice::F32(mem)
            }
            DType::U8 => {
                let mem = self.alloc_zeros::<u8>(elem_count)?;
                RocmStorageSlice::U8(mem)
            }
            _ => {
                // For now, use CPU fallback for other types
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm zero init",
                    dtype
                )));
            }
        };

        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();

        let slice = match dtype {
            DType::F32 => {
                let mem = self.alloc::<f32>(elem_count)?;
                RocmStorageSlice::F32(mem)
            }
            DType::U8 => {
                let mem = self.alloc::<u8>(elem_count)?;
                RocmStorageSlice::U8(mem)
            }
            _ => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm alloc",
                    dtype
                )));
            }
        };

        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let mem = match T::DTYPE {
            DType::F32 => {
                let slice: &[f32] = bytemuck::cast_slice(data);
                let mem = self.clone_htod(slice)?;
                RocmStorageSlice::F32(mem)
            }
            DType::U8 => {
                let mem = self.clone_htod(data)?;
                RocmStorageSlice::U8(mem)
            }
            _ => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported",
                    T::DTYPE
                )));
            }
        };

        Ok(RocmStorage {
            slice: mem,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        // Convert CpuStorage to device storage
        match storage {
            CpuStorage::F32(data) => {
                let mem = self.clone_htod(data.as_slice())?;
                Ok(RocmStorage {
                    slice: RocmStorageSlice::F32(mem),
                    device: self.clone(),
                })
            }
            CpuStorage::U8(data) => {
                let mem = self.clone_htod(data.as_slice())?;
                Ok(RocmStorage {
                    slice: RocmStorageSlice::U8(mem),
                    device: self.clone(),
                })
            }
            _ => Err(crate::Error::Msg(
                "Dtype not yet supported for ROCm storage_from_cpu_storage".to_string(),
            )),
        }
    }

    fn storage_from_cpu_storage_owned(&self, _storage: CpuStorage) -> Result<Self::Storage> {
        // For simplicity, delegate to reference version
        self.storage_from_cpu_storage(&_storage)
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _hi: f64,
    ) -> Result<Self::Storage> {
        // TODO: Implement using rocRAND
        Err(crate::Error::Msg(
            "Random generation not yet implemented for ROCm".to_string(),
        ))
    }

    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Self::Storage> {
        // TODO: Implement using rocRAND
        Err(crate::Error::Msg(
            "Random generation not yet implemented for ROCm".to_string(),
        ))
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        // TODO: Implement rocRAND seed
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        // TODO: Return actual seed
        Ok(0)
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
