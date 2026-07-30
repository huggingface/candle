use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape};
use candle_rocm_kernels::KernelCache;
use half::{bf16, f16};
use std::sync::{Arc, Mutex, RwLock};

use super::wrappers::{
    SendSyncDeviceMemory, SendSyncMIOpenHandle, SendSyncPseudoRng, SendSyncRocblasHandle,
    SendSyncStream,
};
use super::{RocmError, RocmStorage, RocmStorageSlice};
use rocm_rs::hip::Device as HipDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    id: DeviceId,
    device: Arc<HipDevice>,
    pub(crate) stream: Arc<SendSyncStream>,
    rocrand: Arc<Mutex<SendSyncPseudoRng>>,
    seed_value: Arc<RwLock<u64>>,
    pub(crate) blas: Arc<SendSyncRocblasHandle>,
    pub(crate) miopen: Arc<SendSyncMIOpenHandle>,
    kernel_manager: Arc<Mutex<KernelCache>>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({:?})", self.id)
    }
}

impl RocmDevice {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = HipDevice::new(device_id as i32)?;
        device.set_current()?;
        let stream = device.get_stream()?;

        let mut rocrand = SendSyncPseudoRng::new(rocm_rs::rocrand::rng_type::PSEUDO_DEFAULT)
            .map_err(|e| crate::Error::Msg(format!("Failed to create rocrand generator: {}", e)))?;
        let seed = 299792458u64;
        rocrand
            .set_seed(seed)
            .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;

        let blas = SendSyncRocblasHandle::new().map_err(|e| RocmError::Rocblas(e.to_string()))?;
        blas.set_stream(&stream)
            .map_err(|e| RocmError::Rocblas(e.to_string()))?;

        let miopen =
            SendSyncMIOpenHandle::new(&stream).map_err(|e| RocmError::MIOpen(e.to_string()))?;

        let kernel_manager =
            Arc::new(Mutex::new(KernelCache::new(None).map_err(|e| {
                crate::Error::Msg(format!("Failed to create kernel cache: {}", e))
            })?));

        Ok(Self {
            id: DeviceId::new(),
            device: Arc::new(device),
            stream: Arc::new(SendSyncStream(stream)),
            rocrand: Arc::new(Mutex::new(rocrand)),
            seed_value: Arc::new(RwLock::new(seed)),
            blas: Arc::new(blas),
            miopen: Arc::new(miopen),
            kernel_manager,
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn alloc<T>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        SendSyncDeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))
    }

    pub fn alloc_zeros<T: Default + Clone>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        let mut mem = SendSyncDeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        mem.memset(0)
            .map_err(|e| crate::Error::Msg(format!("Failed to memset: {}", e)))?;
        Ok(mem)
    }

    pub fn clone_htod<T: Clone>(&self, src: &[T]) -> Result<SendSyncDeviceMemory<T>> {
        let count = src.len();
        let mut dst = SendSyncDeviceMemory::new(count)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        dst.copy_from_host(src)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy host to device: {}", e)))?;
        Ok(dst)
    }

    pub fn clone_dtoh<T: Default + Clone>(&self, src: &SendSyncDeviceMemory<T>) -> Result<Vec<T>> {
        let count = src.count();
        let mut dst: Vec<T> = vec![T::default(); count];
        src.copy_to_host(&mut dst)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy device to host: {}", e)))?;
        Ok(dst)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| crate::Error::Msg(format!("Synchronize failed: {}", e)))
    }

    pub(crate) fn kernel_manager(&self) -> &std::sync::Mutex<KernelCache> {
        &self.kernel_manager
    }

    pub(crate) fn miopen(&self) -> &Arc<SendSyncMIOpenHandle> {
        &self.miopen
    }

    /// Get a reference to the underlying HIP stream.
    /// This is public so that candle-nn and other crates can launch custom kernels.
    pub fn stream(&self) -> &rocm_rs::hip::Stream {
        &self.stream.0
    }

    /// Locks the rocrand generator.
    ///
    /// The lock is deliberately not unwrapped: a failure inside any `rand_*`
    /// call would poison the mutex and turn every later call into a panic.
    pub(super) fn rocrand(&self) -> Result<std::sync::MutexGuard<'_, SendSyncPseudoRng>> {
        self.rocrand
            .lock()
            .map_err(|_| crate::Error::Msg("Failed to lock rocrand generator".to_string()))
    }

    /// Get or load a kernel function from the cache.
    /// This is public so that candle-nn and other crates can launch custom kernels.
    pub fn get_or_load_func(
        &self,
        kernel_name: &str,
        module: &candle_rocm_kernels::Module,
    ) -> crate::Result<rocm_rs::hip::Function> {
        let kernel_manager = self
            .kernel_manager
            .lock()
            .map_err(|_| crate::Error::Msg("Failed to lock kernel manager".to_string()))?;
        let module = kernel_manager
            .get_or_load(module)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        let func = module
            .get_function(kernel_name)
            .map_err(|e| crate::Error::Msg(format!("Kernel {} not found: {}", kernel_name, e)))?;
        Ok(func)
    }
}

macro_rules! dispatch_dtypes {
    ($method:ident, ($self:expr, $elem_count:expr, $dtype:expr) -> |$slice:ident| $body:expr) => {
        match $dtype {
            DType::U8 => {
                let $slice = RocmStorageSlice::U8($self.$method::<u8>($elem_count)?);
                $body
            }
            DType::U32 => {
                let $slice = RocmStorageSlice::U32($self.$method::<u32>($elem_count)?);
                $body
            }
            DType::I16 => {
                let $slice = RocmStorageSlice::I16($self.$method::<i16>($elem_count)?);
                $body
            }
            DType::I32 => {
                let $slice = RocmStorageSlice::I32($self.$method::<i32>($elem_count)?);
                $body
            }
            DType::I64 => {
                let $slice = RocmStorageSlice::I64($self.$method::<i64>($elem_count)?);
                $body
            }
            DType::BF16 => {
                let $slice = RocmStorageSlice::BF16($self.$method::<bf16>($elem_count)?);
                $body
            }
            DType::F16 => {
                let $slice = RocmStorageSlice::F16($self.$method::<f16>($elem_count)?);
                $body
            }
            DType::F32 => {
                let $slice = RocmStorageSlice::F32($self.$method::<f32>($elem_count)?);
                $body
            }
            DType::F64 => {
                let $slice = RocmStorageSlice::F64($self.$method::<f64>($elem_count)?);
                $body
            }
            DType::F8E4M3 => {
                let $slice = RocmStorageSlice::F8E4M3($self.$method::<u8>($elem_count)?);
                $body
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm",
                    $dtype
                )));
            }
        }
    };
}

impl RocmDevice {
    /// Single host-to-device path shared by `storage_from_slice` and
    /// `storage_from_cpu_storage`.
    fn slice_from_cpu_storage_ref(&self, data: CpuStorageRef<'_>) -> Result<RocmStorageSlice> {
        let slice = match data {
            CpuStorageRef::U8(data) => RocmStorageSlice::U8(self.clone_htod(data)?),
            CpuStorageRef::U32(data) => RocmStorageSlice::U32(self.clone_htod(data)?),
            CpuStorageRef::I16(data) => RocmStorageSlice::I16(self.clone_htod(data)?),
            CpuStorageRef::I32(data) => RocmStorageSlice::I32(self.clone_htod(data)?),
            CpuStorageRef::I64(data) => RocmStorageSlice::I64(self.clone_htod(data)?),
            CpuStorageRef::BF16(data) => RocmStorageSlice::BF16(self.clone_htod(data)?),
            CpuStorageRef::F16(data) => RocmStorageSlice::F16(self.clone_htod(data)?),
            CpuStorageRef::F32(data) => RocmStorageSlice::F32(self.clone_htod(data)?),
            CpuStorageRef::F64(data) => RocmStorageSlice::F64(self.clone_htod(data)?),
            CpuStorageRef::F8E4M3(data) => {
                // `RocmStorageSlice::F8E4M3` holds bytes, and `float8::F8E4M3`
                // is a `repr(transparent)` wrapper over a single `u8` (asserted
                // in `rocm_backend::mod`), so the slice can be viewed as bytes
                // without a copy.
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
                RocmStorageSlice::F8E4M3(self.clone_htod(bytes)?)
            }
            CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F4(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(crate::Error::Msg(
                    "F6E2M3/F6E3M2/F4/F8E8M0 storage is not yet supported for ROCm".to_string(),
                ))
            }
        };
        Ok(slice)
    }
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
        dispatch_dtypes!(alloc_zeros, (self, elem_count, dtype) -> |slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        dispatch_dtypes!(alloc, (self, elem_count, dtype) -> |slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        Ok(RocmStorage {
            slice: self.slice_from_cpu_storage_ref(T::cpu_storage_ref(data))?,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let data = match storage {
            CpuStorage::U8(v) => CpuStorageRef::U8(v),
            CpuStorage::U32(v) => CpuStorageRef::U32(v),
            CpuStorage::I16(v) => CpuStorageRef::I16(v),
            CpuStorage::I32(v) => CpuStorageRef::I32(v),
            CpuStorage::I64(v) => CpuStorageRef::I64(v),
            CpuStorage::BF16(v) => CpuStorageRef::BF16(v),
            CpuStorage::F16(v) => CpuStorageRef::F16(v),
            CpuStorage::F32(v) => CpuStorageRef::F32(v),
            CpuStorage::F64(v) => CpuStorageRef::F64(v),
            CpuStorage::F8E4M3(v) => CpuStorageRef::F8E4M3(v),
            CpuStorage::F6E2M3(v) => CpuStorageRef::F6E2M3(v),
            CpuStorage::F6E3M2(v) => CpuStorageRef::F6E3M2(v),
            CpuStorage::F4(v) => CpuStorageRef::F4(v),
            CpuStorage::F8E8M0(v) => CpuStorageRef::F8E8M0(v),
        };
        Ok(RocmStorage {
            slice: self.slice_from_cpu_storage_ref(data)?,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        self.rand_uniform_impl(shape, dtype, lo, hi)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        self.rand_normal_impl(shape, dtype, mean, std)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut rocrand = self.rocrand()?;
        rocrand
            .set_seed(seed)
            .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;
        *self
            .seed_value
            .write()
            .map_err(|_| crate::Error::Msg("Failed to lock ROCm seed value".to_string()))? = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let seed = self
            .seed_value
            .read()
            .map_err(|_| crate::Error::Msg("Failed to lock ROCm seed value".to_string()))?;
        Ok(*seed)
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
