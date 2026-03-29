use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Result, Shape};
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{RocmError, RocmStorage, RocmStorageSlice};
use rocm_rs::hip::{kernel::AsKernelArg, Device as HipDevice, DeviceMemory, Module, Stream};
use rocm_rs::rocblas;
use rocm_rs::rocrand::PseudoRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct ModuleCache {
    modules: HashMap<String, Arc<Module>>,
}

impl ModuleCache {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    id: DeviceId,
    device: Arc<HipDevice>,
    pub(crate) stream: Arc<Stream>,
    modules: Arc<Mutex<ModuleCache>>,
    rocrand: Arc<Mutex<PseudoRng>>,
    seed_value: Arc<RwLock<u64>>,
    pub(crate) blas: Arc<rocblas::Handle>,
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

        let mut rocrand = PseudoRng::new(rocm_rs::rocrand::rng_type::PSEUDO_DEFAULT)
            .map_err(|e| crate::Error::Msg(format!("Failed to create rocrand generator: {}", e)))?;
        let seed = 299792458u64;
        rocrand
            .set_seed(seed)
            .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;

        let blas = rocblas::Handle::new().map_err(|e| RocmError::Rocblas(e.to_string()))?;
        blas.set_stream(&stream)
            .map_err(|e| RocmError::Rocblas(e.to_string()))?;

        Ok(Self {
            id: DeviceId::new(),
            device: Arc::new(device),
            stream: Arc::new(stream),
            modules: Arc::new(Mutex::new(ModuleCache::new())),
            rocrand: Arc::new(Mutex::new(rocrand)),
            seed_value: Arc::new(RwLock::new(seed)),
            blas: Arc::new(blas),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn alloc<T>(&self, len: usize) -> Result<DeviceMemory<T>> {
        DeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))
    }

    pub fn alloc_zeros<T: Default + Clone>(&self, len: usize) -> Result<DeviceMemory<T>> {
        let mut mem = DeviceMemory::new(len)
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        mem.memset(0)
            .map_err(|e| crate::Error::Msg(format!("Failed to memset: {}", e)))?;
        Ok(mem)
    }

    pub fn clone_htod<T: Clone>(&self, src: &[T]) -> Result<DeviceMemory<T>> {
        let count = src.len();
        let mut dst = DeviceMemory::new(count)?;
        dst.copy_from_host(src)?;
        Ok(dst)
    }

    pub fn clone_dtoh<T: Default + Clone>(&self, src: &DeviceMemory<T>) -> Result<Vec<T>> {
        let count = src.count();
        let mut dst: Vec<T> = vec![T::default(); count];
        src.copy_to_host(&mut dst)?;
        Ok(dst)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| crate::Error::Msg(format!("Synchronize failed: {}", e)))
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

macro_rules! dispatch_cpu_storage {
    ($storage:expr, $self:expr, |$data:ident, $variant:ident| $body:expr) => {
        match $storage {
            CpuStorage::U8($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::U8(mem);
                $body
            }
            CpuStorage::U32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::U32(mem);
                $body
            }
            CpuStorage::I16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I16(mem);
                $body
            }
            CpuStorage::I32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I32(mem);
                $body
            }
            CpuStorage::I64($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I64(mem);
                $body
            }
            CpuStorage::BF16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::BF16(mem);
                $body
            }
            CpuStorage::F16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F16(mem);
                $body
            }
            CpuStorage::F32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F32(mem);
                $body
            }
            CpuStorage::F64($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F64(mem);
                $body
            }
            _ => {
                return Err(crate::Error::Msg(format!(
                    "CpuStorage variant not yet supported for ROCm"
                )));
            }
        }
    };
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
        let mem = self.clone_htod(data)?;
        let slice = match T::DTYPE {
            DType::U8 => RocmStorageSlice::U8(unsafe { std::mem::transmute(mem) }),
            DType::U32 => RocmStorageSlice::U32(unsafe { std::mem::transmute(mem) }),
            DType::I16 => RocmStorageSlice::I16(unsafe { std::mem::transmute(mem) }),
            DType::I32 => RocmStorageSlice::I32(unsafe { std::mem::transmute(mem) }),
            DType::I64 => RocmStorageSlice::I64(unsafe { std::mem::transmute(mem) }),
            DType::BF16 => RocmStorageSlice::BF16(unsafe { std::mem::transmute(mem) }),
            DType::F16 => RocmStorageSlice::F16(unsafe { std::mem::transmute(mem) }),
            DType::F32 => RocmStorageSlice::F32(unsafe { std::mem::transmute(mem) }),
            DType::F64 => RocmStorageSlice::F64(unsafe { std::mem::transmute(mem) }),
            DType::F8E4M3 => RocmStorageSlice::F8E4M3(unsafe { std::mem::transmute(mem) }),
            dtype => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm storage_from_slice",
                    dtype
                )));
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        dispatch_cpu_storage!(storage, self, |data, slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _hi: f64,
    ) -> Result<Self::Storage> {
        Err(crate::Error::Msg(
            "Random generation not yet implemented for ROCm".to_string(),
        ))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut rocrand = self.rocrand.lock().unwrap();
        // rocrand can only generate an even number of normal values.
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16
            | DType::F8E4M3
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "dtype {:?} not supported for rocm rand_normal",
                    dtype
                )));
            }
            DType::F32 => {
                let mut data = self.alloc::<f32>(elem_count_round)?;
                rocrand
                    .generate_normal(&mut data, mean as f32, std as f32)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal failed: {}", e))
                    })?;
                RocmStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = self.alloc::<f64>(elem_count_round)?;
                rocrand
                    .generate_normal_double(&mut data, mean, std)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal_double failed: {}", e))
                    })?;
                RocmStorageSlice::F64(data)
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut rocrand = self.rocrand.lock().unwrap();
        rocrand
            .set_seed(seed)
            .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
