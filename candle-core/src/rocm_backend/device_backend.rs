//! `BackendDevice` for [`RocmDevice`].
//!
//! Split out of `device` so that file stays with the device's own plumbing —
//! stream, allocator, kernel cache — while the dtype dispatch that the trait
//! needs lives next to the trait impl.

use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape};
use half::{bf16, f16};

use super::{rocm_error, RocmDevice, RocmStorage, RocmStorageSlice};

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
                let $slice =
                    RocmStorageSlice::F8E4M3($self.$method::<float8::F8E4M3>($elem_count)?);
                $body
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(rocm_error(format!(
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
            CpuStorageRef::F8E4M3(data) => RocmStorageSlice::F8E4M3(self.clone_htod(data)?),
            CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F4(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(rocm_error(
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
            .map_err(|e| rocm_error(format!("Failed to set rocrand seed: {}", e)))?;
        *self
            .seed_value
            .write()
            .map_err(|_| rocm_error("Failed to lock ROCm seed value".to_string()))? = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let seed = self
            .seed_value
            .read()
            .map_err(|_| rocm_error("Failed to lock ROCm seed value".to_string()))?;
        Ok(*seed)
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
