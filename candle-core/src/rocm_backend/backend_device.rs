//! BackendDevice trait implementation for RocmDevice
//! Created by: TEAM-509 (Wiring up rand_uniform, rand_normal, zeros_impl)
//! CUDA parity verified by: TEAM-509

use super::{RocmDevice, RocmError, RocmStorage, RocmStorageSlice as S};
use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape, WithDType};

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs:262-279
    fn new(ordinal: usize) -> Result<Self> {
        RocmDevice::new(ordinal)
    }

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs location()
    fn location(&self) -> DeviceLocation {
        DeviceLocation::Rocm {
            gpu_id: self.id(),
        }
    }

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs same_device()
    fn same_device(&self, rhs: &Self) -> bool {
        self == rhs
    }

    // TEAM-509: Implement zeros_impl (CUDA parity)
    // Matches cuda_backend/device.rs:299-320
    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                S::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                S::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                S::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<half::bf16>(elem_count)?;
                S::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<half::f16>(elem_count)?;
                S::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                S::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                S::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<float8::F8E4M3>(elem_count)?;
                S::F8E4M3(data)
            }
        };
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement alloc_uninit (CUDA parity)
    // Matches cuda_backend/device.rs alloc_uninit pattern
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => S::U8(self.alloc::<u8>(elem_count)?),
            DType::U32 => S::U32(self.alloc::<u32>(elem_count)?),
            DType::I64 => S::I64(self.alloc::<i64>(elem_count)?),
            DType::BF16 => S::BF16(self.alloc::<half::bf16>(elem_count)?),
            DType::F16 => S::F16(self.alloc::<half::f16>(elem_count)?),
            DType::F32 => S::F32(self.alloc::<f32>(elem_count)?),
            DType::F64 => S::F64(self.alloc::<f64>(elem_count)?),
            DType::F8E4M3 => S::F8E4M3(self.alloc::<float8::F8E4M3>(elem_count)?),
        };
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement storage_from_slice (CUDA parity)
    // Matches cuda_backend/device.rs storage_from_slice pattern
    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let dev_mem = self.memcpy_stod(data)?;
        let slice = T::to_rocm_storage(dev_mem);
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement storage_from_cpu_storage (CUDA parity)
    // Matches cuda_backend/device.rs storage_from_cpu_storage pattern
    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let slice = match storage {
            CpuStorage::U8(data) => S::U8(self.memcpy_stod(data)?),
            CpuStorage::U32(data) => S::U32(self.memcpy_stod(data)?),
            CpuStorage::I64(data) => S::I64(self.memcpy_stod(data)?),
            CpuStorage::BF16(data) => S::BF16(self.memcpy_stod(data)?),
            CpuStorage::F16(data) => S::F16(self.memcpy_stod(data)?),
            CpuStorage::F32(data) => S::F32(self.memcpy_stod(data)?),
            CpuStorage::F64(data) => S::F64(self.memcpy_stod(data)?),
            CpuStorage::F8E4M3(data) => S::F8E4M3(self.memcpy_stod(data)?),
        };
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement storage_from_cpu_storage_owned (CUDA parity)
    // Matches cuda_backend/device.rs storage_from_cpu_storage_owned pattern
    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    // TEAM-509: Implement rand_uniform (CUDA parity)
    // Matches cuda_backend/device.rs:341-376
    // Fixed by: TEAM-509 (range scaling with Affine operation)
    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut rng = self.rocrand().lock().unwrap();
        
        let slice = match dtype {
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                rng.0.generate_uniform(&mut data)
                    .map_err(|e| RocmError::InternalError(&format!("rocRAND uniform failed: {:?}", e)))?;
                S::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                rng.0.generate_uniform_double(&mut data)
                    .map_err(|e| RocmError::InternalError(&format!("rocRAND uniform failed: {:?}", e)))?;
                S::F64(data)
            }
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 | DType::F8E4M3 => {
                return Err(RocmError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                }
                .into());
            }
        };
        
        // Scale from [0, 1) to [lo, up) using Affine operation
        // Matches cuda_backend/device.rs:365-371
        let slice = if lo == 0.0 && up == 1.0 {
            slice
        } else {
            use super::ops::Affine;
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement rand_normal (CUDA parity)
    // Matches cuda_backend/device.rs:378-416
    // Fixed by: TEAM-509 (odd element count handling)
    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut rng = self.rocrand().lock().unwrap();
        
        // rocRAND (like cuRAND) can only generate an even number of values for normal distribution
        // See: https://github.com/huggingface/candle/issues/734
        // Round up to even count, then we'll only use elem_count elements
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        
        let slice = match dtype {
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                rng.0.generate_normal(&mut data, mean as f32, std as f32)
                    .map_err(|e| RocmError::InternalError(&format!("rocRAND normal failed: {:?}", e)))?;
                S::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                rng.0.generate_normal_double(&mut data, mean, std)
                    .map_err(|e| RocmError::InternalError(&format!("rocRAND normal failed: {:?}", e)))?;
                S::F64(data)
            }
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 | DType::F8E4M3 => {
                return Err(RocmError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                }
                .into());
            }
        };
        
        Ok(RocmStorage::new(slice, self.clone()))
    }

    // TEAM-509: Implement set_seed (CUDA parity)
    // Matches cuda_backend/device.rs set_seed behavior
    fn set_seed(&self, seed: u64) -> Result<()> {
        self.set_seed(seed)
    }

    // Created by: TEAM-488 | CUDA parity: cuda_backend/device.rs synchronize()
    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}

// TEAM-509: Helper trait to convert WithDType to RocmStorageSlice
trait ToRocmStorage: WithDType {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S;
}

impl ToRocmStorage for u8 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::U8(data)
    }
}

impl ToRocmStorage for u32 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::U32(data)
    }
}

impl ToRocmStorage for i64 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::I64(data)
    }
}

impl ToRocmStorage for half::bf16 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::BF16(data)
    }
}

impl ToRocmStorage for half::f16 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::F16(data)
    }
}

impl ToRocmStorage for f32 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::F32(data)
    }
}

impl ToRocmStorage for f64 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::F64(data)
    }
}

impl ToRocmStorage for float8::F8E4M3 {
    fn to_rocm_storage(data: rocm_rs::hip::DeviceMemory<Self>) -> S {
        S::F8E4M3(data)
    }
}
