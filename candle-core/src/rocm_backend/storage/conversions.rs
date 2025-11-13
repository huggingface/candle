//! Type conversion operations for RocmStorage

use crate::rocm_backend::{kernels, RocmError};
use crate::Result;
use super::{RocmStorage, RocmStorageSlice};

type S = RocmStorageSlice;

impl RocmStorage {
    pub(super) fn to_cpu_storage_impl(&self) -> Result<crate::CpuStorage> {
        let cpu_storage = match &self.slice {
            S::U8(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U8(data)
            }
            S::U32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U32(data)
            }
            S::I64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::I64(data)
            }
            S::BF16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::BF16(data)
            }
            S::F16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F16(data)
            }
            S::F32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F32(data)
            }
            S::F64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F64(data)
            }
            S::F8E4M3(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F8E4M3(data)
            }
        };
        Ok(cpu_storage)
    }

    pub(super) fn to_dtype_impl(&self, layout: &crate::Layout, dtype: crate::DType) -> Result<Self> {
        use crate::DType;
        let dev = self.device.hip_device();
        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());

        let slice = match (&self.slice, dtype) {
            (S::U8(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::U32(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U32(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::I64(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::I64(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::BF16(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::BF16(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::F16(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F16(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::F32(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F32(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::F64(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F64(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),

            (S::F8E4M3(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::U32) => S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::I64) => S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::BF16) => S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F16) => S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F32) => S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F64) => S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::F8E4M3(src), DType::F8E4M3) => S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?),
        };

        Ok(Self { slice, device: self.device.clone() })
    }
}
