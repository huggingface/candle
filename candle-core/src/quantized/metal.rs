use super::{GgmlDType, QStorage, QuantizedType};
use crate::{MetalDevice, Result};
use metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    pub fn new(buffer: Arc<Buffer>, dtype: GgmlDType) -> Self {
        Self { buffer, dtype }
    }
}

// pub fn load_quantized_metal<T: super::GgmlType + Send + Sync + 'static>(
//     device: &MetalDevice,
//     data: &[T],
// ) -> Result<QStorage> {
//     let buffer = device.new_buffer_with_data(data)?;
//     Ok(QStorage::Metal(QMetalStorage {
//         dtype: T::DTYPE,
//         block_size: T::BLCK_SIZE,
//         buffer,
//     }))
// }
