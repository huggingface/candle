use super::{GgmlDType, QStorage, QuantizedType};
use crate::{DType, MetalDevice, MetalStorage, Result};
use metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: GgmlDType) -> Self {
        Self {
            device,
            buffer,
            dtype,
        }
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        let buffer = self
            .device
            .new_buffer(elem_count, DType::F32, "dequantize")?;
        // TODO Implement dequantize
        // todo!("To float metal");
        dbg!("TODO dequantize");
        Ok(MetalStorage::new(buffer, self.device.clone(), DType::F32))
    }

    pub fn quantize(&self, src: &MetalStorage) -> Result<()> {
        // let buffer = self
        //     .device
        //     .new_buffer(elem_count, DType::F32, "quantize")?;
        // TODO Implement dequantize
        // todo!("To float metal");
        dbg!("TODO quantize");
        Ok(())
        // Ok(MetalStorage::new(buffer, self.device.clone(), DType::F32))
    }
}

pub fn load_quantized_metal<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    dbg!("TODO load");
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}
