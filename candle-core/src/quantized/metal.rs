use super::{GgmlDType, QStorage};
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

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: GgmlDType) -> Self {
        Self {
            device,
            buffer,
            dtype,
        }
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        let buffer = self.device.new_buffer_managed(self.buffer.length())?;
        let command_buffer = self.device.command_buffer()?;
        command_buffer.set_label("to_cpu");
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                f32::to_float(&vec, &mut out)?;
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                half::f16::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, elem_count);
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ2K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ3K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ4K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ5K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ6K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> =
                    read_to_vec(&buffer, elem_count / self.dtype.block_size());
                use crate::quantized::k_quants::GgmlType;
                crate::quantized::BlockQ8K::to_float(&vec, &mut out)?;
            }
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(buffer, self.device.clone(), DType::F32))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }
}

pub fn load_quantized_metal<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}
