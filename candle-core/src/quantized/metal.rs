use super::{GgmlDType, QuantizedType};
use crate::{MetalDevice, Result};

pub struct QStorage;

impl QuantizedType for QStorage {
    fn dtype(&self) -> GgmlDType {
        todo!();
    }

    fn matmul_t(&self, _mkn: (usize, usize, usize), _lhs: &[f32], _dst: &mut [f32]) -> Result<()> {
        todo!();
    }

    fn to_float(&self, _ys: &mut [f32]) -> Result<()> {
        todo!();
    }

    fn storage_size_in_bytes(&self) -> usize {
        todo!();
    }

    fn as_ptr(&self) -> *const u8 {
        todo!();
    }

    fn block_size(&self) -> usize {
        todo!();
    }
}

pub fn load_quantized_metal<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<Box<QStorage>> {
    todo!("Implement the load");
}
