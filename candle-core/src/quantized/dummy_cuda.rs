#[cfg(not(feature = "cuda"))]
mod dummy {
    use crate::quantized::GgmlDType;
    use crate::{
        quantized::{GgmlType, QuantizedBackend, QuantizedDevice},
        CudaDevice, CudaStorage, Error, Result,
    };

    #[derive(Debug, Clone)]
    pub struct QCudaStorage {
        dtype: GgmlDType,
        device: CudaDevice,
    }

    impl QuantizedDevice<QCudaStorage> for CudaDevice {
        type Storage = CudaStorage;

        fn qzeros(&self, _elem_count: usize, _dtype: GgmlDType) -> Result<QCudaStorage> {
            Err(Error::NotCompiledWithCudaSupport)
        }

        fn load_quantized<T: GgmlType + Send + Sync + 'static>(
            self: &Self,
            _data: &[T],
        ) -> Result<QCudaStorage> {
            Err(Error::NotCompiledWithCudaSupport)
        }
    }

    impl QuantizedBackend for QCudaStorage {
        type Storage = CudaStorage;
        type Device = CudaDevice;

        fn block_size(&self) -> usize {
            0
        }

        fn dtype(&self) -> GgmlDType {
            self.dtype
        }

        fn storage_size_in_bytes(&self) -> usize {
            0
        }

        fn quantize(&mut self, _src: &Self::Storage) -> Result<()> {
            Err(Error::NotCompiledWithCudaSupport)
        }

        fn dequantize(&self, _elem_count: usize) -> Result<Self::Storage> {
            Err(Error::NotCompiledWithCudaSupport)
        }

        fn data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
            crate::bail!("not implemented");
        }

        fn device(&self) -> impl AsRef<Self::Device> {
            &self.device
        }
    }

    impl QCudaStorage {
        pub fn fwd(
            &self,
            _self_shape: &crate::Shape,
            _storage: &CudaStorage,
            _layout: &crate::Layout,
        ) -> Result<(CudaStorage, crate::Shape)> {
            Err(Error::NotCompiledWithCudaSupport)
        }

        pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
            Err(Error::NotCompiledWithCudaSupport)
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use dummy::*;
