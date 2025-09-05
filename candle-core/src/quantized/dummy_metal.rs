#[cfg(not(feature = "metal"))]
mod dummy {
    use crate::{
        quantized::{GgmlDType, GgmlType, QuantizedBackend, QuantizedDevice},
        Error, MetalDevice, MetalStorage, Result,
    };

    #[derive(Debug)]
    pub struct QMetalStorage {
        dtype: GgmlDType,
        device: MetalDevice,
    }

    impl QuantizedDevice<QMetalStorage> for MetalDevice {
        type Storage = MetalStorage;

        fn qzeros(&self, _elem_count: usize, _dtype: GgmlDType) -> Result<QMetalStorage> {
            Err(Error::NotCompiledWithCudaSupport)
        }

        fn load_quantized<T: GgmlType + Send + Sync + 'static>(
            self: &Self,
            _data: &[T],
        ) -> Result<QMetalStorage> {
            Err(Error::NotCompiledWithCudaSupport)
        }
    }

    impl QuantizedBackend for QMetalStorage {
        type Storage = MetalStorage;
        type Device = MetalDevice;

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
            Err(Error::NotCompiledWithMetalSupport)
        }

        fn dequantize(&self, _elem_count: usize) -> Result<Self::Storage> {
            Err(Error::NotCompiledWithMetalSupport)
        }

        fn data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
            crate::bail!("not implemented");
        }

        fn device(&self) -> impl AsRef<Self::Device> {
            &self.device
        }
    }

    impl QMetalStorage {
        pub fn fwd(
            &self,
            _self_shape: &crate::Shape,
            _storage: &MetalStorage,
            _layout: &crate::Layout,
        ) -> Result<(MetalStorage, crate::Shape)> {
            Err(Error::NotCompiledWithMetalSupport)
        }
    }
}

#[cfg(not(feature = "metal"))]
pub use dummy::*;
