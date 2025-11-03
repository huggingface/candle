/// Core trait for quantized data types
///
/// Provides metadata and storage calculations for quantized tensor formats.
/// Types implementing this trait can be registered with `register_quantized_types!`
/// for automatic dispatch and backend detection.
///
/// # Implementation Requirements
///
/// - Must be `Send + Sync` for thread-safe usage
/// - Should typically be zero-sized types (ZSTs) used as type-level markers
/// - Must implement `Default` for the dispatch system
///
/// # Example
///
/// ```rust
/// use candle_macros_types::QuantizedType;
///
/// #[derive(Default)]
/// pub struct GgmlQ4_0;
///
/// impl QuantizedType for GgmlQ4_0 {
///     const NAME: &'static str = "q4_0";
///     const SIZE_IN_BYTES: usize = 18; // 16 values + 2 bytes metadata per block
///
///     fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
///         // Q4_0 stores 32 f32 values per 18-byte block
///         ((num_elements + 31) / 32) * 18
///     }
///
///     fn infer_element_count(&self, data_len: usize) -> usize {
///         // Each 18-byte block contains 32 f32 values
///         (data_len / 18) * 32
///     }
/// }
/// ```
pub trait QuantizedType: Send + Sync {
    /// Unique identifier for this quantized type (e.g., "q4_0", "q8_0")
    const NAME: &'static str;

    /// Size in bytes of the basic storage unit for this format
    ///
    /// For block-based formats, this is the block size.
    /// For element-based formats, this is the size per element.
    const SIZE_IN_BYTES: usize;

    /// Calculate storage size in bytes for a given number of f32 elements
    ///
    /// # Arguments
    /// * `num_elements` - Number of f32 elements to quantize
    ///
    /// # Returns
    /// Number of bytes required to store the quantized representation
    fn storage_size_in_bytes(&self, num_elements: usize) -> usize;

    /// Infer the number of f32 elements from quantized data size
    ///
    /// # Arguments
    /// * `data_len` - Length of quantized data in bytes
    ///
    /// # Returns
    /// Number of f32 elements represented by the quantized data
    fn infer_element_count(&self, data_len: usize) -> usize;
}

/// CPU operations for quantized types (optional)
///
/// Implement this trait to enable CPU-based quantization, dequantization, and
/// matrix multiplication operations. Types implementing this trait are automatically
/// detected at compile time using stable Rust techniques.
///
/// **Note**: This trait is optional. Only implement it if your type has CPU support.
/// The dispatch system will automatically handle types that don't implement this trait
/// by returning appropriate errors at runtime.
///
/// # Trait Object Safety
///
/// This trait is dyn-compatible (object-safe). All methods take `&self` to allow
/// usage as `Box<dyn QuantizedCpuOps>`, even though most implementations are zero-sized types.
///
/// # Example
///
/// ```rust
/// use candle_macros_types::{QuantizedType, QuantizedCpuOps};
///
/// #[derive(Default)]
/// pub struct GgmlQ4_0;
///
/// impl QuantizedType for GgmlQ4_0 {
///     const NAME: &'static str = "q4_0";
///     const SIZE_IN_BYTES: usize = 18;
///
///     fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
///         ((num_elements + 31) / 32) * 18
///     }
///
///     fn infer_element_count(&self, data_len: usize) -> usize {
///         (data_len / 18) * 32
///     }
/// }
///
/// impl QuantizedCpuOps for GgmlQ4_0 {
///     fn quantize(&self, input: &[f32]) -> Result<Vec<u8>, String> {
///         // Implement quantization: convert f32 slice to q4_0 format
///         // ... quantization logic ...
///         Ok(vec![])
///     }
///
///     fn dequantize(&self, data: &[u8], output: &mut [f32]) -> Result<(), String> {
///         // Implement dequantization: convert q4_0 data back to f32
///         // ... dequantization logic ...
///         Ok(())
///     }
///
///     fn matmul(
///         &self,
///         lhs_f32: &[f32],
///         lhs_shape: &[usize],
///         rhs_data: &[u8],
///         rhs_shape: &[usize],
///     ) -> Result<Vec<f32>, String> {
///         // Implement optimized matmul: f32 × quantized → f32
///         // ... matmul logic ...
///         Ok(vec![])
///     }
/// }
/// ```
pub trait QuantizedCpuOps: QuantizedType {
    /// Convert f32 values to this quantized format
    ///
    /// # Arguments
    /// * `input` - Input f32 values to quantize
    ///
    /// # Returns
    /// Quantized data as a byte vector
    ///
    /// # Errors
    /// Returns error if quantization fails
    fn quantize(&self, input: &[f32]) -> Result<Vec<u8>, String>;

    /// Convert quantized data back to f32 values
    ///
    /// # Arguments
    /// * `data` - Quantized data in this format's byte representation
    /// * `output` - Pre-allocated output buffer for f32 values
    ///
    /// # Errors
    /// Returns error if data is invalid or output buffer size is incorrect
    fn dequantize(&self, data: &[u8], output: &mut [f32]) -> Result<(), String>;

    /// Optimized matrix multiplication: f32 × quantized → f32
    ///
    /// Computes `lhs_f32 @ rhs_quantized` where the right-hand side remains
    /// in quantized format for memory efficiency.
    ///
    /// # Arguments
    /// * `lhs_f32` - Left matrix in f32, row-major layout
    /// * `lhs_shape` - Shape `[M, K]` of left matrix
    /// * `rhs_data` - Right matrix in quantized format, column-major or as required by impl
    /// * `rhs_shape` - Shape `[K, N]` of right matrix
    ///
    /// # Returns
    /// Result matrix in f32 format with shape `[M, N]`, row-major layout
    ///
    /// # Errors
    /// Returns error if shapes are incompatible or computation fails
    fn matmul(
        &self,
        lhs_f32: &[f32],
        lhs_shape: &[usize],
        rhs_data: &[u8],
        rhs_shape: &[usize],
    ) -> Result<Vec<f32>, String>;
}

/// CUDA GPU operations for quantized types (optional)
///
/// Implement this trait to enable CUDA-accelerated quantization operations on NVIDIA GPUs.
/// Requires the `cuda` feature to be enabled.
///
/// **Note**: This trait is optional. Only implement it if your type has CUDA support.
/// Default implementations return errors, allowing graceful handling of unsupported operations.
///
/// # Feature Gate
///
/// Only available with `features = ["cuda"]`
///
/// # Trait Object Safety
///
/// This trait is dyn-compatible (object-safe), allowing usage as `Box<dyn QuantizedCudaOps>`.
#[cfg(feature = "cuda")]
pub trait QuantizedCudaOps: QuantizedType {
    /// Convert f32 values to quantized format on GPU
    ///
    /// # Arguments
    /// * `input` - Input f32 slice on GPU device memory
    /// * `device` - Reference to the CUDA device for kernel launches
    ///
    /// # Returns
    /// Quantized data as GPU device memory
    ///
    /// # Errors
    /// Returns error if CUDA operation fails
    fn quantize_cuda<D: CudaStorageDevice>(
        &self,
        _input: &cudarc::driver::CudaSlice<f32>,
        _device: &D,
    ) -> Result<cudarc::driver::CudaSlice<u8>, String> {
        Err("CUDA quantize not implemented for this type".to_string())
    }

    /// Convert quantized data back to f32 values on GPU
    ///
    /// # Arguments
    /// * `data` - Quantized data on GPU device memory
    /// * `output` - Pre-allocated output buffer on GPU device memory
    /// * `device` - Reference to the CUDA device for kernel launches
    ///
    /// # Errors
    /// Returns error if data is invalid, buffer size incorrect, or CUDA operation fails
    fn dequantize_cuda<D: CudaStorageDevice>(
        &self,
        _data: &cudarc::driver::CudaSlice<u8>,
        _output: &mut cudarc::driver::CudaSlice<f32>,
        _device: &D,
    ) -> Result<(), String> {
        Err("CUDA dequantize not implemented for this type".to_string())
    }

    /// GPU-accelerated matrix multiplication: f32 × quantized → f32
    ///
    /// # Arguments
    /// * `lhs_f32` - Left matrix on GPU in f32 format, shape `[M, K]`
    /// * `lhs_shape` - Shape `[M, K]` of left matrix
    /// * `rhs_data` - Right matrix on GPU in quantized format, shape `[K, N]`
    /// * `rhs_shape` - Shape `[K, N]` of right matrix
    /// * `device` - Reference to the CUDA device for kernel launches
    ///
    /// # Returns
    /// Result matrix on GPU in f32 format with shape `[M, N]`
    ///
    /// # Errors
    /// Returns error if shapes are incompatible or CUDA operation fails
    fn matmul_cuda<D: CudaStorageDevice>(
        &self,
        _lhs_f32: &cudarc::driver::CudaSlice<f32>,
        _lhs_shape: &[usize],
        _rhs_data: &cudarc::driver::CudaSlice<u8>,
        _rhs_shape: &[usize],
        _device: &D,
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        Err("CUDA matmul not implemented for this type".to_string())
    }
}

/// Trait for CUDA device types that can allocate GPU memory
///
/// Simplified CudeDevice interface for moving data to/from GPU in quantized operations.
///
/// # Feature Gate
///
/// Only available with `features = ["cuda"]`
#[cfg(feature = "cuda")]
pub trait CudaStorageDevice {
    /// Allocate a zero-initialized buffer on the GPU
    ///
    /// # Arguments
    /// * `len` - Number of elements to allocate
    ///
    /// # Returns
    /// A `CudaSlice` containing `len` zero-initialized elements
    ///
    /// # Errors
    /// Returns error if GPU allocation fails
    fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>, Box<dyn std::error::Error + Send + Sync>>;

    /// Downcast to Any for type-specific operations
    ///
    /// This allows implementations to be downcasted to their concrete types
    /// when specific functionality is needed (e.g., CudaDevice for kernel launches)
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Metal GPU operations for quantized types (optional)
///
/// Implement this trait to enable Metal-accelerated quantization operations on Apple Silicon GPUs.
/// Requires the `metal` feature to be enabled.
///
/// **Note**: This trait is optional. Only implement it if your type has Metal support.
/// Default implementations return errors, allowing graceful handling of unsupported operations.
///
/// # Feature Gate
///
/// Only available with `features = ["metal"]`
///
/// # Trait Object Safety
///
/// This trait is dyn-compatible (object-safe), allowing usage as `Box<dyn QuantizedMetalOps>`.
#[cfg(feature = "metal")]
pub trait QuantizedMetalOps: QuantizedType {
    /// Convert quantized data back to f32 values on GPU using Metal
    ///
    /// # Arguments
    /// * `data` - Quantized data in GPU buffer
    /// * `output` - Pre-allocated output buffer on GPU
    ///
    /// # Errors
    /// Returns error if data is invalid, buffer size incorrect, or Metal operation fails
    fn dequantize_metal(
        &self,
        _data: &metal::Buffer,
        _output: &mut metal::Buffer,
    ) -> Result<(), String> {
        Err("Metal dequantize not implemented for this type".to_string())
    }

    /// GPU-accelerated matrix multiplication using Metal: f32 × quantized → f32
    ///
    /// # Arguments
    /// * `lhs_f32` - Left matrix on GPU in f32 format, shape `[M, K]`
    /// * `lhs_shape` - Shape `[M, K]` of left matrix
    /// * `rhs_data` - Right matrix on GPU in quantized format, shape `[K, N]`
    /// * `rhs_shape` - Shape `[K, N]` of right matrix
    ///
    /// # Returns
    /// Result matrix on GPU in f32 format with shape `[M, N]`
    ///
    /// # Errors
    /// Returns error if shapes are incompatible or Metal operation fails
    fn matmul_metal(
        &self,
        _lhs_f32: &metal::Buffer,
        _lhs_shape: &[usize],
        _rhs_data: &metal::Buffer,
        _rhs_shape: &[usize],
    ) -> Result<metal::Buffer, String> {
        Err("Metal matmul not implemented for this type".to_string())
    }
}
