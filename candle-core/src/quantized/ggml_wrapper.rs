//! Wrappers for GGML quantization types
//! These wrappers allow each GGML type to be registered as a separate
//! QuantizedDType variant, enabling users to specify exact quantization types.

use crate::quantized::k_quants::GgmlType;
use crate::Result;

// Helper trait to associate wrapper types with their block types
// This enables zero-cost abstraction - all methods are inlined and monomorphized
trait GgmlTypeWrapper {
    type Block: GgmlType;
    const BLOCK_SIZE: usize;
    const TYPE_SIZE: usize;
}

/// Shared implementation for all GGML wrapper types
/// This provides common dequantize/quantize/matmul logic to avoid code duplication
mod shared_impl {
    use super::*;

    /// Dequantize data from compressed format to f32
    ///
    /// This is a generic implementation that works for all GGML block types.
    /// Thanks to monomorphization and inlining, this has zero overhead compared to
    /// type-specific implementations.
    ///
    /// # Safety
    /// Assumes `data` contains valid quantized blocks. The caller must ensure:
    /// - `data.len()` is a multiple of `TYPE_SIZE`
    /// - Data represents valid quantized blocks (validated by `to_float`)
    #[inline]
    pub fn dequantize<W: GgmlTypeWrapper>(data: &[u8], output: &mut [f32]) -> Result<()> {
        let block_count = data.len() / W::TYPE_SIZE;

        // SAFETY:
        // - Caller ensures data.len() is correct for the quantized format
        // - W::Block is repr(C) with well-defined layout
        // - to_float validates block contents and handles any invalid data
        let blocks =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const W::Block, block_count) };
        W::Block::to_float(blocks, output);
        Ok(())
    }

    /// Quantize f32 data to compressed format
    ///
    /// Generic implementation that works for all GGML block types with zero overhead.
    #[inline]
    pub fn quantize<W: GgmlTypeWrapper>(input: &[f32]) -> Result<Vec<u8>> {
        let block_count = input.len() / W::BLOCK_SIZE;
        let mut blocks = vec![W::Block::zeros(); block_count];
        W::Block::from_float(input, &mut blocks);

        // Convert blocks to bytes
        let byte_len = blocks.len() * W::TYPE_SIZE;
        let data = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
        Ok(data.to_vec())
    }

    /// Calculate storage size in bytes for given number of elements
    #[inline]
    pub fn storage_size_in_bytes<W: GgmlTypeWrapper>(num_elements: usize) -> usize {
        let block_count = num_elements.div_ceil(W::BLOCK_SIZE);
        block_count * W::TYPE_SIZE
    }

    /// Infer number of f32 elements from byte length of quantized data
    #[inline]
    pub fn infer_element_count<W: GgmlTypeWrapper>(data_len: usize) -> usize {
        debug_assert!(
            data_len.is_multiple_of(W::TYPE_SIZE),
            "data length must be multiple of TYPE_SIZE"
        );
        let num_blocks = data_len / W::TYPE_SIZE;
        num_blocks * W::BLOCK_SIZE
    }

    /// Perform matrix multiplication with quantized right-hand side
    ///
    /// Generic implementation for f32 × quantized → f32 matmul.
    ///
    /// # Safety
    /// Assumes `rhs_data` contains valid quantized blocks. The caller must ensure:
    /// - `rhs_data.len()` is a multiple of `TYPE_SIZE`
    /// - Data represents valid quantized blocks
    #[inline]
    pub fn matmul<W: GgmlTypeWrapper>(
        lhs_f32: &[f32],
        lhs_shape: &[usize],
        rhs_data: &[u8],
        rhs_shape: &[usize],
    ) -> Result<Vec<f32>> {
        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            crate::bail!("matmul requires at least 2D tensors");
        }

        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let n = rhs_shape[rhs_shape.len() - 1];

        if rhs_shape[rhs_shape.len() - 2] != k {
            crate::bail!("incompatible matmul dimensions");
        }

        let block_count = rhs_data.len() / W::TYPE_SIZE;

        // SAFETY:
        // - Caller ensures rhs_data.len() is correct for the quantized format
        // - W::Block is repr(C) with well-defined layout
        // - k_quants::matmul validates block contents
        let rhs_blocks = unsafe {
            std::slice::from_raw_parts(rhs_data.as_ptr() as *const W::Block, block_count)
        };

        let mut result = vec![0.0f32; m * n];
        crate::quantized::k_quants::matmul((m, k, n), lhs_f32, rhs_blocks, &mut result)?;
        Ok(result)
    }
}

macro_rules! define_ggml_wrapper {
    ($name:ident, $block_type:ident, $block_size:expr, $type_size:expr) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl GgmlTypeWrapper for $name {
            type Block = crate::quantized::k_quants::$block_type;
            const BLOCK_SIZE: usize = $block_size;
            const TYPE_SIZE: usize = $type_size;
        }

        // Implement QuantizedType trait from candle-macros-types
        impl candle_macros_types::QuantizedType for $name {
            const NAME: &'static str = stringify!($name);
            const SIZE_IN_BYTES: usize = $type_size;

            #[inline]
            fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
                shared_impl::storage_size_in_bytes::<Self>(num_elements)
            }

            #[inline]
            fn infer_element_count(&self, data_len: usize) -> usize {
                shared_impl::infer_element_count::<Self>(data_len)
            }
        }

        // Implement QuantizedCpuOps trait
        impl candle_macros_types::QuantizedCpuOps for $name {
            #[inline]
            fn quantize(&self, input: &[f32]) -> std::result::Result<Vec<u8>, String> {
                shared_impl::quantize::<Self>(input).map_err(|e| format!("{}", e))
            }

            #[inline]
            fn dequantize(
                &self,
                data: &[u8],
                output: &mut [f32],
            ) -> std::result::Result<(), String> {
                shared_impl::dequantize::<Self>(data, output).map_err(|e| format!("{}", e))
            }

            #[inline]
            fn matmul(
                &self,
                lhs_f32: &[f32],
                lhs_shape: &[usize],
                rhs_data: &[u8],
                rhs_shape: &[usize],
            ) -> std::result::Result<Vec<f32>, String> {
                shared_impl::matmul::<Self>(lhs_f32, lhs_shape, rhs_data, rhs_shape)
                    .map_err(|e| format!("{}", e))
            }
        }
    };
}

// Define all GGML quantization type wrappers with their block sizes and type sizes
define_ggml_wrapper!(GgmlQ4_0, BlockQ4_0, 32, 18); // QK4_0=32, sizeof(BlockQ4_0)=18
define_ggml_wrapper!(GgmlQ4_1, BlockQ4_1, 32, 20); // QK4_1=32, sizeof(BlockQ4_1)=20
define_ggml_wrapper!(GgmlQ5_0, BlockQ5_0, 32, 22); // QK5_0=32, sizeof(BlockQ5_0)=22
define_ggml_wrapper!(GgmlQ5_1, BlockQ5_1, 32, 24); // QK5_1=32, sizeof(BlockQ5_1)=24
define_ggml_wrapper!(GgmlQ8_0, BlockQ8_0, 32, 34); // QK8_0=32, sizeof(BlockQ8_0)=34
define_ggml_wrapper!(GgmlQ8_1, BlockQ8_1, 32, 36); // QK8_1=32, sizeof(BlockQ8_1)=36
define_ggml_wrapper!(GgmlQ2K, BlockQ2K, 256, 84); // QK_K=256, sizeof(BlockQ2K)=84
define_ggml_wrapper!(GgmlQ3K, BlockQ3K, 256, 110); // QK_K=256, sizeof(BlockQ3K)=110
define_ggml_wrapper!(GgmlQ4K, BlockQ4K, 256, 144); // QK_K=256, sizeof(BlockQ4K)=144
define_ggml_wrapper!(GgmlQ5K, BlockQ5K, 256, 176); // QK_K=256, sizeof(BlockQ5K)=176
define_ggml_wrapper!(GgmlQ6K, BlockQ6K, 256, 210); // QK_K=256, sizeof(BlockQ6K)=210
define_ggml_wrapper!(GgmlQ8K, BlockQ8K, 256, 292); // QK_K=256, sizeof(BlockQ8K)=292
