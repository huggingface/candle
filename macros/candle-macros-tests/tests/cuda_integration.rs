// Integration test for CUDA operations with real candle-core
#![cfg(feature = "cuda")]

mod common;

use candle_macros::register_quantized_types;
use candle_macros_types::{CudaStorageDevice, QuantizedCudaOps, QuantizedType};

pub use common::{CudaDevice, Error, Result};

// Test quantized type with CUDA support
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Q4_0;

impl QuantizedType for Q4_0 {
    const NAME: &'static str = "Q4_0";
    const SIZE_IN_BYTES: usize = 18; // block_size=32, 32/2 + 2 scales

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 31) / 32) * 18
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 18) * 32
    }
}

impl QuantizedCudaOps for Q4_0 {
    fn quantize_cuda<D: CudaStorageDevice>(
        &self,
        _input: &cudarc::driver::CudaSlice<f32>,
        _device: &D,
    ) -> std::result::Result<cudarc::driver::CudaSlice<u8>, String> {
        // Not actually implemented - just verifying the macro generates correct CUDA code
        Err("Not implemented in test".to_string())
    }

    fn dequantize_cuda<D: CudaStorageDevice>(
        &self,
        _data: &cudarc::driver::CudaSlice<u8>,
        _output: &mut cudarc::driver::CudaSlice<f32>,
        _device: &D,
    ) -> std::result::Result<(), String> {
        Err("Not implemented in test".to_string())
    }

    fn matmul_cuda<D: CudaStorageDevice>(
        &self,
        _lhs: &cudarc::driver::CudaSlice<f32>,
        _lhs_shape: &[usize],
        _rhs: &cudarc::driver::CudaSlice<u8>,
        _rhs_shape: &[usize],
        _device: &D,
    ) -> std::result::Result<cudarc::driver::CudaSlice<f32>, String> {
        Err("Not implemented in test".to_string())
    }
}

// Type without CUDA support
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NoCuda;

impl QuantizedType for NoCuda {
    const NAME: &'static str = "NoCuda";
    const SIZE_IN_BYTES: usize = 18;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 31) / 32) * 18
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 18) * 32
    }
}

// Register types
register_quantized_types! {
    Q4_0,
    NoCuda
}

#[test]
fn test_cuda_dispatch_with_real_cudarc() {
    // This test uses the real cudarc types
    // It verifies the macro generates correct code that compiles with actual dependencies

    let dtype_q4 = QuantizedDType::Q4_0;
    eprintln!("Q4_0 has_cuda: {}", dtype_q4.has_cuda());
    assert!(dtype_q4.has_cuda());

    let dtype_no_cuda = QuantizedDType::NoCuda;
    eprintln!("NoCuda has_cuda: {}", dtype_no_cuda.has_cuda());
    assert!(!dtype_no_cuda.has_cuda());

    println!("CUDA integration test with real cudarc passed!");
}
