// Integration test for CPU-only operations using mocked candle-core types
mod common;

use candle_macros::register_quantized_types;
use candle_macros_types::{QuantizedCpuOps, QuantizedType};

pub use common::{Error, Result};

#[cfg(feature = "cuda")]
pub use common::CudaDevice;

// Test quantized type with CPU support
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Q8_0;

impl QuantizedType for Q8_0 {
    const NAME: &'static str = "Q8_0";
    const SIZE_IN_BYTES: usize = 34;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 31) / 32) * 34
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 34) * 32
    }
}

impl QuantizedCpuOps for Q8_0 {
    fn dequantize(&self, _data: &[u8], _output: &mut [f32]) -> std::result::Result<(), String> {
        Ok(())
    }

    fn quantize(&self, _input: &[f32]) -> std::result::Result<Vec<u8>, String> {
        Ok(vec![0u8; Self::SIZE_IN_BYTES])
    }

    fn matmul(
        &self,
        _lhs_f32: &[f32],
        _lhs_shape: &[usize],
        _rhs_data: &[u8],
        _rhs_shape: &[usize],
    ) -> std::result::Result<Vec<f32>, String> {
        Ok(vec![0.0f32; 8])
    }
}

// Type without CPU support
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NoCpu;

impl QuantizedType for NoCpu {
    const NAME: &'static str = "NoCpu";
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
    Q8_0,
    NoCpu
}

#[test]
fn test_cpu_dispatch() {
    use quantized_dispatch;

    let dtype_q8 = QuantizedDType::Q8_0;
    eprintln!("Q8_0 has_cpu: {}", dtype_q8.has_cpu());
    assert!(dtype_q8.has_cpu());

    let dtype_no_cpu = QuantizedDType::NoCpu;
    eprintln!("NoCpu has_cpu: {}", dtype_no_cpu.has_cpu());
    assert!(!dtype_no_cpu.has_cpu());

    // Test actual operations
    let data = vec![0u8; 34];
    let mut output = vec![0.0f32; 32];
    let result = quantized_dispatch::dequantize_cpu(dtype_q8, &data, &mut output);
    assert!(result.is_ok());

    // Test unsupported type
    let result = quantized_dispatch::dequantize_cpu(dtype_no_cpu, &data, &mut output);
    assert!(result.is_err());

    println!("CPU-only integration test passed!");
}
