#![allow(unexpected_cfgs)]
#![allow(unused_imports)]

// Simulate an external crate registering a custom quantized type
// The trick is to alias the crate root as candle_core so the macro works

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

// Step 1: Define the core types and traits that would come from candle-core
pub mod dtype {
    pub trait QuantizedType {
        const NAME: &'static str;
        const SIZE_IN_BYTES: usize;
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Msg(String),
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Msg(msg)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Msg(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for Error {}

// Step 2: Register built-in types
pub mod quantized {
    use super::*;
    use super::dtype::QuantizedType;

    // Built-in type for testing
    #[derive(Debug, Copy, Clone)]
    pub struct Q4_0;

    impl Q4_0 {
        pub const NAME: &'static str = "q4_0";
        pub const SIZE_IN_BYTES: usize = 18;
        pub const BLOCK_SIZE: usize = 32;

        pub fn dequantize(_data: &[u8], _output: &mut [f32]) -> Result<()> {
            Ok(())
        }

        pub fn quantize(_input: &[f32]) -> Result<Vec<u8>> {
            Ok(vec![])
        }

        pub fn storage_size_in_bytes(_n: usize) -> usize {
            0
        }

        pub fn infer_element_count(data_len: usize) -> usize {
            let num_blocks = data_len / Self::SIZE_IN_BYTES;
            num_blocks * Self::BLOCK_SIZE
        }

        pub fn matmul(
            _lhs: &[f32],
            _lhs_shape: &[usize],
            _rhs: &[u8],
            _rhs_shape: &[usize],
        ) -> Result<Vec<f32>> {
            Ok(vec![])
        }
    }

    impl QuantizedType for Q4_0 {
        const NAME: &'static str = "q4_0";
        const SIZE_IN_BYTES: usize = 18;
    }

    candle_macros::register_quantized_types! { Q4_0 }
}

// Step 3: External crate defines its own quantized type
// This would normally be in a separate crate that depends on candle
mod my_external_crate {
    // Alias the crate root as candle_core so the macro expansion works
    use crate as candle_core;
    use candle_core::{dtype::QuantizedType, Result};

    // Define custom quantized type using the derive macro
    #[derive(Debug, Copy, Clone, candle_macros::QuantizedType)]
    #[quantized(name = "my_custom_q4", size_in_bytes = 20)]
    pub struct MyCustomQ4;

    // Implement required operations
    impl MyCustomQ4 {
        pub const BLOCK_SIZE: usize = 32;

        pub fn dequantize(_data: &[u8], output: &mut [f32]) -> Result<()> {
            // Custom dequantization logic
            for elem in output.iter_mut() {
                *elem = 0.5; // Dummy implementation
            }
            Ok(())
        }

        pub fn quantize(input: &[f32]) -> Result<Vec<u8>> {
            // Custom quantization logic
            let num_blocks = (input.len() + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
            Ok(vec![0u8; num_blocks * Self::SIZE_IN_BYTES])
        }

        pub fn storage_size_in_bytes(num_elements: usize) -> usize {
            let num_blocks = (num_elements + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
            num_blocks * Self::SIZE_IN_BYTES
        }

        pub fn infer_element_count(data_len: usize) -> usize {
            let num_blocks = data_len / Self::SIZE_IN_BYTES;
            num_blocks * Self::BLOCK_SIZE
        }

        pub fn matmul(
            _lhs: &[f32],
            _lhs_shape: &[usize],
            _rhs: &[u8],
            _rhs_shape: &[usize],
        ) -> Result<Vec<f32>> {
            // Custom matmul logic
            Ok(vec![1.0])
        }
    }

    // Step 4: Register the external type using the macro
    // This generates a get_quantized_dtype() function
    candle_macros::register_external_quantized_type!(MyCustomQ4);
}

// Step 5: Use the external type
fn main() {
    use quantized::{quantized_dispatch, QuantizedDType};

    // Get the quantized dtype for the external type
    let my_dtype = my_external_crate::get_quantized_dtype();

    // Verify it's an external type
    assert!(my_dtype.is_external());
    assert_eq!(my_dtype.name(), "my_custom_q4");
    assert_eq!(my_dtype.size_in_bytes(), 20);

    // Test quantize/dequantize roundtrip with exactly one block (32 elements)
    let input = vec![1.0f32; 32];
    let quantized = quantized_dispatch::quantize_cpu(my_dtype, &input).unwrap();
    
    // Verify quantized size is one block
    assert_eq!(quantized.len(), 20); // One block = 20 bytes
    
    let mut output = vec![0.0f32; input.len()];
    quantized_dispatch::dequantize_cpu(my_dtype, &quantized, &mut output).unwrap();

    // Test convenience dequantize method
    let output2 = my_dtype.dequantize(&quantized, None).unwrap();
    assert_eq!(output.len(), output2.len());

    // Test infer_element_count
    let elem_count = quantized_dispatch::infer_element_count(my_dtype, quantized.len());
    assert_eq!(elem_count, 32); // One block = 32 elements

    // Test storage_size calculation
    let storage_size = quantized_dispatch::storage_size_in_bytes(my_dtype, 64);
    assert_eq!(storage_size, 40); // 2 blocks * 20 bytes

    // Test matmul
    let lhs = vec![1.0f32; 10];
    let rhs = vec![0u8; 20];
    let result = quantized_dispatch::matmul_cpu(&lhs, &[10], my_dtype, &rhs, &[20]).unwrap();
    assert!(!result.is_empty());

    println!("External quantized type works correctly!");
}
