// Test: Basic macro usage with multiple types
// This should compile successfully
#![allow(unexpected_cfgs)]
#![allow(unused_imports)]

// Root-level types for macro (simulating candle-core crate root)
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Msg(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

// Mock candle_core for testing
mod candle_core {
    pub mod dtype {
        pub trait QuantizedType {
            const NAME: &'static str;
            const SIZE_IN_BYTES: usize;
        }
    }
    
    pub type Result<T> = std::result::Result<T, super::Error>;
    pub use super::Error;
}

use candle_macros::register_quantized_types;
use candle_core::dtype::QuantizedType;  // Import the trait so NAME is accessible

struct Q4_0;
impl candle_core::dtype::QuantizedType for Q4_0 {
    const NAME: &'static str = "Q4_0";
    const SIZE_IN_BYTES: usize = 1;
}

impl Q4_0 {
    fn quantize(_input: &[f32]) -> candle_core::Result<Vec<u8>> {
        Ok(vec![])
    }
    
    fn dequantize(_data: &[u8], _output: &mut [f32]) -> candle_core::Result<()> {
        Ok(())
    }
    
    fn storage_size_in_bytes(_n: usize) -> usize {
        0
    }

    fn infer_element_count(_data_len: usize) -> usize {
        0
    }
    
    fn matmul(
        _lhs: &[f32],
        _lhs_shape: &[usize],
        _rhs: &[u8],
        _rhs_shape: &[usize]
    ) -> candle_core::Result<Vec<f32>> {
        Ok(vec![])
    }
}

struct Q8_0;
impl candle_core::dtype::QuantizedType for Q8_0 {
    const NAME: &'static str = "Q8_0";
    const SIZE_IN_BYTES: usize = 1;
}

impl Q8_0 {
    fn quantize(_input: &[f32]) -> candle_core::Result<Vec<u8>> {
        Ok(vec![])
    }
    
    fn dequantize(_data: &[u8], _output: &mut [f32]) -> candle_core::Result<()> {
        Ok(())
    }
    
    fn storage_size_in_bytes(_n: usize) -> usize {
        0
    }

    fn infer_element_count(_data_len: usize) -> usize {
        0
    }
    
    fn matmul(
        _lhs: &[f32],
        _lhs_shape: &[usize],
        _rhs: &[u8],
        _rhs_shape: &[usize]
    ) -> candle_core::Result<Vec<f32>> {
        Ok(vec![])
    }
}

// This should compile successfully
register_quantized_types! {
    Q4_0,
    Q8_0
}

fn main() {
    // Test that the generated code works
    let dtype = QuantizedDType::Q4_0;
    assert_eq!(dtype.name(), "Q4_0");
    assert!(!dtype.is_external());
}
