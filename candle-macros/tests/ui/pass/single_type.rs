// Test: Single type registration
// This should compile successfully

mod candle_core {
    pub mod dtype {
        pub trait QuantizedType {
            const NAME: &'static str;
        }
    }
    
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
}

use candle_macros::register_quantized_types;
use candle_core::dtype::QuantizedType;

struct Q4_0;
impl candle_core::dtype::QuantizedType for Q4_0 {
    const NAME: &'static str = "Q4_0";
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
    
    fn matmul(
        _lhs: &[f32],
        _lhs_shape: &[usize],
        _rhs: &[u8],
        _rhs_shape: &[usize]
    ) -> candle_core::Result<Vec<f32>> {
        Ok(vec![])
    }
}

// Single type should work
register_quantized_types! {
    Q4_0
}

fn main() {
    assert_eq!(QuantizedDType::BUILTIN_COUNT, 1);
}
