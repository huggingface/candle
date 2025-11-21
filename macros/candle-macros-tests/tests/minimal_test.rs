// Minimal test of autoderef specialization

mod common;

use candle_macros_types::{QuantizedCpuOps, QuantizedType};

#[derive(Default)]
pub struct Q8_0;

impl QuantizedType for Q8_0 {
    const NAME: &'static str = "Q8_0";
    const SIZE_IN_BYTES: usize = 34;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        num_elements.div_ceil(32) * 34
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 34) * 32
    }
}

impl QuantizedCpuOps for Q8_0 {
    fn dequantize(&self, _data: &[u8], _output: &mut [f32]) -> Result<(), String> {
        Ok(())
    }

    fn quantize(&self, _input: &[f32]) -> Result<Vec<u8>, String> {
        Ok(vec![0u8; 34])
    }

    fn matmul(
        &self,
        _lhs_f32: &[f32],
        _lhs_shape: &[usize],
        _rhs_data: &[u8],
        _rhs_shape: &[usize],
    ) -> Result<Vec<f32>, String> {
        Ok(vec![0.0f32; 8])
    }
}

#[test]
fn test_autoderef() {
    // Manual autoderef test
    trait CpuCheck {
        fn has_cpu(&self) -> bool;
    }

    struct Wrap<T>(std::marker::PhantomData<T>);

    impl<T: candle_macros_types::QuantizedType> CpuCheck for Wrap<T> {
        fn has_cpu(&self) -> bool {
            false
        }
    }

    impl<T: candle_macros_types::QuantizedCpuOps> CpuCheck for &Wrap<T> {
        fn has_cpu(&self) -> bool {
            true
        }
    }

    // Direct autoderef on concrete type - no helper function!
    let result = (&&Wrap::<Q8_0>(std::marker::PhantomData)).has_cpu();
    println!("Q8_0 has_cpu: {}", result);
    assert!(result, "Expected Q8_0 to have CPU support!");
}
