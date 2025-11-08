// Integration test showcasing external quantized types
//
// Note: This test demonstrates the external type system but doesn't use the
// register_external_quantized_type! macro because that macro is designed for
// use within candle-core. Instead, we show the manual approach and concepts.

mod common;

use candle_macros::register_quantized_types;
use candle_macros_types::{QuantizedCpuOps, QuantizedType};

pub use common::{Error, Result};

// Re-export for macro if needed
pub use common::Error as CoreError;

// ==================== Define External Types ====================

// External type with full CPU support
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ExternalQ4;

impl QuantizedType for ExternalQ4 {
    const NAME: &'static str = "external_q4";
    const SIZE_IN_BYTES: usize = 18;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 31) / 32) * 18
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 18) * 32
    }
}

impl QuantizedCpuOps for ExternalQ4 {
    fn dequantize(&self, data: &[u8], output: &mut [f32]) -> std::result::Result<(), String> {
        // Simple mock implementation
        for (i, val) in output.iter_mut().enumerate() {
            *val = data.get(i % data.len()).copied().unwrap_or(0) as f32;
        }
        Ok(())
    }

    fn quantize(&self, input: &[f32]) -> std::result::Result<Vec<u8>, String> {
        // Simple mock implementation
        let storage_size = self.storage_size_in_bytes(input.len());
        let mut output = vec![0u8; storage_size];
        for (i, &val) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = val as u8;
            }
        }
        Ok(output)
    }

    fn matmul(
        &self,
        lhs_f32: &[f32],
        lhs_shape: &[usize],
        _rhs_data: &[u8],
        rhs_shape: &[usize],
    ) -> std::result::Result<Vec<f32>, String> {
        // Simple mock matmul: just return zeros with correct size
        let m = lhs_shape[0];
        let n = rhs_shape[1];
        Ok(vec![lhs_f32.get(0).copied().unwrap_or(0.0); m * n])
    }
}

// External type WITHOUT CPU support (to test optional CPU ops)
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ExternalNoCpu;

impl QuantizedType for ExternalNoCpu {
    const NAME: &'static str = "external_no_cpu";
    const SIZE_IN_BYTES: usize = 20;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 15) / 16) * 20
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 20) * 16
    }
}

// Note: ExternalNoCpu does NOT implement QuantizedCpuOps

// ==================== Manual External Type Registration ====================
//
// This demonstrates how external types would be registered manually.
// In actual candle-core usage, the register_external_quantized_type! macro
// would handle this automatically.

use std::sync::OnceLock;

static EXTERNAL_Q4_DTYPE: OnceLock<()> = OnceLock::new();
static EXTERNAL_NO_CPU_DTYPE: OnceLock<()> = OnceLock::new();

// Mock registration functions (in real code these would use the actual registration)
fn get_external_q4_dtype() -> &'static str {
    EXTERNAL_Q4_DTYPE.get_or_init(|| {
        println!("Registering ExternalQ4 (would call register_external_quant_type here)");
    });
    "external_q4"
}

fn get_external_no_cpu_dtype() -> &'static str {
    EXTERNAL_NO_CPU_DTYPE.get_or_init(|| {
        println!("Registering ExternalNoCpu (would call register_external_quant_type here)");
    });
    "external_no_cpu"
}

// Also register some built-in types for comparison
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BuiltInQ8;

impl QuantizedType for BuiltInQ8 {
    const NAME: &'static str = "builtin_q8";
    const SIZE_IN_BYTES: usize = 34;

    fn storage_size_in_bytes(&self, num_elements: usize) -> usize {
        ((num_elements + 31) / 32) * 34
    }

    fn infer_element_count(&self, data_len: usize) -> usize {
        (data_len / 34) * 32
    }
}

impl QuantizedCpuOps for BuiltInQ8 {
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

register_quantized_types! {
    BuiltInQ8
}

// ==================== Tests ====================

#[test]
fn test_external_type_concepts() {
    // Demonstrate external type concept
    let ext_q4_name = get_external_q4_dtype();
    let ext_no_cpu_name = get_external_no_cpu_dtype();

    println!("External Q4 would be registered as: '{}'", ext_q4_name);
    println!(
        "External NoCpu would be registered as: '{}'",
        ext_no_cpu_name
    );

    // Show that types can be instantiated
    let q4 = ExternalQ4::default();
    let no_cpu = ExternalNoCpu::default();

    // Test their trait implementations
    assert_eq!(ExternalQ4::NAME, "external_q4");
    assert_eq!(ExternalQ4::SIZE_IN_BYTES, 18);
    assert_eq!(q4.storage_size_in_bytes(64), 36);
    assert_eq!(q4.infer_element_count(36), 64);

    assert_eq!(ExternalNoCpu::NAME, "external_no_cpu");
    assert_eq!(ExternalNoCpu::SIZE_IN_BYTES, 20);
    assert_eq!(no_cpu.storage_size_in_bytes(32), 40);

    println!("External type concepts test passed!");
}

#[test]
fn test_external_type_cpu_operations() {
    // Test the CPU operations directly on the type
    let q4 = ExternalQ4::default();

    // Test quantize
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = q4.quantize(&input);
    assert!(quantized.is_ok(), "Quantize should succeed");
    let quantized = quantized.unwrap();
    println!(
        "Quantized {} elements to {} bytes",
        input.len(),
        quantized.len()
    );

    // Test dequantize
    let data = vec![
        1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    ];
    let mut output = vec![0.0f32; 32];
    let result = q4.dequantize(&data, &mut output);
    assert!(result.is_ok(), "Dequantize should succeed");
    println!(
        "Dequantized {} bytes to {} elements",
        data.len(),
        output.len()
    );

    // Test matmul
    let lhs = vec![1.0, 2.0, 3.0, 4.0];
    let lhs_shape = vec![2, 2];
    let rhs = vec![5u8; 18];
    let rhs_shape = vec![2, 2];
    let result = q4.matmul(&lhs, &lhs_shape, &rhs, &rhs_shape);
    assert!(result.is_ok(), "Matmul should succeed");
    let result = result.unwrap();
    println!("Matmul result: {:?}", result);
    assert_eq!(result.len(), 4); // 2x2 matrix

    println!("External type CPU operations test passed!");
}

#[test]
fn test_builtin_vs_external_traits() {
    // Compare built-in and external types at the trait level
    let builtin = BuiltInQ8::default();
    let external = ExternalQ4::default();

    // Both implement QuantizedType
    assert_eq!(BuiltInQ8::NAME, "builtin_q8");
    assert_eq!(ExternalQ4::NAME, "external_q4");

    // Both implement QuantizedCpuOps
    let input = vec![1.0, 2.0, 3.0, 4.0];
    assert!(builtin.quantize(&input).is_ok());
    assert!(external.quantize(&input).is_ok());

    println!("Built-in vs external traits test passed!");
}

#[test]
fn test_optional_cpu_ops() {
    // ExternalNoCpu does NOT implement QuantizedCpuOps
    let no_cpu = ExternalNoCpu::default();

    // But it still implements QuantizedType
    assert_eq!(ExternalNoCpu::NAME, "external_no_cpu");
    assert_eq!(no_cpu.storage_size_in_bytes(32), 40);

    // This demonstrates that CPU ops are optional
    // In actual usage with register_external_quantized_type!, the macro would
    // detect at compile-time that ExternalNoCpu doesn't implement QuantizedCpuOps
    // and would register it without CPU operations.

    println!("Optional CPU ops test passed!");
}

#[test]
fn test_builtin_type_dispatch() {
    use quantized_dispatch;

    let builtin = QuantizedDType::BuiltInQ8;

    // Test that built-in types work with dispatch
    assert!(builtin.has_cpu());
    assert!(!builtin.is_external());

    // Test operations
    let data = vec![0u8; 34];
    let mut output = vec![0.0f32; 32];
    let result = quantized_dispatch::dequantize_cpu(builtin, &data, &mut output);
    assert!(result.is_ok());

    println!("Built-in type dispatch test passed!");
}

#[test]
fn test_registration_workflow() {
    // This test documents the workflow for external type registration

    // Step 1: Define a type implementing QuantizedType (required)
    let _q4 = ExternalQ4::default();

    // Step 2: Optionally implement QuantizedCpuOps (CPU operations)
    // ExternalQ4 has this

    // Step 3: Optionally implement QuantizedCudaOps (CUDA operations)
    // Not implemented in this test

    // Step 4: Optionally implement QuantizedMetalOps (Metal operations)
    // Not implemented in this test

    // Step 5: Use register_external_quantized_type! macro
    // In actual code: register_external_quantized_type!(ExternalQ4);
    // This would:
    // - Extract function pointers for all implemented operations
    // - Register them in the global registry
    // - Return a QuantizedDType::External variant

    // Step 6: Use the registered type just like built-in types
    // let dtype = get_quantized_dtype();
    // dtype.quantize(...);
    // dtype.dequantize(...);

    println!("Registration workflow documentation test passed!");
}
