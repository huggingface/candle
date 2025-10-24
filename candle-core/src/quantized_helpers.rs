//! Helper functions for handling quantized data types
//! 
//! This module provides utilities for:
//! - Automatic dequantization when operations don't have quantized implementations
//! - Automatic requantization after operations
//! - Dispatch to specialized quantized implementations when available

use crate::{CpuStorage, DType, Error, Layout, Result};

/// Dequantize quantized storage to f32
/// 
/// This is used as a fallback for operations that don't have specialized
/// quantized implementations.
pub fn dequantize_storage(
    id: crate::dtype::QuantizedDType,
    data: &[u8],
    layout: &Layout,
) -> Result<Vec<f32>> {
    let num_elements = layout.shape().elem_count();
    let mut output = vec![0.0f32; num_elements];
    
    // Use the generated dispatch to dequantize
    crate::dtype::quantized_dispatch::dequantize_cpu(id, data, &mut output)?;
    
    Ok(output)
}

/// Quantize f32 storage back to quantized format
#[allow(unused_variables)]
pub fn quantize_storage(
    id: crate::dtype::QuantizedDType,
    data: &[f32],
    layout: &Layout,
) -> Result<Vec<u8>> {
    // Use the generated dispatch to quantize
    crate::dtype::quantized_dispatch::quantize_cpu(id, data)
}

/// Helper for Map2 operations: dequantize -> operate -> requantize
/// 
/// This is used when an operation doesn't have a specialized quantized implementation.
/// 
/// # Arguments
/// * `id` - The quantized dtype
/// * `lhs_data` - Left hand side quantized data
/// * `lhs_layout` - Left hand side layout
/// * `rhs_data` - Right hand side quantized data  
/// * `rhs_layout` - Right hand side layout
/// * `op_name` - Name of the operation (for error reporting)
/// * `f` - The operation to perform on dequantized f32 data
#[allow(unused_variables)]
pub fn map2_via_dequant<F>(
    id: crate::dtype::QuantizedDType,
    lhs_data: &[u8],
    lhs_layout: &Layout,
    rhs_data: &[u8],
    rhs_layout: &Layout,
    op_name: &str,
    f: F,
) -> Result<CpuStorage>
where
    F: FnOnce(&[f32], &Layout, &[f32], &Layout) -> Result<Vec<f32>>,
{
    // Dequantize both inputs
    let lhs_f32 = dequantize_storage(id, lhs_data, lhs_layout)?;
    let rhs_f32 = dequantize_storage(id, rhs_data, rhs_layout)?;
    
    // Perform the operation on dequantized data
    let result_f32 = f(&lhs_f32, lhs_layout, &rhs_f32, rhs_layout)?;
    
    // Requantize the result
    let result_layout = Layout::contiguous(lhs_layout.shape());
    let result_quantized = quantize_storage(id, &result_f32, &result_layout)?;
    
    Ok(CpuStorage::Quantized(id, result_quantized))
}

/// Helper for Map1 operations: dequantize -> operate -> requantize
#[allow(unused_variables)]
pub fn map1_via_dequant<F>(
    id: crate::dtype::QuantizedDType,
    data: &[u8],
    layout: &Layout,
    op_name: &str,
    f: F,
) -> Result<CpuStorage>
where
    F: FnOnce(&[f32], &Layout) -> Result<Vec<f32>>,
{
    // Dequantize input
    let data_f32 = dequantize_storage(id, data, layout)?;
    
    // Perform the operation on dequantized data
    let result_f32 = f(&data_f32, layout)?;
    
    // Requantize the result
    let result_layout = Layout::contiguous(layout.shape());
    let result_quantized = quantize_storage(id, &result_f32, &result_layout)?;
    
    Ok(CpuStorage::Quantized(id, result_quantized))
}

/// Check if two quantized types are compatible for binary operations
#[inline]
pub fn check_quantized_compat(
    id1: crate::dtype::QuantizedDType,
    id2: crate::dtype::QuantizedDType,
    op: &'static str,
) -> Result<()> {
    if id1 != id2 {
        Err(Error::DTypeMismatchBinaryOp {
            lhs: DType::Quantized(id1),
            rhs: DType::Quantized(id2),
            op,
        }
        .bt())
    } else {
        Ok(())
    }
}
