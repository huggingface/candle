//! Regression tests for the unaligned copy path in `candle-core/src/safetensors.rs`.
//!
//! `convert_slice` and `convert_slice_with_cast` round `elem_count` down, so the
//! `Vec<T>` they allocate holds `elem_count * size_of::<T>()` bytes. Copying
//! `data.len()` bytes into it overran the allocation whenever the input length was
//! not a multiple of the element size. `Tensor::from_raw_buffer` is public and
//! safe, so the overflow was reachable from safe code.

use candle_core::{DType, Device, Tensor};

/// Writes `payload` at an odd address inside `buf` and returns that subslice.
/// An odd address is never 2-, 4- or 8-aligned, so this reliably forces the
/// unaligned copy branch on any allocator.
fn place_unaligned<'a>(buf: &'a mut [u8], payload: &[u8]) -> &'a [u8] {
    let base = buf.as_ptr() as usize;
    let off = if base.is_multiple_of(2) { 1 } else { 2 };
    buf[off..off + payload.len()].copy_from_slice(payload);
    let data = &buf[off..off + payload.len()];
    assert!(
        !(data.as_ptr() as usize).is_multiple_of(2),
        "expected an odd address"
    );
    data
}

#[test]
fn unaligned_trailing_bytes_do_not_overflow() {
    // 7 bytes of f32 data: elem_count rounds down to 1, so exactly 4 bytes may
    // be copied. Copying all 7 wrote past the end of the allocation.
    let mut buf = vec![0u8; 32];
    let payload = [1u8; 7];
    let data = place_unaligned(&mut buf, &payload);
    let t = Tensor::from_raw_buffer(data, DType::F32, &[1], &Device::Cpu).unwrap();
    assert_eq!(t.dims(), [1]);
}

#[test]
fn unaligned_data_loads_correct_values() {
    let values: [f32; 4] = [1.0, -2.5, 3.25, 4.0];
    let mut payload = Vec::new();
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let mut buf = vec![0u8; payload.len() + 4];
    let data = place_unaligned(&mut buf, &payload);
    let t = Tensor::from_raw_buffer(data, DType::F32, &[4], &Device::Cpu).unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), values);
}

#[test]
fn aligned_data_loads_correct_values() {
    let values: [f32; 4] = [1.0, -2.5, 3.25, 4.0];
    let mut payload = Vec::new();
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let t = Tensor::from_raw_buffer(&payload, DType::F32, &[4], &Device::Cpu).unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), values);
}

#[test]
fn unaligned_wide_dtype_loads_correct_values() {
    // f64 has the largest element size, so it rounds down the most.
    let values: [f64; 3] = [1.5, -2.25, 1e10];
    let mut payload = Vec::new();
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let mut buf = vec![0u8; payload.len() + 4];
    let data = place_unaligned(&mut buf, &payload);
    let t = Tensor::from_raw_buffer(data, DType::F64, &[3], &Device::Cpu).unwrap();
    assert_eq!(t.to_vec1::<f64>().unwrap(), values);
}
