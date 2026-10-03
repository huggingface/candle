//! Regression tests for the raw-slice handling in `from_raw_data`, see
//! huggingface/candle#3815.
//!
//! `qtensor_from_ggml` is public and safe, but it reinterpreted a `&[u8]` as a
//! `&[T]` with `slice::from_raw_parts` and no checks. A byte slice only carries
//! 1-byte alignment and its length is unrelated to the size implied by `dims`, so
//! both a misaligned and a too-short buffer were undefined behaviour reachable
//! from safe code.

use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType};
use candle_core::Device;

/// The values must survive whether the buffer is borrowed or copied, so this
/// asserts on the loaded contents rather than on which path was taken.
#[test]
fn unaligned_data_loads_correctly() {
    let values: [f32; 4] = [1.0, -2.5, 3.25, 4.0];
    let mut buf = vec![0u8]; // leading byte offsets the payload
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let t = qtensor_from_ggml(GgmlDType::F32, &buf[1..], vec![4], &Device::Cpu).unwrap();
    let got = t
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, values);
}

#[test]
fn aligned_data_loads_correctly() {
    let values: [f32; 4] = [1.0, -2.5, 3.25, 4.0];
    let mut buf = Vec::new();
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let t = qtensor_from_ggml(GgmlDType::F32, &buf, vec![4], &Device::Cpu).unwrap();
    let got = t
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, values);
}

#[test]
fn rejects_truncated_data() {
    // dims=[32] with Q4_0 is one 18-byte block; supply fewer bytes than that.
    let buf = vec![0u8; 4];
    let err = qtensor_from_ggml(GgmlDType::Q4_0, &buf, vec![32], &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(err.contains("truncated"), "unexpected error: {err}");
}

#[test]
fn accepts_empty_tensor() {
    // An empty `Vec<u8>` has a dangling pointer that is not aligned for the block
    // type, so this used to trip the `from_raw_parts` alignment precondition.
    let buf: Vec<u8> = Vec::new();
    let t = qtensor_from_ggml(GgmlDType::Q4_0, &buf, vec![0], &Device::Cpu).unwrap();
    assert_eq!(t.shape().dims(), [0]);
}
