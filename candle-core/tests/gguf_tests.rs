//! Regression tests for the allocation caps added in huggingface/candle#3533.

use candle_core::quantized::gguf_file::Content;
use candle_core::Device;
use std::io::Cursor;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_V3: u32 = 3;

fn header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_kv_count.to_le_bytes());
    buf
}

fn length_prefixed(s: &[u8]) -> Vec<u8> {
    let mut buf = (s.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(s);
    buf
}

/// Pad past the upfront `count * min_per_entry` sufficient-bytes check so
/// the inner per-field cap is what fires.
fn pad(buf: &mut Vec<u8>, n: usize) {
    buf.resize(buf.len() + n, 0);
}

fn assert_rejects(buf: Vec<u8>, msg_contains: &str) {
    let mut cursor = Cursor::new(buf);
    let err = Content::read(&mut cursor).expect_err("expected Err");
    let msg = format!("{err}");
    assert!(msg.contains(msg_contains), "unexpected error: {msg}");
}

#[test]
fn empty_header_loads() {
    let mut cursor = Cursor::new(header(0, 0));
    let content = Content::read(&mut cursor).expect("empty header should parse");
    assert!(content.metadata.is_empty());
    assert!(content.tensor_infos.is_empty());
}

#[test]
fn rejects_oversized_metadata_kv_count() {
    assert_rejects(header(0, 1u64 << 31), "metadata_kv_count");
}

#[test]
fn rejects_oversized_tensor_count() {
    assert_rejects(header(1u64 << 31, 0), "tensor_count");
}

#[test]
fn rejects_oversized_string_length() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

#[test]
fn rejects_oversized_array_length() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    buf.extend_from_slice(&0u32.to_le_bytes()); // inner type = U8
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes()); // element count
    pad(&mut buf, 64);
    assert_rejects(buf, "array length");
}

#[test]
fn rejects_oversized_n_dimensions() {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "dimensions");
}

#[test]
fn rejects_deeply_nested_arrays() {
    // 256 nested arrays of arrays, well above GGUF_MAX_VALUE_DEPTH (64).
    // Each layer is type=Array(9) + inner_type(u32) + len=1(u64) = 16 bytes
    // except the innermost which terminates with a U8 (type=0) + len=0(u64).
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // outer value_type = Array
    for _ in 0..256 {
        buf.extend_from_slice(&9u32.to_le_bytes()); // inner type = Array
        buf.extend_from_slice(&1u64.to_le_bytes()); // len = 1
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // terminal inner type = U8
    buf.extend_from_slice(&0u64.to_le_bytes()); // terminal len = 0
    assert_rejects(buf, "nesting depth");
}

#[test]
fn empty_v1_header_loads() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version 1: u32 length prefixes
    buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u32.to_le_bytes()); // metadata_kv_count
    let mut cursor = Cursor::new(buf);
    Content::read(&mut cursor).expect("empty v1 header should parse");
}

#[test]
fn rejects_tensor_size_exceeding_file() {
    // Create a gguf tensor that claims shape [1_073_741_824] F32 (4 GiB), but has no actual tensor data.
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&1u32.to_le_bytes()); // n dims
    buf.extend_from_slice(&1_073_741_824u64.to_le_bytes()); // 1 GiB elements
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // no offset
    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).expect("header should parse");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .expect_err("expected Err from oversized tensor load");
    let msg = format!("{err}");
    assert!(msg.contains("remaining"), "unexpected error: {msg}");
}

#[test]
fn rejects_zero_general_alignment() {
    // `general.alignment` is read from untrusted metadata and only rejected
    // when negative; a zero value used to reach `position.div_ceil(0)` and
    // panic with "attempt to divide by zero". It must be rejected instead.
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"general.alignment"));
    buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = U32
    buf.extend_from_slice(&0u32.to_le_bytes()); // alignment = 0
    assert_rejects(buf, "alignment");
}

#[test]
fn accepts_non_power_of_two_general_alignment() {
    // Per the GGUF spec `general.alignment` need only be a non-zero multiple of
    // 8, not a power of two (e.g. 24). `div_ceil(alignment) * alignment` pads
    // correctly for any positive value, so such files must still parse.
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"general.alignment"));
    buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = U32
    buf.extend_from_slice(&24u32.to_le_bytes()); // alignment = 24 (mult of 8, not pow2)
    let mut cursor = Cursor::new(buf);
    Content::read(&mut cursor).expect("alignment 24 (a multiple of 8) should parse");
}

#[test]
fn rejects_general_alignment_not_multiple_of_8() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"general.alignment"));
    buf.extend_from_slice(&4u32.to_le_bytes()); // value_type = U32
    buf.extend_from_slice(&12u32.to_le_bytes()); // alignment = 12 (not a multiple of 8)
    assert_rejects(buf, "alignment");
}

#[test]
fn rejects_tensor_shape_product_overflow() {
    // Two dims of 2^40 overflow `Shape::elem_count()` (their product is 2^80).
    // Loading the tensor used to panic ("attempt to multiply with overflow") in
    // debug builds and wrap to a bogus small size in release builds.
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&2u32.to_le_bytes()); // n dims
    buf.extend_from_slice(&(1u64 << 40).to_le_bytes()); // dim 0
    buf.extend_from_slice(&(1u64 << 40).to_le_bytes()); // dim 1
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset
    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).expect("header should parse");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .expect_err("expected Err from overflowing tensor shape");
    let msg = format!("{err}");
    assert!(msg.contains("overflow"), "unexpected error: {msg}");
}

#[test]
fn rejects_string_length_above_remaining_file_bytes() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 20).to_le_bytes()); // 1 MB, below cap, above file size
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}
