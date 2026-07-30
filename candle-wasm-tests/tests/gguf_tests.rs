
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
//! Regression tests for the allocation caps added in huggingface/candle#3533.
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::quantized::gguf_file::Content;
use candle::Device;
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
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn empty_header_loads() {
    let mut cursor = Cursor::new(header(0, 0));
    let content = Content::read(&mut cursor).expect("empty header should parse");
    assert!(content.metadata.is_empty());
    assert!(content.tensor_infos.is_empty());
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_oversized_metadata_kv_count() {
    assert_rejects(header(0, 1u64 << 31), "metadata_kv_count");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_oversized_tensor_count() {
    assert_rejects(header(1u64 << 31, 0), "tensor_count");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_oversized_string_length() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_oversized_array_length() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "array length");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_oversized_n_dimensions() {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "dimensions");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_deeply_nested_arrays() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes());
    for _ in 0..256 {
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    assert_rejects(buf, "nesting depth");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn empty_v1_header_loads() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    let mut cursor = Cursor::new(buf);
    Content::read(&mut cursor).expect("empty v1 header should parse");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_tensor_size_exceeding_file() {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1_073_741_824u64.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).expect("header should parse");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .expect_err("expected Err from oversized tensor load");
    let msg = format!("{err}");
    assert!(msg.contains("remaining"), "unexpected error: {msg}");
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn rejects_string_length_above_remaining_file_bytes() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 20).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}
