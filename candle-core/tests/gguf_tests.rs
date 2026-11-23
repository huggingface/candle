//! Regression tests for the allocation caps added in huggingface/candle#3533.

use candle_core::quantized::gguf_file::{self, Content};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
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
fn rejects_string_length_above_remaining_file_bytes() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 20).to_le_bytes()); // 1 MB, below cap, above file size
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

fn next_tag() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static TAG: AtomicUsize = AtomicUsize::new(0);
    TAG.fetch_add(1, Ordering::Relaxed)
}

fn write_one_tensor(name: &str, dtype: GgmlDType, shape: (usize, usize)) -> std::path::PathBuf {
    let dev = Device::Cpu;
    let src = Tensor::rand(-1f32, 1f32, shape, &dev).unwrap();
    let q = QTensor::quantize(&src, dtype).unwrap();
    let path = std::env::temp_dir().join(format!(
        "candle-gguf-mapped-{}-{}-{name}.gguf",
        std::process::id(),
        next_tag()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    gguf_file::write(&mut file, &[], &[(name, &q)]).unwrap();
    path
}

/// A mapped tensor has to be the copied tensor: same block bytes, same dequantised values.
/// Nothing may depend on where the storage lives.
#[test]
fn mapped_tensor_matches_the_copied_one() {
    let dev = Device::Cpu;
    let path = write_one_tensor("w", GgmlDType::Q4K, (4, 512));

    let copied = {
        let mut file = std::fs::File::open(&path).unwrap();
        let content = Content::read(&mut file).unwrap();
        content.tensor(&mut file, "w", &dev).unwrap()
    };
    // SAFETY: the test owns the file and does not modify it while the mapping is alive.
    let (content, map) = unsafe { Content::from_mmap(&path).unwrap() };
    let mapped = content.tensor_mapped(&map, "w", &dev).unwrap();

    assert_eq!(copied.dtype(), mapped.dtype());
    assert_eq!(copied.shape(), mapped.shape());
    assert_eq!(
        copied.data().unwrap().as_ref(),
        mapped.data().unwrap().as_ref()
    );
    assert_eq!(
        copied.dequantize(&dev).unwrap().to_vec2::<f32>().unwrap(),
        mapped.dequantize(&dev).unwrap().to_vec2::<f32>().unwrap()
    );

    drop(mapped);
    std::fs::remove_file(&path).unwrap();
}

/// A window that is not a whole number of blocks, is misaligned, or runs past the mapping is
/// an error rather than a silent slice.
#[test]
fn mapped_window_is_checked() {
    let path = write_one_tensor("w", GgmlDType::Q4K, (2, 512));
    let (content, map) = unsafe { Content::from_mmap(&path).unwrap() };
    let info = content.tensor_infos.get("w").unwrap();
    let start = (content.tensor_data_offset + info.offset) as usize;
    let size = info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();

    assert!(QStorage::from_mapped(map.clone(), start..start + size - 1, info.ggml_dtype).is_err());
    assert!(
        QStorage::from_mapped(map.clone(), start + 1..start + 1 + size, info.ggml_dtype).is_err()
    );
    assert!(QStorage::from_mapped(map.clone(), start..map.len() + 1, info.ggml_dtype).is_err());
    let storage = QStorage::from_mapped(map, start..start + size, info.ggml_dtype).unwrap();
    let qtensor = QTensor::new(storage, info.shape.clone()).unwrap();
    assert_eq!(qtensor.storage_size_in_bytes(), size);

    std::fs::remove_file(&path).unwrap();
}

/// The matmul path, including the repack cache, has to work straight off a mapped storage.
#[test]
fn mapped_tensor_matmuls_like_the_copied_one() {
    let dev = Device::Cpu;
    let path = write_one_tensor("w", GgmlDType::Q4K, (4, 512));

    let copied = {
        let mut file = std::fs::File::open(&path).unwrap();
        let content = Content::read(&mut file).unwrap();
        content.tensor(&mut file, "w", &dev).unwrap()
    };
    // SAFETY: the test owns the file and does not modify it while the mapping is alive.
    let (content, map) = unsafe { Content::from_mmap(&path).unwrap() };
    let mapped = content.tensor_mapped(&map, "w", &dev).unwrap();

    let xs = Tensor::rand(-1f32, 1f32, (3, 512), &dev).unwrap();
    let a = QMatMul::from_qtensor(copied).unwrap().forward(&xs).unwrap();
    let b = QMatMul::from_qtensor(mapped).unwrap().forward(&xs).unwrap();
    assert_eq!(a.dims(), b.dims());
    assert_eq!(a.to_vec2::<f32>().unwrap(), b.to_vec2::<f32>().unwrap());

    std::fs::remove_file(&path).unwrap();
}
