use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::quantized::gguf_file::Content;
use std::io::Cursor;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 2;

fn write_header(data: &mut Vec<u8>, tensor_count: u64, metadata_count: u64) {
    data.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    data.write_u32::<LittleEndian>(GGUF_VERSION).unwrap();
    data.write_u64::<LittleEndian>(tensor_count).unwrap();
    data.write_u64::<LittleEndian>(metadata_count).unwrap();
}

fn write_string(data: &mut Vec<u8>, value: &str) {
    data.write_u64::<LittleEndian>(value.len() as u64).unwrap();
    data.extend_from_slice(value.as_bytes());
}

fn read_err(data: Vec<u8>) -> String {
    let mut reader = Cursor::new(data);
    Content::read(&mut reader).unwrap_err().to_string()
}

#[test]
fn empty_header_loads() {
    let mut data = Vec::new();
    write_header(&mut data, 0, 0);
    let mut reader = Cursor::new(data);
    let content = Content::read(&mut reader).unwrap();
    assert!(content.metadata.is_empty());
    assert!(content.tensor_infos.is_empty());
}

#[test]
fn rejects_oversized_counts() {
    let mut data = Vec::new();
    write_header(&mut data, (1 << 20) + 1, 0);
    assert!(read_err(data).contains("tensor count"));

    let mut data = Vec::new();
    write_header(&mut data, 0, (1 << 20) + 1);
    assert!(read_err(data).contains("metadata count"));
}

#[test]
fn rejects_oversized_or_truncated_string() {
    let mut data = Vec::new();
    write_header(&mut data, 0, 1);
    data.write_u64::<LittleEndian>((1 << 30) + 1).unwrap();
    assert!(read_err(data).contains("string length"));

    let mut data = Vec::new();
    write_header(&mut data, 0, 1);
    data.write_u64::<LittleEndian>(8).unwrap();
    assert!(read_err(data).contains("need 8 bytes for string"));
}

#[test]
fn rejects_oversized_or_truncated_array() {
    let mut data = Vec::new();
    write_header(&mut data, 0, 1);
    write_string(&mut data, "array");
    data.write_u32::<LittleEndian>(9).unwrap();
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u64::<LittleEndian>((1 << 30) + 1).unwrap();
    assert!(read_err(data).contains("array length"));

    let mut data = Vec::new();
    write_header(&mut data, 0, 1);
    write_string(&mut data, "array");
    data.write_u32::<LittleEndian>(9).unwrap();
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u64::<LittleEndian>(8).unwrap();
    assert!(read_err(data).contains("array values"));
}

#[test]
fn rejects_duplicate_metadata_key() {
    let mut data = Vec::new();
    write_header(&mut data, 0, 2);
    write_string(&mut data, "dup");
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u8(1).unwrap();
    write_string(&mut data, "dup");
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u8(2).unwrap();
    assert!(read_err(data).contains("duplicate metadata key"));
}

#[test]
fn rejects_bad_alignment() {
    for alignment in [0, 3] {
        let mut data = Vec::new();
        write_header(&mut data, 0, 1);
        write_string(&mut data, "general.alignment");
        data.write_u32::<LittleEndian>(4).unwrap();
        data.write_u32::<LittleEndian>(alignment).unwrap();
        assert!(read_err(data).contains("invalid alignment"));
    }
}

#[test]
fn rejects_bad_tensor_metadata() {
    let mut data = Vec::new();
    write_header(&mut data, 1, 0);
    write_string(&mut data, "tensor");
    data.write_u32::<LittleEndian>(5).unwrap();
    assert!(read_err(data).contains("tensor rank"));

    let mut data = Vec::new();
    write_header(&mut data, 2, 0);
    for _ in 0..2 {
        write_string(&mut data, "tensor");
        data.write_u32::<LittleEndian>(1).unwrap();
        data.write_u64::<LittleEndian>(1).unwrap();
        data.write_u32::<LittleEndian>(0).unwrap();
        data.write_u64::<LittleEndian>(0).unwrap();
    }
    assert!(read_err(data).contains("duplicate tensor name"));
}

#[test]
fn rejects_bad_tensor_data_range() {
    let mut data = Vec::new();
    write_header(&mut data, 1, 0);
    write_string(&mut data, "tensor");
    data.write_u32::<LittleEndian>(1).unwrap();
    data.write_u64::<LittleEndian>(1).unwrap();
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u64::<LittleEndian>(1024).unwrap();
    assert!(read_err(data).contains("tensor data range"));
}

#[test]
fn rejects_bad_quantized_tensor_shape() {
    let mut data = Vec::new();
    write_header(&mut data, 1, 0);
    write_string(&mut data, "tensor");
    data.write_u32::<LittleEndian>(2).unwrap();
    data.write_u64::<LittleEndian>(16).unwrap();
    data.write_u64::<LittleEndian>(2).unwrap();
    data.write_u32::<LittleEndian>(2).unwrap();
    data.write_u64::<LittleEndian>(0).unwrap();
    assert!(read_err(data).contains("last dim divisible by block size"));
}

#[test]
fn rejects_tensor_byte_size_overflow() {
    let mut data = Vec::new();
    write_header(&mut data, 1, 0);
    write_string(&mut data, "tensor");
    data.write_u32::<LittleEndian>(1).unwrap();
    data.write_u64::<LittleEndian>(usize::MAX as u64).unwrap();
    data.write_u32::<LittleEndian>(0).unwrap();
    data.write_u64::<LittleEndian>(0).unwrap();
    assert!(read_err(data).contains("tensor byte size"));
}
