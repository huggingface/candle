//! Regression tests hardening the legacy GGML loader against crafted files,
//! mirroring the GGUF caps added in huggingface/candle#3533.

use candle_core::quantized::ggml_file::Content;
use candle_core::Device;
use std::io::Cursor;

const GGML_MAGIC: u32 = 0x67676d6c; // "ggml", unversioned (no align32).

fn u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

/// Build a minimal unversioned GGML file: magic, 7 hparam u32s (n_vocab = 0 so
/// the vocab section is empty), followed by a single crafted tensor header.
fn ggml_with_tensor(n_dims: u32, name_len: u32, dtype: u32, dims: &[u32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&u32(GGML_MAGIC));
    for _ in 0..7 {
        buf.extend_from_slice(&u32(0)); // hparams, n_vocab = 0
    }
    buf.extend_from_slice(&u32(n_dims));
    buf.extend_from_slice(&u32(name_len));
    buf.extend_from_slice(&u32(dtype));
    for &d in dims {
        buf.extend_from_slice(&u32(d));
    }
    buf
}

fn assert_rejects(buf: Vec<u8>, msg_contains: &str) {
    let mut cursor = Cursor::new(buf);
    let msg = match Content::read(&mut cursor, &Device::Cpu) {
        Ok(_) => panic!("expected Err, got Ok"),
        Err(e) => format!("{e}"),
    };
    assert!(msg.contains(msg_contains), "unexpected error: {msg}");
}

#[test]
fn rejects_tensor_shape_overflow() {
    // Four dims of u32::MAX overflow `dims.iter().product::<usize>()`, which
    // used to panic ("attempt to multiply with overflow") in debug builds and
    // wrap in release builds.
    let buf = ggml_with_tensor(4, 0, 0, &[u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    assert_rejects(buf, "overflow");
}

#[test]
fn rejects_tensor_size_exceeding_file() {
    // A single [1_073_741_824] F32 tensor claims 4 GiB but the file carries no
    // tensor data; it must be rejected rather than allocating 4 GiB up front.
    let buf = ggml_with_tensor(1, 0, 0, &[1_073_741_824]);
    assert_rejects(buf, "remaining");
}

#[test]
fn rejects_oversized_tensor_name() {
    // name_len far exceeds the file; must be rejected before allocating.
    let buf = ggml_with_tensor(1, 1u32 << 30, 0, &[1]);
    assert_rejects(buf, "name");
}
