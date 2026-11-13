//! Round trips through `gguf_file::write` and `Content::read`: the writer pads
//! to the alignment the metadata declares, as the reader expects.

use candle_core::quantized::gguf_file::{self, Content, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::Cursor;

const A: [f32; 3] = [1.0, 2.0, 3.0];
const B: [f32; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

fn qtensor(values: &[f32]) -> QTensor {
    QTensor::quantize(&Tensor::new(values, &Device::Cpu).unwrap(), GgmlDType::F32).unwrap()
}

/// Writes tensors `a` and `b` with the given metadata. `name_len` sets the
/// length of a `general.name` value, which moves where the header ends.
fn write(alignment: &[Value], name_len: usize) -> candle_core::Result<Vec<u8>> {
    let name = Value::String("x".repeat(name_len));
    let mut metadata = vec![("general.name", &name)];
    for v in alignment {
        metadata.push(("general.alignment", v));
    }
    let (a, b) = (qtensor(&A), qtensor(&B));
    let mut w = Cursor::new(Vec::new());
    gguf_file::write(&mut w, &metadata, &[("a", &a), ("b", &b)])?;
    Ok(w.into_inner())
}

/// Reads the file back and checks the tensor contents and that every tensor
/// offset is a multiple of `alignment`. Returns what was read.
fn assert_round_trip(bytes: Vec<u8>, alignment: u64) -> Content {
    let mut r = Cursor::new(bytes);
    let content = Content::read(&mut r).unwrap();
    for (name, want) in [("a", &A[..]), ("b", &B[..])] {
        assert_eq!(
            content.tensor_infos[name].offset % alignment,
            0,
            "offset of {name}"
        );
        let got = content
            .tensor(&mut r, name, &Device::Cpu)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, want, "tensor {name}");
    }
    content
}

#[test]
fn round_trips_declared_alignment() {
    for alignment in [8u32, 16, 64, 128] {
        // Every header end modulo the alignment.
        for name_len in 0..alignment as usize {
            let bytes = write(&[Value::U32(alignment)], name_len).unwrap();
            assert_round_trip(bytes, alignment as u64);
        }
    }
}

#[test]
fn pads_to_32_without_a_declared_alignment() {
    // The layout is unchanged: `a` (12 bytes) and `b` (20 bytes) are each
    // padded to 32, and the file ends right after the second pad.
    for alignment in [vec![], vec![Value::U32(32)]] {
        for name_len in 0..32 {
            let bytes = write(&alignment, name_len).unwrap();
            let len = bytes.len() as u64;
            let content = assert_round_trip(bytes, 32);
            assert_eq!(content.tensor_infos["a"].offset, 0);
            assert_eq!(content.tensor_infos["b"].offset, 32);
            assert_eq!(len, content.tensor_data_offset + 64, "file length");
        }
    }
}

#[test]
fn round_trips_alignment_of_another_integer_type() {
    for name_len in 0..16 {
        let content = assert_round_trip(write(&[Value::U8(16)], name_len).unwrap(), 16);
        // `a` (12 bytes) padded to 16, not to the default of 32.
        assert_eq!(content.tensor_infos["b"].offset, 16);
    }
}

#[test]
fn uses_the_last_of_repeated_alignments() {
    // The reader keeps the last occurrence of a repeated key.
    for name_len in 0..64 {
        let bytes = write(&[Value::U32(32), Value::U32(64)], name_len).unwrap();
        assert_round_trip(bytes, 64);
        let bytes = write(&[Value::U32(64), Value::U32(16)], name_len).unwrap();
        assert_round_trip(bytes, 16);
        // The last entry decides even when the reader ignores its type, as
        // for u64, which leaves the default of 32.
        let bytes = write(&[Value::U32(64), Value::U64(16)], name_len).unwrap();
        assert_round_trip(bytes, 32);
        let bytes = write(&[Value::U64(16), Value::U32(64)], name_len).unwrap();
        assert_round_trip(bytes, 64);
    }
}

#[test]
fn ignores_a_negative_alignment() {
    // The reader ignores a negative alignment and pads to 32, so the writer
    // must too. Writing into a fixed buffer makes a writer that reads -1 as
    // an unsigned alignment fail at once instead of padding without end.
    let negative = Value::I32(-1);
    let (a, b) = (qtensor(&A), qtensor(&B));
    let mut buf = vec![0u8; 4096];
    let mut w = Cursor::new(&mut buf[..]);
    gguf_file::write(
        &mut w,
        &[("general.alignment", &negative)],
        &[("a", &a), ("b", &b)],
    )
    .unwrap();
    let len = w.position() as usize;
    assert_round_trip(buf[..len].to_vec(), 32);
}

#[test]
fn refuses_zero_alignment() {
    let zero = Value::U32(0);
    let a = qtensor(&A);
    let mut w = Cursor::new(Vec::new());
    let err = gguf_file::write(&mut w, &[("general.alignment", &zero)], &[("a", &a)]).unwrap_err();
    assert!(
        format!("{err}").contains("general.alignment must not be zero"),
        "unexpected error: {err}"
    );
    assert!(w.into_inner().is_empty(), "nothing is written");
}
