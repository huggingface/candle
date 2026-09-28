use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, DType, Device, Layout, Result, Tensor};
use float8::F8E4M3;
use half::{bf16, f16};

const DTYPES: [DType; 10] = [
    DType::U8,
    DType::U32,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::F8E4M3,
];

fn samples() -> Vec<CpuStorage> {
    vec![
        CpuStorage::U8((0..12).collect()),
        CpuStorage::U32((0..12).collect()),
        CpuStorage::I16((0..12).collect()),
        CpuStorage::I32((0..12).collect()),
        CpuStorage::I64((0..12).collect()),
        CpuStorage::BF16((0..12).map(|v| bf16::from_f32(v as f32)).collect()),
        CpuStorage::F16((0..12).map(|v| f16::from_f32(v as f32)).collect()),
        CpuStorage::F32((0..12).map(|v| v as f32).collect()),
        CpuStorage::F64((0..12).map(|v| v as f64).collect()),
        CpuStorage::F8E4M3((0..12).map(|v| F8E4M3::from_f32(v as f32)).collect()),
    ]
}

// Read the result directly, without exercising to_dtype a second time.
fn values(storage: &CpuStorage) -> Vec<f64> {
    match storage {
        CpuStorage::U8(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::U32(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::I16(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::I32(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::I64(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::BF16(v) => v.iter().map(|v| v.to_f64()).collect(),
        CpuStorage::F16(v) => v.iter().map(|v| v.to_f64()).collect(),
        CpuStorage::F32(v) => v.iter().map(|&v| v as f64).collect(),
        CpuStorage::F64(v) => v.clone(),
        CpuStorage::F8E4M3(v) => v.iter().map(|v| v.to_f64()).collect(),
        _ => panic!("unexpected packed storage"),
    }
}

#[test]
fn cpu_dtype_matrix_preserves_layout_order() -> Result<()> {
    let layouts = [
        (Layout::contiguous((2, 3)), vec![0., 1., 2., 3., 4., 5.]),
        (
            Layout::contiguous_with_offset((2, 3), 2),
            vec![2., 3., 4., 5., 6., 7.],
        ),
        (
            Layout::new((3, 2).into(), vec![1, 3], 0),
            vec![0., 3., 1., 4., 2., 5.],
        ),
        (
            Layout::new((2, 2).into(), vec![4, 2], 1),
            vec![1., 3., 5., 7.],
        ),
        (
            Layout::new((2, 3).into(), vec![0, 1], 2),
            vec![2., 3., 4., 2., 3., 4.],
        ),
        (Layout::contiguous((0, 3)), vec![]),
    ];
    for src in samples() {
        for dst in DTYPES {
            for (layout, expected) in &layouts {
                let converted = src.to_dtype(layout, dst)?;
                assert_eq!(converted.dtype(), dst);
                assert_eq!(
                    values(&converted),
                    *expected,
                    "{:?} -> {dst:?}",
                    src.dtype()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_dtype_integer_precision_and_truncation() -> Result<()> {
    let input = vec![i64::MIN, i64::MAX, (1_i64 << 53) + 1, -(1_i64 << 53) - 1];
    let storage = CpuStorage::I64(input.clone());
    let layout = Layout::contiguous(input.len());
    let CpuStorage::I64(identity) = storage.to_dtype(&layout, DType::I64)? else {
        panic!("expected i64")
    };
    assert_eq!(identity, input);
    let CpuStorage::I32(narrowed) = storage.to_dtype(&layout, DType::I32)? else {
        panic!("expected i32")
    };
    assert_eq!(narrowed, [0, -1, 1, -1]);
    let storage = CpuStorage::U32(vec![u32::MAX, 65_536, 32_768, 255]);
    let CpuStorage::I16(narrowed) = storage.to_dtype(&layout, DType::I16)? else {
        panic!("expected i16")
    };
    assert_eq!(narrowed, [-1, 0, i16::MIN, 255]);
    Ok(())
}

#[test]
fn cpu_dtype_float_to_integer_saturates() -> Result<()> {
    let input = vec![
        f64::NAN,
        f64::NEG_INFINITY,
        -1.9,
        -0.,
        0.,
        1.9,
        256.,
        f64::INFINITY,
    ];
    let layout = Layout::contiguous(input.len());
    let storage = CpuStorage::F64(input);
    let CpuStorage::U8(u8s) = storage.to_dtype(&layout, DType::U8)? else {
        panic!("expected u8")
    };
    assert_eq!(u8s, [0, 0, 0, 0, 0, 1, 255, 255]);
    let CpuStorage::I32(i32s) = storage.to_dtype(&layout, DType::I32)? else {
        panic!("expected i32")
    };
    assert_eq!(i32s, [0, i32::MIN, -1, 0, 0, 1, 256, i32::MAX]);
    Ok(())
}

#[test]
fn cpu_dtype_f64_retains_direct_conversion_paths() -> Result<()> {
    let input = vec![
        0.,
        -0.,
        1.000_488_281_25 + 2_f64.powi(-30),
        1.003_906_25 + 2_f64.powi(-30),
        1.0625 + 2_f64.powi(-30),
        -1.0625 - 2_f64.powi(-30),
        1e-80,
        1e80,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let layout = Layout::contiguous(input.len());
    let storage = CpuStorage::F64(input.clone());
    let CpuStorage::F16(f16s) = storage.to_dtype(&layout, DType::F16)? else {
        panic!("expected f16")
    };
    let CpuStorage::BF16(bf16s) = storage.to_dtype(&layout, DType::BF16)? else {
        panic!("expected bf16")
    };
    let CpuStorage::F8E4M3(f8s) = storage.to_dtype(&layout, DType::F8E4M3)? else {
        panic!("expected f8e4m3")
    };
    for (i, &v) in input.iter().enumerate() {
        assert_eq!(f16s[i].to_bits(), f16::from_f64(v).to_bits());
        assert_eq!(bf16s[i].to_bits(), bf16::from_f64(v).to_bits());
        assert_eq!(f8s[i].to_bits(), F8E4M3::from_f64(v).to_bits());
    }
    Ok(())
}

#[test]
fn cpu_dtype_small_float_identity_preserves_all_bits() -> Result<()> {
    let f16s: Vec<_> = (0..=u16::MAX).map(f16::from_bits).collect();
    let bf16s: Vec<_> = (0..=u16::MAX).map(bf16::from_bits).collect();
    let f8s: Vec<_> = (0..=u8::MAX).map(F8E4M3::from_bits).collect();
    macro_rules! check_identity {
        ($input:ident, $variant:ident) => {
            let layout = Layout::contiguous($input.len());
            let storage = CpuStorage::$variant($input.clone());
            let CpuStorage::$variant(output) = storage.to_dtype(&layout, DType::$variant)? else {
                panic!("unexpected dtype")
            };
            assert_eq!(
                output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                $input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        };
    }
    check_identity!(f16s, F16);
    check_identity!(bf16s, BF16);
    check_identity!(f8s, F8E4M3);
    Ok(())
}

#[test]
fn cpu_dtype_packed_formats_remain_unsupported() {
    let packed = [
        CpuStorage::F6E2M3(vec![0]),
        CpuStorage::F6E3M2(vec![0]),
        CpuStorage::F4(vec![0]),
        CpuStorage::F8E8M0(vec![0]),
    ];
    let layout = Layout::contiguous(1);
    for src in samples().iter().chain(packed.iter()) {
        for dst in DTYPES.into_iter().chain(packed.iter().map(|p| p.dtype())) {
            let dst_packed = packed.iter().any(|p| p.dtype() == dst);
            let src_packed = packed.iter().any(|p| p.dtype() == src.dtype());
            if src_packed || dst_packed {
                let expected_dtype = if dst_packed { dst } else { src.dtype() };
                let mut error = src.to_dtype(&layout, dst).unwrap_err();
                while let candle_core::Error::WithBacktrace { inner, .. } = error {
                    error = *inner;
                }
                assert!(matches!(
                    error,
                    candle_core::Error::UnsupportedDTypeForOp(dtype, "to_dtype")
                        if dtype == expected_dtype
                ));
            }
        }
    }
}

#[test]
fn cpu_tensor_to_dtype_preserves_a_transposed_view() -> Result<()> {
    let tensor = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.]], &Device::Cpu)?;
    let view = tensor.transpose(0, 1)?.narrow(0, 1, 2)?;
    let converted = view.to_dtype(DType::I16)?;
    assert_eq!(converted.dims(), &[2, 2]);
    assert_eq!(converted.to_vec2::<i16>()?, [[1, 4], [2, 5]]);
    Ok(())
}

#[test]
fn cpu_dtype_f32_to_f16_matches_scalar_bits() -> Result<()> {
    let mut input = vec![
        0.,
        -0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        65_504.,
        65_520.,
    ];
    input.extend((0..=u16::MAX).map(|bits| f16::from_bits(bits).to_f32()));
    // Exercise both signs and adjacent f32 values around every positive finite
    // f16 rounding midpoint, including the normal/subnormal boundary.
    for bits in 0..0x7bff {
        let a = f16::from_bits(bits).to_f32();
        let b = f16::from_bits(bits + 1).to_f32();
        let midpoint = ((a + b) * 0.5).to_bits();
        for bits in [midpoint - 1, midpoint, midpoint + 1] {
            let v = f32::from_bits(bits);
            input.extend([v, -v]);
        }
    }
    let mut state = 0x1234_5678_u32;
    for _ in 0..200_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        input.push(f32::from_bits(state));
    }
    let expected: Vec<_> = input.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let layout = Layout::contiguous(input.len());
    let CpuStorage::F16(output) = CpuStorage::F32(input).to_dtype(&layout, DType::F16)? else {
        panic!("expected f16")
    };
    assert_eq!(
        output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected
    );
    Ok(())
}

#[test]
fn cpu_dtype_f32_to_f16_offsets_and_tails() -> Result<()> {
    let input: Vec<f32> = (0..4096).map(|i| (i as f32 - 64.) / 7.).collect();
    let storage = CpuStorage::F32(input.clone());
    for offset in [0, 1, 3, 7] {
        for len in [
            0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 2047,
            2048, 2049,
        ] {
            let layout = Layout::contiguous_with_offset(len, offset);
            let CpuStorage::F16(output) = storage.to_dtype(&layout, DType::F16)? else {
                panic!("expected f16")
            };
            let expected: Vec<_> = input[offset..offset + len]
                .iter()
                .map(|&v| f16::from_f32(v).to_bits())
                .collect();
            assert_eq!(
                output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected,
                "offset {offset}, length {len}"
            );
        }
    }
    Ok(())
}
