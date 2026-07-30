use super::*;
use crate::quantized::k_quants::{BlockQ4K, BlockQ4_0};
use crate::{Device, IndexOp, Module, Tensor};

/// `RocmDevice::new` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

/// The quantized kernels read whole `MATRIX_ROW_PADDING`-element rows, so the
/// allocation must outrun the payload by exactly the CUDA backend's amount —
/// a shorter buffer reads out of bounds, a differently shaped one desynchronises
/// the two backends.
#[test]
fn padding_matches_the_cuda_formula() {
    // Q4_0: 32 elements per 18-byte block -> 512 elements of padding are 16
    // blocks, i.e. 288 bytes.
    assert_eq!(padded_len(1000, GgmlDType::Q4_0), 1000 + 288);
    // Q4K: 256 elements per 144-byte block -> 512 elements are 2 blocks.
    assert_eq!(padded_len(1000, GgmlDType::Q4K), 1000 + 288);
    // f32 has a block size of one, so the padding is the full 512 elements.
    assert_eq!(padded_len(1000, GgmlDType::F32), 1000 + 2048);
}

#[test]
fn zeros_reports_the_unpadded_payload() -> Result<()> {
    let dev = rocm_device!();
    let el_count = 1024;
    let storage = QRocmStorage::zeros(&dev, el_count, GgmlDType::Q4_0)?;
    let blocks = el_count / GgmlDType::Q4_0.block_size();
    let expected = blocks * GgmlDType::Q4_0.type_size();
    assert_eq!(storage.storage_size_in_bytes(), expected);
    // `data` hands back the payload only, never the padding.
    let data = storage.data()?;
    assert_eq!(data.len(), expected);
    assert!(data.iter().all(|&b| b == 0));
    // ... while the allocation itself carries the padding.
    assert_eq!(
        storage.data.count(),
        (el_count + MATRIX_ROW_PADDING) / GgmlDType::Q4_0.block_size()
            * GgmlDType::Q4_0.type_size()
    );
    Ok(())
}

/// A round trip through `load_quantized` must leave the bytes untouched — the
/// upload clamps to the shorter of host and device buffer, which is only right
/// while the device buffer is the longer one.
#[test]
fn load_quantized_round_trips_the_payload() -> Result<()> {
    let dev = rocm_device!();
    let xs: Vec<f32> = (0..1024).map(|i| i as f32 / 128.).collect();
    let mut blocks = vec![BlockQ4_0::zeros(); xs.len() / 32];
    BlockQ4_0::from_float(&xs, &mut blocks);
    let expected = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            std::mem::size_of_val(blocks.as_slice()),
        )
    }
    .to_vec();

    let storage = load_quantized(&dev, &blocks)?;
    match &storage {
        QStorage::Rocm(s) => {
            assert_eq!(s.storage_size_in_bytes(), expected.len());
            assert_eq!(s.data()?, expected);
        }
        _ => crate::bail!("load_quantized did not produce a rocm storage"),
    }
    Ok(())
}

/// The device dequantize kernels have to agree with the reference CPU
/// implementation for every dtype that has one.
#[test]
fn dequantize_matches_cpu() -> Result<()> {
    let dev = rocm_device!();
    let el_count = 1024;
    let xs: Vec<f32> = (0..el_count)
        .map(|i| (i as f32 / 37.).sin() * 4.2)
        .collect();
    let cpu = Device::Cpu;
    let src = Tensor::from_slice(&xs, el_count, &cpu)?;

    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        let cpu_q = crate::quantized::QTensor::quantize(&src, dtype)?;
        let cpu_deq = cpu_q.dequantize(&cpu)?.to_vec1::<f32>()?;

        let bytes = cpu_q.data()?;
        let rocm_q =
            crate::quantized::QStorage::from_data(bytes, &Device::Rocm(dev.clone()), dtype)?;
        let rocm_deq = match rocm_q.dequantize(el_count)? {
            crate::Storage::Rocm(s) => s.to_cpu_storage()?.as_slice::<f32>()?.to_vec(),
            _ => crate::bail!("dequantize did not stay on the rocm device"),
        };

        assert_eq!(rocm_deq.len(), cpu_deq.len(), "{dtype:?}");
        for (i, (a, b)) in rocm_deq.iter().zip(cpu_deq.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.),
                "{dtype:?} element {i}: rocm {a} vs cpu {b}"
            );
        }
    }
    Ok(())
}

/// `dequantize_f16` has a dedicated kernel per dtype, but only for the dtypes
/// that also have an f32 one — the rest fall back through `dequantize`.
#[test]
fn dequantize_f16_covers_every_dtype() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev.clone());
    let el_count = 512;
    let xs: Vec<f32> = (0..el_count).map(|i| (i as f32 / 23.).sin() * 2.).collect();
    let src = Tensor::from_slice(&xs, el_count, &Device::Cpu)?;

    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q6K,
        GgmlDType::F32,
        GgmlDType::F16,
    ] {
        let qt = crate::quantized::QTensor::quantize_onto(&src, dtype, &device)?;
        let f32 = qt.dequantize(&device)?.to_vec1::<f32>()?;
        let f16 = qt.dequantize_f16(&device)?.to_vec1::<half::f16>()?;
        assert_eq!(f16.len(), f32.len(), "{dtype:?}");
        for (i, (a, b)) in f16.iter().zip(f32.iter()).enumerate() {
            assert!(
                (a.to_f32() - b).abs() <= 1e-2 * b.abs().max(1.),
                "{dtype:?} element {i}: f16 {a} vs f32 {b}"
            );
        }
    }
    Ok(())
}

/// `embedding` gathers rows through `get_rows`, and the ids may arrive as a
/// non-zero-offset view of a larger buffer (`Tensor::narrow`). The launcher has
/// to bias the ids pointer by that element offset.
#[test]
fn embedding_honours_the_ids_offset() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev.clone());
    let (rows, hidden) = (8usize, 256usize);
    let xs: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 / 91.).cos() * 3.)
        .collect();
    let src = Tensor::from_slice(&xs, (rows, hidden), &Device::Cpu)?;

    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        let qt = crate::quantized::QTensor::quantize_onto(&src, dtype, &device)?;
        let reference = qt.dequantize(&device)?;

        let all_ids = Tensor::new(&[7u32, 0, 5, 2, 6, 1], &device)?;
        // A narrowed view starts three elements into `all_ids`.
        let ids = all_ids.narrow(0, 3, 3)?;
        assert_eq!(ids.layout().start_offset(), 3);

        let got = qt.embedding(&ids)?.to_vec2::<f32>()?;
        let want: Vec<Vec<f32>> = [2usize, 6, 1]
            .iter()
            .map(|&r| reference.i(r)?.to_vec1::<f32>())
            .collect::<Result<_>>()?;
        assert_eq!(got, want, "{dtype:?}");
    }
    Ok(())
}

/// `fwd` dequantizes and defers to the rocBLAS GEMM; the result has to track
/// the same computation run on the CPU with the *same* dequantized weights.
#[test]
fn fwd_matches_a_dequantized_matmul() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev.clone());
    let (m, k, n) = (3usize, 256usize, 4usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
    // `QTensor` stores the weights transposed, i.e. `(n, k)`.
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

    let qt = crate::quantized::QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let want = lhs
        .matmul(&qt.dequantize(&device)?.t()?)?
        .to_vec2::<f32>()?;
    let got = crate::quantized::QMatMul::from_qtensor(qt)?
        .forward(&lhs)?
        .to_vec2::<f32>()?;

    for (row_got, row_want) in got.iter().zip(want.iter()) {
        for (a, b) in row_got.iter().zip(row_want.iter()) {
            assert!((a - b).abs() <= 1e-3, "{a} vs {b}");
        }
    }
    Ok(())
}

/// Every block type has to survive the host round trip in `dequantize`'s
/// fallback path, which reads the buffer unaligned.
#[test]
fn deq_reads_unaligned_buffers() -> Result<()> {
    let xs: Vec<f32> = (0..256).map(|i| i as f32 / 8.).collect();
    let mut blocks = vec![BlockQ4K::zeros(); 1];
    BlockQ4K::from_float(&xs, &mut blocks);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            std::mem::size_of_val(blocks.as_slice()),
        )
    };
    // Offset the payload by one byte so it cannot be naturally aligned.
    let mut shifted = vec![0u8];
    shifted.extend_from_slice(bytes);

    let mut want = vec![0f32; 256];
    BlockQ4K::to_float(&blocks, &mut want);
    let mut got = vec![0f32; 256];
    deq::<BlockQ4K>(&shifted[1..], 1, &mut got)?;
    assert_eq!(got, want);

    // A short buffer is an error, never an out-of-bounds read.
    assert!(deq::<BlockQ4K>(&bytes[..10], 1, &mut got).is_err());
    Ok(())
}
