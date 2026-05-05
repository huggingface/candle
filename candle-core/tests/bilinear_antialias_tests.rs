// This test file pastes PyTorch reference values at the precision PyTorch
// printed (6 decimals). For values >= ~100 that exceeds the f32 mantissa, so
// clippy::excessive_precision would flag ~130 of these literals. The trailing
// digits round to the same f32 bit pattern (verified by every test passing
// under the 1e-4 tolerance gate); we accept the literals as-is to keep the
// fixtures byte-identical to the upstream PyTorch print output. The allow is
// scoped to this fixture-only test file.
#![allow(clippy::excessive_precision)]

use candle_core::{DType, Device, Result, Tensor};

// ============================================================================
// PyTorch Exact Comparison Tests
// ============================================================================
// These tests compare against exact PyTorch outputs from
// F.interpolate(..., mode='bilinear', align_corners=False, antialias=True)
// to ensure correctness of upsample_bilinear2d_antialias.
//
// CPU-only: Metal/CUDA backends bail on this op (kernels are a follow-up PR);
// we therefore do not use the test_device! macro here.

const TOL: f32 = 1e-4;

fn assert_close(out: &Tensor, expected: &Tensor) -> Result<()> {
    let diff_t = (out - expected)?.abs()?;
    let max_diff = diff_t.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    if max_diff >= TOL {
        let out_v = out.flatten_all()?.to_vec1::<f32>()?;
        let exp_v = expected.flatten_all()?.to_vec1::<f32>()?;
        let (idx, _) = out_v
            .iter()
            .zip(exp_v.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        panic!(
            "max abs diff {} >= tolerance {} at flat index {}: got {}, expected {}, shape {:?}",
            max_diff,
            TOL,
            idx,
            out_v[idx],
            exp_v[idx],
            out.shape()
        );
    }
    Ok(())
}

/* PyTorch reference (input is torch.arange(16).reshape(1,1,4,4)):
let y = F.interpolate(x, size=(2, 2), mode="bilinear", antialias=True, align_corners=False)
*/
#[test]
fn aa_2x_downscale_4_to_2() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(2, 2)?;
    let expected =
        Tensor::new(&[3.571429f32, 5.142858, 9.857143, 11.428572], dev)?.reshape((1, 1, 2, 2))?;
    assert_close(&output, &expected)
}

/* arange(64).reshape(1,1,8,8) -> (4, 4) */
#[test]
fn aa_8x8_to_4x4() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let output = input.upsample_bilinear2d_antialias(4, 4)?;
    let expected = Tensor::new(
        &[
            6.428572f32,
            8.214286,
            10.214286,
            12.000001,
            20.714287,
            22.500000,
            24.500000,
            26.285713,
            36.714287,
            38.500000,
            40.500000,
            42.285713,
            51.000000,
            52.785717,
            54.785713,
            56.571430,
        ],
        dev,
    )?
    .reshape((1, 1, 4, 4))?;
    assert_close(&output, &expected)
}

/* arange(256).reshape(1,1,16,16) -> (8, 8). Mirrors a SigLIP2-style halve. */
#[test]
fn aa_16x16_to_8x8() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 256f32, dev)?.reshape((1, 1, 16, 16))?;
    let output = input.upsample_bilinear2d_antialias(8, 8)?;
    let expected = Tensor::new(
        &[
            12.142859f32,
            13.928572,
            15.928572,
            17.928572,
            19.928572,
            21.928572,
            23.928572,
            25.714287,
            40.714287,
            42.500000,
            44.500000,
            46.500000,
            48.500000,
            50.500000,
            52.500000,
            54.285713,
            72.714287,
            74.500000,
            76.500000,
            78.500000,
            80.500000,
            82.500000,
            84.500000,
            86.285713,
            104.714287,
            106.500000,
            108.500000,
            110.500000,
            112.500000,
            114.500000,
            116.500000,
            118.285713,
            136.714294,
            138.500000,
            140.500000,
            142.500000,
            144.500000,
            146.500000,
            148.500000,
            150.285721,
            168.714294,
            170.500000,
            172.500000,
            174.500000,
            176.500000,
            178.500000,
            180.500000,
            182.285721,
            200.714294,
            202.500000,
            204.500000,
            206.500000,
            208.500000,
            210.500000,
            212.500000,
            214.285721,
            229.285736,
            231.071426,
            233.071442,
            235.071426,
            237.071426,
            239.071442,
            241.071426,
            242.857162,
        ],
        dev,
    )?
    .reshape((1, 1, 8, 8))?;
    assert_close(&output, &expected)
}

/* arange(64).reshape(1,1,8,8) -> (5, 3). Asymmetric, non-integer ratios. */
#[test]
fn aa_asymmetric_8x8_to_5x3() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let output = input.upsample_bilinear2d_antialias(5, 3)?;
    let expected = Tensor::new(
        &[
            4.377990f32,
            6.772727,
            9.167464,
            16.512671,
            18.907410,
            21.302145,
            29.105263,
            31.500002,
            33.894737,
            41.697857,
            44.092594,
            46.487331,
            53.832539,
            56.227276,
            58.622013,
        ],
        dev,
    )?
    .reshape((1, 1, 5, 3))?;
    assert_close(&output, &expected)
}

/* arange(48).reshape(1,2,4,6) -> (3, 4). Multi-channel non-square. */
#[test]
fn aa_non_square_4x6_to_3x4() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 48f32, dev)?.reshape((1, 2, 4, 6))?;
    let output = input.upsample_bilinear2d_antialias(3, 4)?;
    let expected = Tensor::new(
        &[
            2.175000f32,
            3.577778,
            5.022222,
            6.425000,
            9.375000,
            10.777779,
            12.222221,
            13.625000,
            16.575001,
            17.977781,
            19.422222,
            20.825001,
            26.175001,
            27.577780,
            29.022223,
            30.425001,
            33.375000,
            34.777779,
            36.222221,
            37.625000,
            40.575001,
            41.977779,
            43.422218,
            44.825001,
        ],
        dev,
    )?
    .reshape((1, 2, 3, 4))?;
    assert_close(&output, &expected)
}

/* arange(16).reshape(1,1,4,4) -> (8, 8). Upscaling: antialias is a no-op,
so output should equal the non-antialiased bilinear reference. */
#[test]
fn aa_upscale_4x4_to_8x8() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(8, 8)?;
    let expected = Tensor::new(
        &[
            0.000000f32,
            0.250000,
            0.750000,
            1.250000,
            1.750000,
            2.250000,
            2.750000,
            3.000000,
            1.000000,
            1.250000,
            1.750000,
            2.250000,
            2.750000,
            3.250000,
            3.750000,
            4.000000,
            3.000000,
            3.250000,
            3.750000,
            4.250000,
            4.750000,
            5.250000,
            5.750000,
            6.000000,
            5.000000,
            5.250000,
            5.750000,
            6.250000,
            6.750000,
            7.250000,
            7.750000,
            8.000000,
            7.000000,
            7.250000,
            7.750000,
            8.250000,
            8.750000,
            9.250000,
            9.750000,
            10.000000,
            9.000000,
            9.250000,
            9.750000,
            10.250000,
            10.750000,
            11.250000,
            11.750000,
            12.000000,
            11.000000,
            11.250000,
            11.750000,
            12.250000,
            12.750000,
            13.250000,
            13.750000,
            14.000000,
            12.000000,
            12.250000,
            12.750000,
            13.250000,
            13.750000,
            14.250000,
            14.750000,
            15.000000,
        ],
        dev,
    )?
    .reshape((1, 1, 8, 8))?;
    assert_close(&output, &expected)
}

/* arange(16).reshape(1,1,4,4) -> (4, 4). Identity case must round-trip exactly. */
#[test]
fn aa_identity_4x4_to_4x4() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(4, 4)?;
    let expected = input.clone();
    assert_close(&output, &expected)
}

/* arange(96).reshape(2,3,4,4) -> (2, 2). Multi-batch + multi-channel. */
#[test]
fn aa_multichannel_2x3x4x4_to_2x2() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 96f32, dev)?.reshape((2, 3, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(2, 2)?;
    let expected = Tensor::new(
        &[
            3.571429f32,
            5.142858,
            9.857143,
            11.428572,
            19.571430,
            21.142859,
            25.857143,
            27.428572,
            35.571430,
            37.142857,
            41.857143,
            43.428574,
            51.571430,
            53.142857,
            57.857143,
            59.428574,
            67.571434,
            69.142860,
            73.857147,
            75.428574,
            83.571434,
            85.142853,
            89.857147,
            91.428574,
        ],
        dev,
    )?
    .reshape((2, 3, 2, 2))?;
    assert_close(&output, &expected)
}

/* Antialias must change the output for downsampling: regression guard
against silently routing to the non-antialiased path. */
#[test]
fn aa_differs_from_non_antialiased_on_downscale() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 256f32, dev)?.reshape((1, 1, 16, 16))?;
    let aa = input.upsample_bilinear2d_antialias(8, 8)?;
    let plain = input.upsample_bilinear2d(8, 8, false)?;
    let diff = (&aa - &plain)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;
    assert!(
        max_diff > 1e-3,
        "Antialiased output should differ from plain bilinear on downscale; \
         max diff was {} (likely fell through to non-AA path)",
        max_diff
    );
    Ok(())
}

/* Non-contiguous input via permute + AA output should equal the AA-of-
original output then permuted the same way. This is stronger than the
weaker "matches contiguous" check because if the kernel had a bug where
stride[2] and stride[3] were swapped, both permuted and permuted.contiguous()
would return wrong (but matching) results. By comparing against the
independently-computed AA on the original tensor (transposed afterwards),
we catch stride swaps. */
#[test]
fn aa_non_contiguous_via_permute() -> Result<()> {
    let dev = &Device::Cpu;
    let raw = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    // AA on the original H,W layout: produces (1,1,4,4) output with axes
    // matching raw's axis order.
    let raw_aa = raw.upsample_bilinear2d_antialias(4, 4)?;
    // Now permute raw to swap H<->W (non-contiguous) and run AA on that.
    // The output should equal raw_aa with the same H<->W swap applied,
    // because AA is axis-symmetric in this respect.
    let permuted = raw.permute((0, 1, 3, 2))?;
    assert!(!permuted.is_contiguous(), "permuted must be non-contiguous");
    let permuted_aa = permuted.upsample_bilinear2d_antialias(4, 4)?;
    let expected = raw_aa.permute((0, 1, 3, 2))?.contiguous()?;
    assert_close(&permuted_aa, &expected)
}

/* Regression guard against re-introducing a stride-unaware identity
short-circuit (a previous revision had one that called src.to_vec()
ignoring layout, silently reinterpreting permuted bytes as contiguous).
With target == input shape and a non-contiguous input, the output must
reflect the logical (permuted) values, not the underlying contiguous
storage. */
#[test]
fn aa_identity_with_non_contiguous_input() -> Result<()> {
    let dev = &Device::Cpu;
    let raw = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let permuted = raw.permute((0, 1, 3, 2))?;
    assert!(!permuted.is_contiguous());
    let out = permuted.upsample_bilinear2d_antialias(8, 8)?;
    let expected = permuted.contiguous()?;
    assert_close(&out, &expected)
}

/* Sliced input via narrow produces a non-zero src_offset; the kernel must
honor layout.start_offset() and not assume the data starts at index 0
of the underlying buffer. */
#[test]
fn aa_narrowed_input_honors_offset() -> Result<()> {
    let dev = &Device::Cpu;
    // Build a wider buffer, take a (1,1,8,8) slice that starts in the middle.
    let big = Tensor::arange(0f32, 256f32, dev)?.reshape((1, 1, 16, 16))?;
    let slice = big.narrow(2, 4, 8)?.narrow(3, 4, 8)?; // 8x8 from rows 4..12, cols 4..12
    let slice_aa = slice.upsample_bilinear2d_antialias(4, 4)?;
    // Reference: copy the same slice contiguously and run AA on that.
    let slice_contig_aa = slice.contiguous()?.upsample_bilinear2d_antialias(4, 4)?;
    assert_close(&slice_aa, &slice_contig_aa)
}

/* Broadcast batch (stride[0] == 0) means every batch reads the same source.
The kernel computes base_idx = src_offset + b * stride[0] + c * stride[1];
with stride[0] = 0, all batches share the same row 0 of the underlying
data, so all batch-slices of the output should be identical. */
#[test]
fn aa_broadcast_batch_produces_identical_outputs() -> Result<()> {
    let dev = &Device::Cpu;
    let single = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let expanded = single.broadcast_as((4, 1, 8, 8))?; // stride[0] = 0
    let out = expanded.upsample_bilinear2d_antialias(4, 4)?;
    // Reference: AA on the single-batch input.
    let ref_out = single.upsample_bilinear2d_antialias(4, 4)?;
    let ref_v = ref_out.flatten_all()?.to_vec1::<f32>()?;
    let out_v = out.flatten_all()?.to_vec1::<f32>()?;
    let per_batch = ref_v.len();
    assert_eq!(out_v.len(), 4 * per_batch);
    for b in 0..4 {
        for i in 0..per_batch {
            let diff = (out_v[b * per_batch + i] - ref_v[i]).abs();
            assert!(
                diff < 1e-4,
                "batch {b} index {i} broadcast mismatch: {} vs {} (diff {})",
                out_v[b * per_batch + i],
                ref_v[i],
                diff
            );
        }
    }
    Ok(())
}

/* The kernel is generic over WithDType. f32 is exercised above; this test
covers the f64 instantiation. f16 / bf16 paths are not exercised because
PyTorch's antialiased-bilinear kernel runs in fp32 internally and casting
the reference values down would compare two slightly different algorithms,
not the same one at lower precision. */
#[test]
fn aa_f64_dtype_matches_f32_within_tolerance() -> Result<()> {
    let dev = &Device::Cpu;
    let input_f32 = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let input_f64 = input_f32.to_dtype(DType::F64)?;
    let out_f32 = input_f32.upsample_bilinear2d_antialias(4, 4)?;
    let out_f64 = input_f64.upsample_bilinear2d_antialias(4, 4)?;
    assert_eq!(
        out_f64.dtype(),
        DType::F64,
        "f64 input must produce f64 output"
    );
    let out_f64_as_f32 = out_f64.to_dtype(DType::F32)?;
    let diff = (&out_f32 - &out_f64_as_f32)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;
    assert!(
        max_diff < 1e-5,
        "f32 vs f64 outputs should match within fp32 noise floor; got {}",
        max_diff
    );
    Ok(())
}

/* Round-trip dtype coverage for f16 and bf16. We do NOT compare to PyTorch
here because PyTorch's antialiased-bilinear path runs in fp32 internally,
so a direct comparison would conflate "Rust kernel correctness" with
"Rust dtype handling matches PyTorch dtype handling." Instead: cast f32
input to the lower-precision dtype, run the kernel, cast output back,
and assert it matches the f32 result within the lower precision's noise
floor. This validates the WithDType generic instantiation produces
sensible output, which is the correctness claim a maintainer cares about. */
#[test]
fn aa_f16_round_trip_within_noise_floor() -> Result<()> {
    let dev = &Device::Cpu;
    let input_f32 = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let input_f16 = input_f32.to_dtype(DType::F16)?;
    let out_f32 = input_f32.upsample_bilinear2d_antialias(4, 4)?;
    let out_f16 = input_f16.upsample_bilinear2d_antialias(4, 4)?;
    assert_eq!(out_f16.dtype(), DType::F16);
    let out_f16_as_f32 = out_f16.to_dtype(DType::F32)?;
    let diff = (&out_f32 - &out_f16_as_f32)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;
    // f16 ULP at output magnitude ~57 is ~57 * 2^-10 ≈ 0.056. The kernel
    // accumulates in f64 regardless of T, so the only loss is input-cast
    // + output-cast: bounded by ~1 ULP at output magnitude. Threshold 5e-2
    // is ~1 ULP at this test's value range, tight enough to catch
    // regressions that would cross into 2-ULP territory.
    assert!(
        max_diff < 5e-2,
        "f32 vs f16 round-trip should match within ~1 f16 ULP; got {}",
        max_diff
    );
    Ok(())
}

#[test]
fn aa_bf16_round_trip_within_noise_floor() -> Result<()> {
    let dev = &Device::Cpu;
    let input_f32 = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let input_bf16 = input_f32.to_dtype(DType::BF16)?;
    let out_f32 = input_f32.upsample_bilinear2d_antialias(4, 4)?;
    let out_bf16 = input_bf16.upsample_bilinear2d_antialias(4, 4)?;
    assert_eq!(out_bf16.dtype(), DType::BF16);
    let out_bf16_as_f32 = out_bf16.to_dtype(DType::F32)?;
    let diff = (&out_f32 - &out_bf16_as_f32)?
        .abs()?
        .flatten_all()?
        .max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;
    // bf16 ULP at output magnitude ~57 is ~57 * 2^-7 ≈ 0.45. Same f64
    // accumulator argument as the f16 test: the only loss is dtype cast at
    // boundaries. Threshold 5e-1 is ~1 ULP at this test's value range,
    // tight enough to catch regressions that would cross 2-ULP.
    assert!(
        max_diff < 5e-1,
        "f32 vs bf16 round-trip should match within ~1 bf16 ULP; got {}",
        max_diff
    );
    Ok(())
}

/* Zero-target-dim must error rather than producing an empty tensor silently. */
#[test]
fn aa_zero_target_dim_errors() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    assert!(
        input.upsample_bilinear2d_antialias(0, 4).is_err(),
        "target_h=0 should error"
    );
    assert!(
        input.upsample_bilinear2d_antialias(4, 0).is_err(),
        "target_w=0 should error"
    );
    Ok(())
}
