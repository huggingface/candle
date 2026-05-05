use candle_core::{Device, Result, Tensor};

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
    let diff = (out - expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;
    assert!(
        max_diff < TOL,
        "Max difference {} exceeds threshold {}",
        max_diff,
        TOL
    );
    Ok(())
}

/* PyTorch reference (input is torch.arange(16).reshape(1,1,4,4)):
let y = F.interpolate(x, size=(2, 2), mode="bilinear", antialias=True, align_corners=False)
*/
#[test]
fn aa_2x_downscale_4_to_2() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(2, 2, false)?;
    let expected = Tensor::new(
        &[3.571429f32, 5.142858, 9.857143, 11.428572],
        dev,
    )?
    .reshape((1, 1, 2, 2))?;
    assert_close(&output, &expected)
}

/* arange(64).reshape(1,1,8,8) -> (4, 4) */
#[test]
fn aa_8x8_to_4x4() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let output = input.upsample_bilinear2d_antialias(4, 4, false)?;
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
    let output = input.upsample_bilinear2d_antialias(8, 8, false)?;
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
    let output = input.upsample_bilinear2d_antialias(5, 3, false)?;
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
    let output = input.upsample_bilinear2d_antialias(3, 4, false)?;
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
    let output = input.upsample_bilinear2d_antialias(8, 8, false)?;
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
    let output = input.upsample_bilinear2d_antialias(4, 4, false)?;
    let expected = input.clone();
    assert_close(&output, &expected)
}

/* arange(96).reshape(2,3,4,4) -> (2, 2). Multi-batch + multi-channel. */
#[test]
fn aa_multichannel_2x3x4x4_to_2x2() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 96f32, dev)?.reshape((2, 3, 4, 4))?;
    let output = input.upsample_bilinear2d_antialias(2, 2, false)?;
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
    let aa = input.upsample_bilinear2d_antialias(8, 8, false)?;
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

/* align_corners=true must error rather than silently producing a wrong result. */
#[test]
fn aa_align_corners_true_returns_error() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let result = input.upsample_bilinear2d_antialias(2, 2, true);
    assert!(result.is_err(), "align_corners=true should error");
    Ok(())
}
