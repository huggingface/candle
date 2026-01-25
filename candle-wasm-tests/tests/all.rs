pub mod bilinear_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, Device, IndexOp, Result, Tensor};
async fn bilinear_pytorch_2x_upscale(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bilinear2d(8, 8, false)?;
    let expected = Tensor::new(
            &[
                0.0000f32,
                0.2500,
                0.7500,
                1.2500,
                1.7500,
                2.2500,
                2.7500,
                3.0000,
                1.0000,
                1.2500,
                1.7500,
                2.2500,
                2.7500,
                3.2500,
                3.7500,
                4.0000,
                3.0000,
                3.2500,
                3.7500,
                4.2500,
                4.7500,
                5.2500,
                5.7500,
                6.0000,
                5.0000,
                5.2500,
                5.7500,
                6.2500,
                6.7500,
                7.2500,
                7.7500,
                8.0000,
                7.0000,
                7.2500,
                7.7500,
                8.2500,
                8.7500,
                9.2500,
                9.7500,
                10.0000,
                9.0000,
                9.2500,
                9.7500,
                10.2500,
                10.7500,
                11.2500,
                11.7500,
                12.0000,
                11.0000,
                11.2500,
                11.7500,
                12.2500,
                12.7500,
                13.2500,
                13.7500,
                14.0000,
                12.0000,
                12.2500,
                12.7500,
                13.2500,
                13.7500,
                14.2500,
                14.7500,
                15.0000,
            ],
            dev,
        )?
        .reshape((1, 1, 8, 8))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-4, "Max difference {} exceeds threshold 1e-4", max_diff);
    Ok(())
}
async fn bilinear_pytorch_downscale(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let output = input.upsample_bilinear2d(4, 4, false)?;
    let expected = Tensor::new(
            &[
                4.5f32,
                6.5,
                8.5,
                10.5,
                20.5,
                22.5,
                24.5,
                26.5,
                36.5,
                38.5,
                40.5,
                42.5,
                52.5,
                54.5,
                56.5,
                58.5,
            ],
            dev,
        )?
        .reshape((1, 1, 4, 4))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-4, "Max difference {} exceeds threshold 1e-4", max_diff);
    Ok(())
}
async fn bilinear_pytorch_multi_channel(dev: &Device) -> Result<()> {
    let input = Tensor::new(
            &[
                1.9269f32,
                1.4873,
                0.9007,
                -2.1055,
                0.6784,
                -1.2345,
                -0.0431,
                -1.6047,
                -0.7521,
                1.6487,
                -0.3925,
                -1.4036,
                -0.7279,
                -0.5594,
                -0.7688,
                0.7624,
                1.6423f32,
                -0.1596,
                -0.4974,
                0.4396,
                -0.7581,
                1.0783,
                0.8008,
                1.6806,
                1.2791,
                1.2964,
                0.6105,
                1.3347,
                -0.2316,
                0.0418,
                -0.2516,
                0.8599,
            ],
            dev,
        )?
        .reshape((1, 2, 4, 4))?;
    let output = input.upsample_bilinear2d(8, 8, false)?;
    assert_eq!(output.dims(), & [1, 2, 8, 8]);
    let output_vec = output.flatten_all()?.to_vec1_async::<f32>().await?;
    for &val in &output_vec {
        assert!(val.is_finite(), "Output contains non-finite value");
    }
    let output_ch0_row0 = output.i((0, 0, 0, ..))?.to_vec1_async::<f32>().await?;
    let expected_ch0_row0 = [
        1.9269f32,
        1.8170,
        1.5972,
        1.3406,
        1.0474,
        0.1492,
        -1.3540,
        -2.1055,
    ];
    for (i, (&out, &exp)) in output_ch0_row0
        .iter()
        .zip(expected_ch0_row0.iter())
        .enumerate()
    {
        let diff = (out - exp).abs();
        assert!(
            diff < 1e-3,
            "Channel 0, row 0, index {} differs: got {}, expected {}, diff {}", i, out,
            exp, diff
        );
    }
    let output_ch1_row0 = output.i((0, 1, 0, ..))?.to_vec1_async::<f32>().await?;
    let expected_ch1_row0 = [
        1.6423f32,
        1.1918,
        0.2909,
        -0.2440,
        -0.4129,
        -0.2632,
        0.2053,
        0.4396,
    ];
    for (i, (&out, &exp)) in output_ch1_row0
        .iter()
        .zip(expected_ch1_row0.iter())
        .enumerate()
    {
        let diff = (out - exp).abs();
        assert!(
            diff < 1e-3,
            "Channel 1, row 0, index {} differs: got {}, expected {}, diff {}", i, out,
            exp, diff
        );
    }
    Ok(())
}
async fn bilinear_pytorch_align_corners_true(dev: &Device) -> Result<()> {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), dev)?;
    let output = input.upsample_bilinear2d(4, 4, true)?;
    let expected = Tensor::new(
            &[
                1.0f32,
                1.3333,
                1.6667,
                2.0,
                1.6667,
                2.0,
                2.3333,
                2.6667,
                2.3333,
                2.6667,
                3.0,
                3.3333,
                3.0,
                3.3333,
                3.6667,
                4.0,
            ],
            dev,
        )?
        .reshape((1, 1, 4, 4))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-3, "Max difference {} exceeds threshold 1e-3", max_diff);
    let output_vec = output.flatten_all()?.to_vec1_async::<f32>().await?;
    assert!((output_vec[0] - 1.0).abs() < 1e-5, "Top-left corner not preserved");
    assert!((output_vec[3] - 2.0).abs() < 1e-5, "Top-right corner not preserved");
    assert!((output_vec[12] - 3.0).abs() < 1e-5, "Bottom-left corner not preserved");
    assert!((output_vec[15] - 4.0).abs() < 1e-5, "Bottom-right corner not preserved");
    Ok(())
}
async fn bilinear_pytorch_scale_factor(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output_scale = input.upsample_bilinear2d_with_scale(2.0, 2.0, false)?;
    let output_size = input.upsample_bilinear2d(8, 8, false)?;
    let diff = (&output_scale - &output_size)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-6, "scale_factor and size methods differ by {}", max_diff);
    Ok(())
}
async fn bilinear_pytorch_non_square_exact(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 24f32, dev)?.reshape((1, 1, 4, 6))?;
    let output = input.upsample_bilinear2d(8, 12, false)?;
    #[rustfmt::skip]
    let expected = Tensor::new(
            &[
                0.0f32,
                0.25,
                0.75,
                1.25,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.0,
                1.5,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                6.25,
                6.5,
                4.5,
                4.75,
                5.25,
                5.75,
                6.25,
                6.75,
                7.25,
                7.75,
                8.25,
                8.75,
                9.25,
                9.5,
                7.5,
                7.75,
                8.25,
                8.75,
                9.25,
                9.75,
                10.25,
                10.75,
                11.25,
                11.75,
                12.25,
                12.5,
                10.5,
                10.75,
                11.25,
                11.75,
                12.25,
                12.75,
                13.25,
                13.75,
                14.25,
                14.75,
                15.25,
                15.5,
                13.5,
                13.75,
                14.25,
                14.75,
                15.25,
                15.75,
                16.25,
                16.75,
                17.25,
                17.75,
                18.25,
                18.5,
                16.5,
                16.75,
                17.25,
                17.75,
                18.25,
                18.75,
                19.25,
                19.75,
                20.25,
                20.75,
                21.25,
                21.5,
                18.0,
                18.25,
                18.75,
                19.25,
                19.75,
                20.25,
                20.75,
                21.25,
                21.75,
                22.25,
                22.75,
                23.0,
            ],
            dev,
        )?
        .reshape((1, 1, 8, 12))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-4, "Max difference {} exceeds threshold 1e-4", max_diff);
    Ok(())
}
async fn bilinear_pytorch_tiny_1x1_to_3x3(dev: &Device) -> Result<()> {
    let input = Tensor::new(&[5.0f32], dev)?.reshape((1, 1, 1, 1))?;
    let output = input.upsample_bilinear2d(3, 3, false)?;
    let expected = Tensor::new(&[5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dev)?
        .reshape((1, 1, 3, 3))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-6, "Max difference {} exceeds threshold 1e-6", max_diff);
    Ok(())
}
async fn bilinear_pytorch_tiny_1x2_to_3x6(dev: &Device) -> Result<()> {
    let input = Tensor::new(&[2.0f32, 8.0], dev)?.reshape((1, 1, 1, 2))?;
    let output = input.upsample_bilinear2d(3, 6, false)?;
    #[rustfmt::skip]
    let expected = Tensor::new(
            &[
                2.0f32,
                2.0,
                4.0,
                6.0,
                8.0,
                8.0,
                2.0,
                2.0,
                4.0,
                6.0,
                8.0,
                8.0,
                2.0,
                2.0,
                4.0,
                6.0,
                8.0,
                8.0,
            ],
            dev,
        )?
        .reshape((1, 1, 3, 6))?;
    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0_async::<f32>().await?;
    assert!(max_diff < 1e-6, "Max difference {} exceeds threshold 1e-6", max_diff);
    Ok(())
}
async fn bilinear_pytorch_large_64x64_to_128x128(dev: &Device) -> Result<()> {
    use candle::DType;
    let input = Tensor::randn(0f32, 1f32, (1, 1, 64, 64), dev)?;
    let output = input.upsample_bilinear2d(128, 128, false)?;
    assert_eq!(output.dims(), & [1, 1, 128, 128]);
    assert_eq!(output.dtype(), DType::F32);
    let output_vec = output.flatten_all()?.to_vec1_async::<f32>().await?;
    for &val in &output_vec {
        assert!(val.is_finite(), "Large tensor output contains non-finite value");
    }
    let min_val = output_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = output_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_val > - 10.0 && max_val < 10.0,
        "Large tensor output values out of expected range: min={}, max={}", min_val,
        max_val
    );
    Ok(())
}
async fn bilinear_output_dimensions(dev: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 32f32, dev)?.reshape((1, 1, 4, 8))?;
    let out1 = t1.upsample_bilinear2d(6, 12, false)?;
    assert_eq!(out1.dims(), & [1, 1, 6, 12], "Non-square upscale failed");
    let t2 = Tensor::arange(0f32, 192f32, dev)?.reshape((4, 3, 4, 4))?;
    let out2 = t2.upsample_bilinear2d(8, 8, false)?;
    assert_eq!(out2.dims(), & [4, 3, 8, 8], "Batch processing failed");
    let t3 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out3 = t3.upsample_bilinear2d_with_scale(2.0, 3.0, false)?;
    assert_eq!(out3.dims(), & [1, 1, 8, 12], "Asymmetric scale failed");
    let t4 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out4 = t4.upsample_bilinear2d_with_scale(1.5, 1.5, false)?;
    assert_eq!(out4.dims(), & [1, 1, 6, 6], "Fractional scale failed");
    let t5 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out5 = t5.upsample_bilinear2d(1, 1, false)?;
    assert_eq!(out5.dims(), & [1, 1, 1, 1], "Single pixel output failed");
    let val = out5.flatten_all()?.to_vec1_async::<f32>().await?[0];
    assert!(val.is_finite(), "Single pixel value is not finite");
    let t6 = Tensor::arange(0f32, 4f32, dev)?.reshape((1, 1, 2, 2))?;
    let out6 = t6.upsample_bilinear2d_with_scale(5.0, 5.0, false)?;
    assert_eq!(out6.dims(), & [1, 1, 10, 10], "Large scale factor failed");
    Ok(())
}
async fn bilinear_identity(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bilinear2d(4, 4, false)?;
    let diff = (&t - &output)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0_async::< f32 > (). await ? < 1e-6);
    Ok(())
}
async fn bilinear_align_corners_difference(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output_false = t.upsample_bilinear2d(8, 8, false)?;
    let output_true = t.upsample_bilinear2d(8, 8, true)?;
    let diff = (&output_false - &output_true)?.abs()?.sum_all()?;
    assert!(diff.to_vec0_async::< f32 > (). await ? > 0.1);
    Ok(())
}
candle_wasm_tests::test_device!(
    bilinear_pytorch_2x_upscale, bilinear_pytorch_2x_upscale_cpu,
    bilinear_pytorch_2x_upscale_gpu, bilinear_pytorch_2x_upscale_metal,
    bilinear_pytorch_2x_upscale_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_downscale, bilinear_pytorch_downscale_cpu,
    bilinear_pytorch_downscale_gpu, bilinear_pytorch_downscale_metal,
    bilinear_pytorch_downscale_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_multi_channel, bilinear_pytorch_multi_channel_cpu,
    bilinear_pytorch_multi_channel_gpu, bilinear_pytorch_multi_channel_metal,
    bilinear_pytorch_multi_channel_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_align_corners_true, bilinear_pytorch_align_corners_true_cpu,
    bilinear_pytorch_align_corners_true_gpu, bilinear_pytorch_align_corners_true_metal,
    bilinear_pytorch_align_corners_true_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_scale_factor, bilinear_pytorch_scale_factor_cpu,
    bilinear_pytorch_scale_factor_gpu, bilinear_pytorch_scale_factor_metal,
    bilinear_pytorch_scale_factor_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_non_square_exact, bilinear_pytorch_non_square_exact_cpu,
    bilinear_pytorch_non_square_exact_gpu, bilinear_pytorch_non_square_exact_metal,
    bilinear_pytorch_non_square_exact_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_tiny_1x1_to_3x3, bilinear_pytorch_tiny_1x1_to_3x3_cpu,
    bilinear_pytorch_tiny_1x1_to_3x3_gpu, bilinear_pytorch_tiny_1x1_to_3x3_metal,
    bilinear_pytorch_tiny_1x1_to_3x3_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_tiny_1x2_to_3x6, bilinear_pytorch_tiny_1x2_to_3x6_cpu,
    bilinear_pytorch_tiny_1x2_to_3x6_gpu, bilinear_pytorch_tiny_1x2_to_3x6_metal,
    bilinear_pytorch_tiny_1x2_to_3x6_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_pytorch_large_64x64_to_128x128, bilinear_pytorch_large_64x64_to_128x128_cpu,
    bilinear_pytorch_large_64x64_to_128x128_gpu,
    bilinear_pytorch_large_64x64_to_128x128_metal,
    bilinear_pytorch_large_64x64_to_128x128_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_output_dimensions, bilinear_output_dimensions_cpu,
    bilinear_output_dimensions_gpu, bilinear_output_dimensions_metal,
    bilinear_output_dimensions_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_identity, bilinear_identity_cpu, bilinear_identity_gpu,
    bilinear_identity_metal, bilinear_identity_wgpu
);
candle_wasm_tests::test_device!(
    bilinear_align_corners_difference, bilinear_align_corners_difference_cpu,
    bilinear_align_corners_difference_gpu, bilinear_align_corners_difference_metal,
    bilinear_align_corners_difference_wgpu
);
}pub mod convert_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::{Ok, Result};
use candle::{test_device, Device, Tensor};
async fn convert(device: &Device) -> Result<()> {
    let vf32 = Tensor::arange(0f32, 4f32, device)?;
    let vf32_u32: Vec<u32> = vf32.to_dtype(candle::DType::U32)?.to_vec1_async().await?;
    assert_eq!(vf32_u32, [0u32, 1u32, 2u32, 3u32]);
    let vu32 = Tensor::new(vf32_u32, device)?;
    let vu32_f32: Vec<f32> = vu32.to_dtype(candle::DType::F32)?.to_vec1_async().await?;
    assert_eq!(vu32_f32, [0f32, 1f32, 2f32, 3f32]);
    let vu32_u8: Vec<u8> = vu32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vu32_u8, [0, 1, 2, 3]);
    let vf32_u8: Vec<u8> = vf32.to_dtype(candle::DType::U8)?.to_vec1_async().await?;
    assert_eq!(vf32_u8, [0, 1, 2, 3]);
    let vu8 = vu32.to_dtype(candle::DType::U8)?;
    let vu8_f32: Vec<f32> = vu8.to_dtype(candle::DType::F32)?.to_vec1_async().await?;
    assert_eq!(vu8_f32, [0f32, 1f32, 2f32, 3f32]);
    Ok(())
}
async fn alloc(device: &Device) -> Result<()> {
    let t = 5.0f64;
    let ratio = (Tensor::ones(1, candle::DType::F32, device)? * t)?;
    assert_eq!(ratio.to_vec1_async::< f32 > (). await ?, [5f32]);
    let ratio = (Tensor::ones(1, candle::DType::U32, device)? * t)?;
    assert_eq!(ratio.to_vec1_async::< u32 > (). await ?, [5u32]);
    Ok(())
}
candle_wasm_tests::test_device!(
    convert, convert_cpu, convert_gpu, convert_metal, convert_wgpu
);
candle_wasm_tests::test_device!(alloc, alloc_cpu, alloc_gpu, alloc_metal, alloc_wgpu);
}pub mod conv_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{test_device, test_utils, Device, IndexOp, Tensor};
async fn conv1d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
            &[
                0.4056f32,
                -0.8689,
                -0.0773,
                -1.5630,
                1.2279,
                -0.9287,
                -1.7030,
                0.1370,
                0.1866,
                0.4145,
                1.8025,
                -0.1536,
                2.2013,
                -0.6836,
                0.2477,
                1.3127,
                -0.6957,
                0.3278,
                -1.0124,
                0.5599,
            ],
            dev,
        )?
        .reshape((1, 4, 5))?;
    let w = Tensor::new(
            &[
                -0.8404f32,
                -0.3490,
                0.0130,
                1.3123,
                0.1763,
                -1.9249,
                1.4270,
                0.9421,
                0.8670,
                -0.7181,
                -1.1111,
                0.8869,
                -1.2429,
                1.8357,
                1.6052,
                -1.3844,
                0.3951,
                -1.2036,
                0.6686,
                1.6261,
                -0.6451,
                -0.0840,
                -1.4247,
                0.5512,
            ],
            dev,
        )?
        .reshape((2, 4, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [2.6357, - 1.3336,
        4.1393, - 1.1784, 3.5675, 0.5069]
    );
    let res = t.conv1d(&w, 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 5]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [2.4509, 2.6357, -
        1.3336, 4.1393, 0.5657, 1.8091, - 1.1784, 3.5675, 0.5069, 3.3352]
    );
    let res = {
        let t = Tensor::cat(&[&t.zeros_like()?, &t, &t.zeros_like()?], 0)?;
        t.conv1d(&w, 1, 1, 1, 1)?
    };
    assert_eq!(res.dims(), [3, 2, 5]);
    assert_eq!(
        to_vec1_round_async(& res.i(0) ?.flatten_all() ?, 4). await ?, [0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.]
    );
    assert_eq!(
        to_vec1_round_async(& res.i(1) ?.flatten_all() ?, 4). await ?, [2.4509, 2.6357, -
        1.3336, 4.1393, 0.5657, 1.8091, - 1.1784, 3.5675, 0.5069, 3.3352]
    );
    let w = w.transpose(0, 1)?;
    for w in [w.clone(), w.contiguous()?] {
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 1)?;
        assert_eq!(res.dims(), [1, 2, 7]);
        assert_eq!(
            to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.0699, - 1.2899,
            8.3018, 5.5873, 2.4572, - 2.6143, - 0.0706, 1.8765, 4.8318, 1.1538, 4.7076, -
            5.9745, - 0.8276, 1.621],
        );
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 2)?;
        assert_eq!(res.dims(), [1, 4, 7]);
        assert_eq!(
            to_vec2_round_async(& res.squeeze(0) ?, 4). await ?, [[- 1.5596, - 1.8099,
            2.0407, 4.8764, - 0.1743, - 0.735, - 0.7819], [0.7816, 3.8152, - 0.5926,
            2.2515, - 5.1844, - 0.3157, 1.4721], [1.6295, 0.52, 6.2611, 0.7109, 2.6315, -
            1.8793, 0.7113], [1.0949, 1.0166, 1.7464, 2.4561, - 0.79, - 0.5119, 0.1488]]
        );
    }
    Ok(())
}
async fn conv1d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[0.4056f32, -0.8689, -0.0773, -1.5630], dev)?
        .reshape((1, 1, 4))?;
    let w = Tensor::new(&[1f32, 0., 0.], dev)?.reshape((1, 1, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 2]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.4056, - 0.8689]
    );
    let res = t.conv1d(&w, 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.0, 0.4056, - 0.8689, -
        0.0773],
    );
    Ok(())
}
async fn conv2d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32,
            -0.8689,
            -0.0773,
            -1.5630,
            -2.8012,
            -1.5059,
            0.3972,
            1.0852,
            0.4997,
            3.0616,
            1.6541,
            0.0964,
            -0.8338,
            -1.6523,
            -0.8323,
            -0.1699,
            0.0823,
            0.3526,
            0.6843,
            0.2395,
            1.2279,
            -0.9287,
            -1.7030,
            0.1370,
            0.6047,
            0.3770,
            -0.6266,
            0.3529,
            2.2013,
            -0.6836,
            0.2477,
            1.3127,
            -0.2260,
            0.2622,
            -1.2974,
            -0.8140,
            -0.8404,
            -0.3490,
            0.0130,
            1.3123,
            1.7569,
            -0.3956,
            -1.8255,
            0.1727,
            -0.3538,
            2.6941,
            1.0529,
            0.4219,
            -0.2071,
            1.1586,
            0.4717,
            0.3865,
            -0.5690,
            -0.5010,
            -0.1310,
            0.7796,
            0.6630,
            -0.2021,
            2.6090,
            0.2049,
            0.6466,
            -0.5042,
            -0.0603,
            -1.6538,
            -1.2429,
            1.8357,
            1.6052,
            -1.3844,
            0.3323,
            -1.3712,
            0.9634,
            -0.4799,
            -0.6451,
            -0.0840,
            -1.4247,
            0.5512,
            -0.1747,
            -0.5509,
            -0.3742,
            0.3790,
            -0.4431,
            -0.4720,
            -0.7890,
            0.2620,
            0.7875,
            0.5377,
            -0.6779,
            -0.8088,
            1.9098,
            1.2006,
            -0.8,
            -0.4983,
            1.5480,
            0.8265,
            -0.1025,
            0.5138,
            0.5748,
            0.3821,
            -0.4607,
            0.0085,
        ],
        dev,
    )?;
    let w = Tensor::new(
        &[
            -0.9325f32,
            0.6451,
            -0.8537,
            0.2378,
            0.8764,
            -0.1832,
            0.2987,
            -0.6488,
            -0.2273,
            -2.4184,
            -0.1192,
            -0.4821,
            -0.5079,
            -0.5766,
            -2.4729,
            1.6734,
            0.4558,
            0.2851,
            1.1514,
            -0.9013,
            1.0662,
            -0.1817,
            -0.0259,
            0.1709,
            0.5367,
            0.7513,
            0.8086,
            -2.2586,
            -0.5027,
            0.9141,
            -1.3086,
            -1.3343,
            -1.5669,
            -0.1657,
            0.7958,
            0.1432,
            0.3896,
            -0.4501,
            0.1667,
            0.0714,
            -0.0952,
            1.2970,
            -0.1674,
            -0.3178,
            1.0677,
            0.3060,
            0.7080,
            0.1914,
            1.1679,
            -0.3602,
            1.9265,
            -1.8626,
            -0.5112,
            -0.0982,
            0.2621,
            0.6565,
            0.5908,
            1.0089,
            -0.1646,
            1.8032,
            -0.6286,
            0.2016,
            -0.3370,
            1.2555,
            0.8009,
            -0.6488,
            -0.4652,
            -1.5685,
            1.5860,
            0.5583,
            0.4623,
            0.6026,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 5, 5))?;
    let w = w.reshape((2, 4, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [- 4.2812, 2.0923,
        5.2187, 7.5184, 0.752, - 14.9426, 10.0087, 4.391, 0.2918, 1.6715, 10.389, 3.6023,
        - 4.2808, 0.2672, 5.3646, - 5.2023, - 2.1955, - 9.4075]
    );
    let res = {
        let t = Tensor::cat(&[&t.zeros_like()?, &t, &t.zeros_like()?], 0)?;
        t.conv2d(&w, 0, 1, 1, 1)?
    };
    assert_eq!(res.dims(), [3, 2, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& res.i(0) ?.flatten_all() ?, 4). await ?, [0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    );
    assert_eq!(
        to_vec1_round_async(& res.i(1) ?.flatten_all() ?, 4). await ?, [- 4.2812, 2.0923,
        5.2187, 7.5184, 0.752, - 14.9426, 10.0087, 4.391, 0.2918, 1.6715, 10.389, 3.6023,
        - 4.2808, 0.2672, 5.3646, - 5.2023, - 2.1955, - 9.4075]
    );
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 7, 7]);
    assert_eq!(
        to_vec3_round_async(& res.i(0) ?, 4). await ?, [[[- 1.9918, 2.6797, - 0.4599, -
        1.6037, 1.4131, - 2.4012, 2.9277], [1.8016, - 3.5361, 1.0757, 3.5395, - 8.2168, -
        3.2023, 0.5375], [0.8243, 1.8675, 7.8929, - 4.0746, - 6.4415, 5.1139, 1.6889],
        [0.2722, 8.9679, 3.3477, 1.8514, - 4.2896, - 3.8228, - 7.5632], [- 8.5412, -
        5.8142, - 7.1587, - 1.6095, 0.4651, 0.2748, - 2.0985], [2.0833, - 0.6482, -
        12.1692, - 4.1284, - 2.9765, - 0.0656, - 4.5114], [5.307, 2.6957, 2.3087, 1.0478,
        0.7808, - 1.1519, - 0.9579]], [[1.089, 0.1872, - 0.6408, - 0.9897, 0.8503,
        1.1019, - 0.9211], [- 0.1741, - 0.2915, 4.2472, 1.9417, 1.65, 0.6303, - 4.7131],
        [1.6555, 2.4026, - 2.9293, 2.9953, 0.5328, 3.5873, - 0.9621], [- 1.4289, -
        3.2787, 4.1747, - 6.0341, - 4.6341, - 5.7945, 4.142], [7.5973, 6.4431, 5.9872,
        2.1639, - 8.6566, 3.3143, - 3.4059], [- 0.8775, - 3.048, 11.6543, 0.6442, 2.3218,
        - 0.4765, 1.1516], [- 5.5423, - 2.5188, 1.0754, - 0.0563, - 2.9386, - 1.1504,
        1.0171]]]
    );
    let res = t.conv2d(&w, 0, 1, 2, 1)?;
    assert_eq!(res.dims(), [1, 2, 1, 1]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [2.45, - 2.3504],
    );
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 2)?;
    assert_eq!(res.dims(), [1, 2, 9, 9]);
    assert_eq!(
        to_vec3_round_async(& res.i(0) ?, 4). await ?, [[[- 1.9918, 3.1652, - 0.6778, -
        4.3442, 4.4351, 0.6652, - 3.0124, - 0.6031, 2.9277], [2.7036, - 1.7156, - 0.3969,
        1.0516, 1.6381, - 2.8886, - 0.205, 2.4682, - 1.0499], [- 0.9459, 3.1631, 3.707, -
        4.8369, - 8.5166, - 1.4496, - 2.7559, - 3.2698, 1.4376], [- 0.2157, 3.7786, -
        2.0252, - 4.2633, 3.6731, - 1.5142, 5.9391, - 0.2622, - 0.141], [- 6.8121, -
        3.1744, 1.5945, 3.0637, - 9.6088, 1.4446, 2.9489, - 3.0082, - 7.3822], [0.2371,
        3.3303, 0.3861, 2.2646, - 4.6784, 4.1235, - 0.0109, 0.3176, - 0.03], [- 2.5339, -
        2.9564, - 3.4518, - 4.4594, - 9.1873, - 1.9709, - 0.4676, 0.51, - 3.5024],
        [4.007, 0.3067, - 2.2954, 1.1105, - 0.1992, 1.6372, - 2.9268, 0.2807, - 1.2787],
        [5.307, 1.1317, 1.3518, 0.9049, 3.8116, - 0.4075, - 0.8874, - 0.2241, - 0.9579]],
        [[1.089, - 0.6483, 0.0726, - 0.4752, - 1.3283, 1.7103, 1.0703, 0.1076, - 0.9211],
        [- 0.8629, 0.1376, 0.3202, 2.0955, 0.9696, 2.8988, - 1.0012, 1.5049, - 0.1278],
        [1.9286, - 1.5255, - 2.9563, 2.4589, 3.3611, - 0.6951, 0.3525, - 1.7724, -
        5.9861], [1.1226, 2.1561, 3.6417, 4.7546, - 0.692, 4.4126, - 5.1902, 6.0805,
        2.3185], [1.0111, 0.3604, 0.6432, - 3.6605, 7.9517, - 9.2955, - 5.2988, - 3.7803,
        - 2.0642], [3.3172, - 1.7967, - 3.6576, - 2.0942, 1.3158, 0.112, - 1.7405,
        2.9167, 0.7957], [5.1001, 1.8995, - 1.8639, 1.1262, 9.9629, 2.683, - 3.6319, -
        1.1607, 0.5856], [- 4.8445, - 0.5642, 4.2317, 0.0856, 1.2267, - 0.5712, 1.736,
        1.0997, 0.6908], [- 5.5423, - 1.1831, - 1.2176, 0.0843, 0.0446, - 0.7545, -
        2.4798, - 0.0827, 1.0171]]]
    );
    Ok(())
}
async fn conv2d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32,
            -0.8689,
            0.6843,
            0.2395,
            1.2279,
            -0.9287,
            -1.7030,
            0.1370,
            0.1866,
            0.4145,
            -0.6266,
            0.3529,
            2.2013,
            -0.6836,
            0.2477,
            1.3127,
            -0.6957,
            0.3278,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-0.9259f32, 1.3017], dev)?;
    let t = t.reshape((1, 2, 3, 3))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.164, - 0.0111, -
        0.1742, 2.6437, - 2.0268, 1.1823, 3.2855, - 1.0324, 0.2539]
    );
    let res = t.conv2d(&w, 2, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 7, 7]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1640, - 0.0111, -
        0.1742, 0.0, 0.0, 0.0, 0.0, 2.6437, - 2.0268, 1.1823, 0.0, 0.0, 0.0, 0.0, 3.2855,
        - 1.0324, 0.2539, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    );
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.164, - 0.0111, -
        0.1742, 2.6437, - 2.0268, 1.1823, 3.2855, - 1.0324, 0.2539],
    );
    let res = t.transpose(0, 1)?.conv_transpose2d(&w, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [2, 2, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [- 0.3755, 0.8045, -
        0.6336, - 0.2218, - 1.1369, 0.8599, 1.5768, - 0.1268, - 0.1728, 0.528, - 1.131,
        0.8908, 0.3118, 1.5984, - 1.2089, - 2.2168, 0.1783, 0.2429, - 0.3838, 0.5802, -
        0.3268, - 2.0382, 0.6329, - 0.2293, - 1.2154, 0.6441, - 0.3035, 0.5396, - 0.8156,
        0.4594, 2.8654, - 0.8898, 0.3224, 1.7087, - 0.9056, 0.4267]
    );
    Ok(())
}
async fn conv2d_smaller(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866],
        dev,
    )?;
    let w = Tensor::new(&[1f32, 1., 1., 1., 1., 1., 1., 1., 1.], dev)?;
    let t = t.reshape((1, 1, 3, 3))?;
    let w = w.reshape((1, 1, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 1]);
    assert_eq!(to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [- 0.6197]);
    Ok(())
}
async fn conv2d_non_square(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32,
            -0.8689,
            -0.0773,
            -1.5630,
            -2.8012,
            -1.5059,
            0.3972,
            1.0852,
            0.4997,
            3.0616,
            1.6541,
            0.0964,
            -0.8338,
            -1.6523,
            -0.8323,
            -0.1699,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-1.1351f32, 1.3841], dev)?;
    let t = t.reshape((1, 2, 4, 2))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4, 2]);
    assert_eq!(
        to_vec1_round_async(& res.flatten_all() ?, 4). await ?, [0.2312, 5.2238, 2.3772,
        1.9076, 2.0256, - 0.5776, - 1.6028, - 1.467]
    );
    Ok(())
}
async fn conv2d_grad(dev: &Device) -> Result<()> {
    use candle::Var;
    let t = Var::from_slice(
        &[
            0.4056f32,
            -0.8689,
            -0.0773,
            -1.5630,
            -2.8012,
            -1.5059,
            0.3972,
            1.0852,
            0.4997,
            3.0616,
            1.6541,
            0.0964,
            -0.8338,
            -1.6523,
            -0.8323,
            -0.1699,
            0.0823,
            0.3526,
            0.6843,
            0.2395,
            1.2279,
            -0.9287,
            -1.7030,
            0.1370,
            0.6047,
            0.3770,
            -0.6266,
            0.3529,
            2.2013,
            -0.6836,
            0.2477,
            1.3127,
            -0.2260,
            0.2622,
            -1.2974,
            -0.8140,
            -0.8404,
            -0.3490,
            0.0130,
            1.3123,
            1.7569,
            -0.3956,
            -1.8255,
            0.1727,
            -0.3538,
            2.6941,
            1.0529,
            0.4219,
            -0.2071,
            1.1586,
            0.4717,
            0.3865,
            -0.5690,
            -0.5010,
            -0.1310,
            0.7796,
            0.6630,
            -0.2021,
            2.6090,
            0.2049,
            0.6466,
            -0.5042,
            -0.0603,
            -1.6538,
            -1.2429,
            1.8357,
            1.6052,
            -1.3844,
            0.3323,
            -1.3712,
            0.9634,
            -0.4799,
            -0.6451,
            -0.0840,
            -1.4247,
            0.5512,
            -0.1747,
            -0.5509,
            -0.3742,
            0.3790,
            -0.4431,
            -0.4720,
            -0.7890,
            0.2620,
            0.7875,
            0.5377,
            -0.6779,
            -0.8088,
            1.9098,
            1.2006,
            -0.8,
            -0.4983,
            1.5480,
            0.8265,
            -0.1025,
            0.5138,
            0.5748,
            0.3821,
            -0.4607,
            0.0085,
        ],
        (1, 4, 5, 5),
        dev,
    )?;
    let w = Var::from_slice(
        &[
            -0.9325f32,
            0.6451,
            -0.8537,
            0.2378,
            0.8764,
            -0.1832,
            0.2987,
            -0.6488,
            -0.2273,
            -2.4184,
            -0.1192,
            -0.4821,
            -0.5079,
            -0.5766,
            -2.4729,
            1.6734,
            0.4558,
            0.2851,
            1.1514,
            -0.9013,
            1.0662,
            -0.1817,
            -0.0259,
            0.1709,
            0.5367,
            0.7513,
            0.8086,
            -2.2586,
            -0.5027,
            0.9141,
            -1.3086,
            -1.3343,
            -1.5669,
            -0.1657,
            0.7958,
            0.1432,
            0.3896,
            -0.4501,
            0.1667,
            0.0714,
            -0.0952,
            1.2970,
            -0.1674,
            -0.3178,
            1.0677,
            0.3060,
            0.7080,
            0.1914,
            1.1679,
            -0.3602,
            1.9265,
            -1.8626,
            -0.5112,
            -0.0982,
            0.2621,
            0.6565,
            0.5908,
            1.0089,
            -0.1646,
            1.8032,
            -0.6286,
            0.2016,
            -0.3370,
            1.2555,
            0.8009,
            -0.6488,
            -0.4652,
            -1.5685,
            1.5860,
            0.5583,
            0.4623,
            0.6026,
        ],
        (2, 4, 3, 3),
        dev,
    )?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(to_vec0_round_async(& loss, 2). await ?, 741.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        to_vec1_round_async(& grad_t.flatten_all() ?, 2). await ?, [9.29, - 2.84, - 5.71,
        3.38, - 7.71, - 19.15, 7.02, 29.1, 9.34, 34.73, - 22.87, 24.35, - 39.88, - 14.01,
        21.08, 9.94, 13.63, - 34.68, 11.21, - 6.26, 7.72, - 6.32, - 16.64, - 1.08, -
        20.22, 21.73, - 0.37, - 4.06, 5.82, - 3.65, - 30.73, 14.55, 87.7, 31.6, 4.53, -
        89.78, - 75.37, - 57.43, - 7.56, 92.96, 18.79, - 4.63, - 159.75, - 42.47, -
        47.26, 52.88, 37.32, 49.0, 12.82, 2.01, - 8.98, 20.18, 16.62, 12.06, 15.38, 20.0,
        2.57, - 15.22, 72.62, - 10.75, 2.25, - 31.2, 3.75, - 0.2, 9.76, - 0.68, 5.21, -
        40.44, - 22.59, - 61.61, 17.28, 20.41, 37.55, 5.23, 6.81, 23.54, 23.62, - 9.99, -
        9.13, 4.87, - 35.06, - 26.1, 63.48, 25.81, - 39.21, - 70.68, - 46.96, 2.33,
        41.81, 82.42, - 28.63, - 11.78, - 35.33, - 10.28, - 28.57, - 9.13, 7.21, - 9.05,
        - 9.62, - 11.25]
    );
    assert_eq!(
        to_vec1_round_async(& grad_w.flatten_all() ?, 2). await ?, [- 28.92, - 22.88, -
        141.23, 73.35, 61.07, 47.81, - 20.0, - 73.71, - 41.82, - 13.59, 21.5, 28.72,
        28.57, - 46.85, - 90.19, 143.61, 16.68, 7.43, 18.88, - 90.81, - 20.29, 54.79,
        82.63, 22.94, 77.81, - 16.39, - 13.2, 9.34, - 40.39, - 26.62, 5.33, - 60.91,
        9.09, - 59.37, 7.08, 58.64, 5.55, 20.52, 2.5, - 17.25, - 6.8, 22.21, 30.15, -
        7.52, - 37.46, 5.67, 22.58, 9.03, 47.05, 17.61, 37.31, - 98.13, - 14.61, - 4.8, -
        6.36, 44.69, 23.34, 8.37, - 13.52, 80.05, - 34.24, - 16.36, - 12.31, 1.92, -
        33.62, - 14.1, - 49.23, - 7.39, 11.5, - 9.98, 9.66, 29.6]
    );
    let res = t.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(to_vec0_round_async(& loss, 2). await ?, 277.16f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        to_vec3_round_async(& grad_t.i(0) ?, 2). await ?, [[[9.29, - 7.03, 0.94, 3.49, -
        7.71], [- 1.8, - 7.82, 8.9, 8.46, 7.43], [- 25.84, 22.09, - 19.27, - 0.22, 1.69],
        [4.02, 18.53, - 18.37, 2.3, - 24.51], [7.72, - 9.68, - 12.34, 5.6, - 20.22]],
        [[21.73, 3.39, - 18.27, 3.86, - 3.65], [8.25, 3.73, 30.73, - 8.61, - 11.93], [-
        72.15, - 15.36, - 17.53, - 12.32, - 1.61], [- 22.32, - 7.79, - 91.82, 6.44, -
        37.69], [52.88, 14.44, 42.75, 9.88, 2.01]], [[- 8.98, 9.91, 6.75, - 4.68, 15.38],
        [4.93, - 0.33, 9.94, - 1.46, 14.78], [13.62, - 30.63, 3.96, - 3.58, - 4.48], [-
        14.13, 1.19, - 34.43, 3.08, - 33.83], [17.28, 12.94, 31.83, - 3.35, 6.81]],
        [[23.54, 6.98, - 24.52, 0.52, 4.87], [9.65, 6.18, 1.71, - 25.23, - 4.93], [-
        54.99, - 23.66, 3.19, - 3.73, 18.58], [- 21.35, - 10.39, - 39.88, 28.73, -
        30.76], [- 9.13, 11.12, - 14.0, - 8.23, - 11.25]]]
    );
    assert_eq!(
        to_vec3_round_async(& grad_w.i(0) ?, 2). await ?, [[[28.34, - 7.91, - 45.75],
        [21.03, 3.86, 29.86], [0.72, - 36.58, - 35.28]], [[- 16.04, 11.53, - 16.38],
        [29.62, - 16.32, - 48.35], [57.5, 28.29, 25.81]], [[2.93, - 19.6, 1.57], [27.15,
        53.88, - 24.64], [12.74, - 22.6, - 26.2]], [[- 0.18, - 14.86, - 6.82], [- 19.55,
        - 2.72, 45.9], [- 2.54, 36.97, 27.11]]]
    );
    let res = t.i((.., .., 0..4, 0..4))?.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(to_vec0_round_async(& loss, 2). await ?, 21.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        to_vec3_round_async(& grad_t.i(0) ?, 2). await ?, [[[9.29, - 7.03, 7.87, 0.0,
        0.0], [- 1.8, - 7.82, 5.9, 0.0, 0.0], [- 3.12, 4.49, 5.52, 0.0, 0.0], [0.0, 0.0,
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], [[21.73, 3.39, 4.77, 0.0, 0.0],
        [8.25, 3.73, 27.61, 0.0, 0.0], [- 20.55, - 5.61, - 2.77, 0.0, 0.0], [0.0, 0.0,
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], [[- 8.98, 9.91, - 7.15, 0.0, 0.0],
        [4.93, - 0.33, 4.56, 0.0, 0.0], [- 6.7, - 5.76, - 8.05, 0.0, 0.0], [0.0, 0.0,
        0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], [[23.54, 6.98, - 10.0, 0.0, 0.0],
        [9.65, 6.18, 18.72, 0.0, 0.0], [3.29, - 5.27, 0.79, 0.0, 0.0], [0.0, 0.0, 0.0,
        0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]
    );
    assert_eq!(
        to_vec3_round_async(& grad_w.i(0) ?, 2). await ?, [[[- 3.47, 7.44, 0.66], [12.89,
        - 3.4, - 9.29], [- 14.16, - 0.83, 7.14]], [[- 3.23, 5.37, - 3.02], [- 2.12, -
        11.24, 1.94], [6.97, 7.2, 2.99]], [[- 4.04, - 3.31, 4.87], [- 6.68, - 5.68,
        1.73], [- 5.54, 4.32, 0.52]], [[- 4.72, 1.5, 4.72], [3.79, 4.04, 6.76], [- 4.6,
        5.8, 6.93]]]
    );
    let padding = 4;
    let outpadding = 2;
    let dilation = 3;
    let stride = 3;
    let t = Var::from_slice(
        &[
            0.4056_f32,
            -0.8689,
            -0.0773,
            -1.5630,
            -2.8012,
            -1.5059,
            0.3972,
            1.0852,
            0.4997,
            3.0616,
            1.6541,
            0.0964,
            -0.8338,
            -1.6523,
            -0.8323,
            -0.1699,
            0.0823,
            0.3526,
            0.6843,
            0.2395,
            1.2279,
            -0.9287,
            -1.7030,
            0.1370,
            0.6047,
            0.3770,
            -0.6266,
            0.3529,
            2.2013,
            -0.6836,
            0.2477,
            1.3127,
            -0.2260,
            0.2622,
            -1.2974,
            -0.8140,
            -0.8404,
            -0.3490,
            0.0130,
            1.3123,
            1.7569,
            -0.3956,
            -1.8255,
            0.1727,
            -0.3538,
            2.6941,
            1.0529,
            0.4219,
            -0.2071,
            1.1586,
            0.4717,
            0.3865,
            -0.5690,
            -0.5010,
            -0.1310,
            0.7796,
            0.6630,
            -0.2021,
            2.6090,
            0.2049,
            0.6466,
            -0.5042,
            -0.0603,
            -1.6538,
            -1.2429,
            1.8357,
            1.6052,
            -1.3844,
            0.3323,
            -1.3712,
            0.9634,
            -0.4799,
            -0.6451,
            -0.0840,
            -1.4247,
            0.5512,
            -0.1747,
            -0.5509,
            -0.3742,
            0.3790,
            -0.4431,
            -0.4720,
            -0.7890,
            0.2620,
            0.5411,
            -1.1715,
            -2.4997,
            2.3249,
            -0.8912,
            -0.4733,
            -0.5701,
            -2.8888,
            -1.4112,
            -0.5471,
            -0.9234,
            -1.1660,
            0.4189,
            -0.7465,
            -0.6473,
            0.1402,
            0.7875,
            0.5377,
            -0.6779,
            -0.8088,
            -0.4864,
            -0.2312,
            0.9279,
            0.1264,
            1.5480,
            0.8265,
            -0.1025,
            0.5138,
            -0.2512,
            0.1576,
            1.2705,
            0.3641,
            -0.9325,
            0.6451,
            -0.8537,
            0.2378,
            0.1794,
            0.2752,
            -0.3687,
            -1.1149,
            -0.1410,
            -0.5829,
            -0.0892,
            1.4258,
            -2.2789,
            0.5270,
            0.1825,
            1.7007,
            -0.5263,
            -0.2954,
            0.4440,
            0.5537,
            0.3492,
            0.6186,
            1.6475,
            0.2219,
        ],
        (1, 4, 7, 5),
        dev,
    )?;
    #[rustfmt::skip]
    let w = Var::from_slice(
        &[
            -1.1744_f32,
            0.3266,
            2.5893,
            1.0142,
            0.1763,
            0.7752,
            0.6604,
            0.2029,
            -0.2145,
            0.7234,
            -0.3441,
            -1.5400,
            -0.6333,
            0.6613,
            0.2083,
            0.6230,
            -1.7002,
            0.3393,
            0.4049,
            1.0762,
            0.2723,
            1.4181,
            0.0029,
            -0.2122,
            1.7668,
            1.4168,
            0.3320,
            -0.2719,
            0.7932,
            -0.7204,
            0.4447,
            0.1211,
            0.5908,
            1.0089,
            -0.1646,
            1.8033,
            -0.6286,
            0.2016,
            -0.3370,
            1.2555,
            0.8009,
            -0.6488,
            -0.4652,
            -1.5685,
            1.5860,
            0.5583,
            0.4623,
            0.6026,
            0.8828,
            2.4990,
            0.6811,
            -0.3369,
            1.3320,
            1.7669,
            -1.1067,
            1.2958,
            -0.9415,
            -0.9655,
            -0.4462,
            0.7181,
            0.5181,
            -1.1658,
            -1.8467,
            -0.7763,
            1.2769,
            0.8651,
            0.9890,
            1.5092,
            0.7207,
            -0.8481,
            0.7417,
            0.3375,
            -1.2685,
            1.4572,
            1.0915,
            0.1093,
            -0.8550,
            -0.5831,
            -0.6309,
            -0.2509,
            0.5220,
            -0.0914,
            0.7900,
            0.1096,
            0.3258,
            0.2723,
            -1.0942,
            -0.3393,
            -0.1653,
            0.5732,
            -0.8014,
            1.8194,
            -1.9023,
            0.2127,
            1.8636,
            -0.8979,
            0.1927,
            -0.2778,
            0.3105,
            0.0071,
            -1.1823,
            0.2476,
            -0.7178,
            -1.3821,
            1.0769,
            -0.4376,
            -0.9967,
            -0.1227,
            1.6197,
            -1.0604,
            0.1372,
            0.8141,
            -0.6163,
            0.7304,
            -0.8285,
            2.0636,
            -0.7176,
            0.2495,
            -0.2581,
            -0.4478,
        ],
        (4, 2, 3, 5),
        dev,
    )?;
    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(to_vec0_round_async(& loss, 0). await ?, 2904.0);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);
    assert_eq!(
        to_vec1_round_async(& grad_w.flatten_all() ?, 1). await ?, [- 89.0, - 135.3,
        136.7, 102.0, - 53.4, 117.9, 118.6, - 43.9, - 218.0, - 58.5, - 114.3, - 150.0, -
        15.6, 172.1, 66.3, - 64.3, - 27.9, - 19.8, 31.7, 62.1, 5.5, 92.6, 28.2, - 29.6,
        55.9, 52.7, - 72.7, - 119.8, 53.8, - 25.5, 128.8, 19.3, 68.0, 190.9, - 64.1, -
        86.2, - 111.2, 106.6, - 67.7, 37.8, 115.9, 50.4, - 77.7, - 54.9, 22.3, - 4.6,
        89.8, 61.7, 122.4, 192.6, - 27.8, - 104.6, 57.0, 166.4, 27.1, 6.1, 18.7, - 93.2,
        31.5, 168.2, - 3.7, - 99.5, - 55.5, - 10.8, 17.5, 20.8, 16.9, 43.8, 42.0, - 89.2,
        18.8, - 9.6, - 84.1, 212.6, 19.7, - 50.0, - 52.0, - 40.0, - 166.6, - 73.2, -
        10.8, - 73.3, 31.5, - 23.4, - 79.3, - 27.0, - 84.4, - 42.9, - 20.3, 51.8, - 16.7,
        76.3, - 120.5, - 65.8, 96.5, - 10.7, - 45.9, - 88.1, 65.4, - 7.0, - 1.5, 92.8, -
        25.1, - 114.2, - 5.8, - 14.8, - 51.2, - 20.7, 54.2, - 79.8, 47.7, - 29.2, - 8.8,
        53.5, - 28.4, 85.0, - 18.3, 107.0, 28.3, - 71.8]
    );
    assert_eq!(
        to_vec3_round_async(& grad_t.i(0) ?, 1). await ?, [[[32.3, - 41.6, - 24.0, 14.1,
        17.6], [- 11.8, 72.5, 87.6, 46.4, 61.5], [115.0, 108.5, - 48.6, - 63.4, - 50.0],
        [51.3, 5.4, 31.3, 91.1, - 30.9], [52.7, 92.8, - 68.0, - 47.0, 83.0], [- 10.2, -
        107.0, - 5.4, 213.1, - 31.4], [- 2.4, 65.1, 9.2, - 146.2, - 24.2]], [[- 72.6, -
        63.9, - 61.9, 45.3, 33.0], [79.3, - 0.5, - 26.2, 78.2, 42.7], [90.9, 141.6, 40.1,
        - 62.7, 37.0], [32.8, 198.2, - 0.8, - 31.1, 27.3], [34.5, 34.9, - 47.9, 127.6, -
        12.3], [- 61.4, - 3.2, - 2.9, - 10.9, - 16.6], [74.6, 60.1, - 68.9, 34.5, -
        50.4]], [[37.5, - 56.9, - 43.6, - 13.5, - 9.9], [40.0, 97.3, 28.6, 14.2, - 30.1],
        [- 22.3, - 126.3, - 68.8, - 8.2, 26.1], [- 32.9, 37.3, 108.5, - 54.8, 29.6],
        [34.9, - 176.9, - 125.0, - 28.3, - 13.9], [- 54.9, 142.6, 62.1, - 80.4, - 65.6],
        [7.4, - 91.1, - 67.6, 35.0, 39.7]], [[- 57.2, - 40.9, - 10.1, 32.6, 29.4], [18.7,
        - 18.0, 29.5, - 1.2, 59.2], [- 14.0, - 74.4, 19.8, - 117.0, 58.2], [- 21.8,
        163.5, - 71.1, - 99.0, 80.9], [- 58.9, - 10.9, 93.8, - 139.6, 98.0], [- 54.4,
        135.3, 6.0, - 79.1, 134.6], [27.5, - 76.0, 43.4, - 2.8, - 7.8]]]
    );
    let padding = 1;
    let outpadding = 1;
    let dilation = 1;
    let stride = 2;
    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(to_vec0_round_async(& loss, 0). await ?, 3627.0);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 7, 5]);
    assert_eq!(grad_w.dims(), [4, 2, 3, 5]);
    #[rustfmt::skip]
    assert_eq!(
        to_vec3_round_async(& grad_t.i(0) ?, 1). await ?, [[[13.2, - 40.7, - 9.7, - 47.3,
        - 82.7], [- 98.2, 9.7, 57.7, - 6.2, 180.7], [100.2, 24.1, 3.7, - 100.5, - 48.1],
        [- 0.3, 13.5, - 2.9, 80.0, - 49.8], [47.2, - 25.6, - 74.4, 61.2, - 18.4], [4.6, -
        69.5, 27.9, 66.5, - 88.1], [- 12.0, 79.2, - 40.0, 4.1, - 97.1],], [[- 42.2, -
        36.5, - 51.1, 7.5, 32.3], [74.1, - 44.6, - 68.8, 19.5, 7.7], [137.1, 54.2, 153.8,
        - 58.0, 45.5], [24.4, - 56.8, 9.7, - 41.0, - 14.5], [- 3.7, 72.6, 8.3, 134.8,
        40.5], [43.2, - 56.9, - 47.5, - 89.4, - 95.4], [68.2, 108.1, - 80.0, 57.0, -
        121.1]], [[31.1, - 11.4, - 34.8, 33.1, - 44.2], [29.4, - 31.6, - 40.2, 13.7,
        13.1], [- 0.8, - 83.8, - 7.8, - 17.3, 78.2], [12.0, - 118.7, 137.5, - 76.7,
        50.8], [- 28.7, - 114.2, - 3.7, - 96.3, - 13.8], [- 31.8, 28.5, - 14.3, 4.6,
        13.4], [28.0, - 0.2, - 38.9, - 29.7, - 59.0]], [[- 16.8, 38.5, 15.5, 26.6, 48.9],
        [14.5, 49.6, - 24.8, 65.6, 61.7], [22.1, - 64.7, - 4.3, - 51.0, 36.3], [31.0, -
        88.9, 47.1, - 123.5, - 3.8], [- 14.8, - 39.8, 128.2, - 110.3, 42.6], [- 7.1,
        95.3, - 21.3, - 58.7, - 13.9], [26.9, 21.3, 16.1, 70.3, 32.1]]]
    );
    #[rustfmt::skip]
    assert_eq!(
        to_vec1_round_async(& grad_w.flatten_all() ?, 1). await ?, [- 2.460e+01, -
        3.100e+00, 2.219e+02, 7.400e+00, 5.620e+01, 7.420e+01, 7.830e+01, 8.900e+00,
        1.050e+01, 2.810e+01, 5.100e+00, - 1.046e+02, - 1.572e+02, 8.710e+01, -
        9.840e+01, - 4.230e+01, - 1.898e+02, 1.860e+01, - 3.570e+01, 9.810e+01,
        4.680e+01, 1.182e+02, 4.020e+01, - 1.900e+00, 1.508e+02, 1.094e+02, 1.018e+02, -
        4.620e+01, 1.591e+02, - 2.320e+01, - 8.450e+01, - 4.600e+00, 6.330e+01,
        1.123e+02, - 7.000e+00, 1.101e+02, - 6.620e+01, 2.090e+01, - 5.120e+01,
        8.990e+01, 9.050e+01, - 6.990e+01, 6.800e+01, - 9.250e+01, 1.380e+02, 4.720e+01,
        4.710e+01, 6.210e+01, 8.870e+01, 2.098e+02, 3.870e+01, - 1.390e+01, 6.270e+01,
        1.484e+02, - 9.920e+01, - 4.200e+01, - 1.505e+02, - 1.480e+01, - 2.620e+01,
        8.220e+01, - 3.350e+01, - 2.260e+01, - 1.198e+02, - 5.080e+01, 1.259e+02,
        5.600e+01, 9.270e+01, 1.209e+02, 6.590e+01, - 8.330e+01, 7.000e+00, - 2.600e+01,
        - 1.133e+02, 3.870e+01, 4.020e+01, - 6.300e+00, - 8.710e+01, - 5.150e+01, -
        8.510e+01, 2.000e-01, 3.640e+01, - 6.100e+00, 6.590e+01, - 2.700e+00, 6.550e+01,
        5.300e+00, - 6.760e+01, - 4.270e+01, - 3.900e+00, 2.880e+01, 5.260e+01,
        6.170e+01, - 1.203e+02, - 1.610e+01, 7.740e+01, - 1.008e+02, - 1.070e+01, -
        9.900e+00, 3.300e+00, - 2.620e+01, - 4.440e+01, 2.580e+01, - 6.920e+01, -
        4.220e+01, 1.108e+02, 1.240e+01, - 3.440e+01, - 2.800e+00, 7.880e+01, -
        6.690e+01, 1.480e+01, 2.310e+01, - 4.260e+01, - 1.500e+00, - 4.760e+01,
        5.350e+01, - 2.260e+01, 8.000e-01, - 3.840e+01, - 2.500e+00]
    );
    Ok(())
}
candle_wasm_tests::test_device!(
    conv1d, conv1d_cpu, conv1d_gpu, conv1d_metal, conv1d_wgpu
);
candle_wasm_tests::test_device!(
    conv1d_small, conv1d_small_cpu, conv1d_small_gpu, conv1d_small_metal,
    conv1d_small_wgpu
);
candle_wasm_tests::test_device!(
    conv2d, conv2d_cpu, conv2d_gpu, conv2d_metal, conv2d_wgpu
);
candle_wasm_tests::test_device!(
    conv2d_non_square, conv2d_non_square_cpu, conv2d_non_square_gpu,
    conv2d_non_square_metal, conv2d_non_square_wgpu
);
candle_wasm_tests::test_device!(
    conv2d_small, conv2d_small_cpu, conv2d_small_gpu, conv2d_small_metal,
    conv2d_small_wgpu
);
candle_wasm_tests::test_device!(
    conv2d_smaller, conv2d_smaller_cpu, conv2d_smaller_gpu, conv2d_smaller_metal,
    conv2d_smaller_wgpu
);
candle_wasm_tests::test_device!(
    conv2d_grad, conv2d_grad_cpu, conv2d_grad_gpu, conv2_grad_metal, conv2_grad_wgpu
);
}pub mod custom_op_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::backend::BackendStorage;
use candle::cpu_backend;
use candle::test_utils::to_vec1_round;
use candle::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};
fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}
struct Elu {
    alpha: f64,
}
impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle::map_dtype!(
            "elu", s, | s | cpu_backend::unary_map(s, l, | v | fwd(v, self.alpha)),
            (F8E4M3, BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}
#[test]
async fn custom_op1_no_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round_async(& elu_t, 4). await ?, & [- 0.9933, - 0.9817, - 0.9502, -
        0.8647, - 0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}
fn bwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        T::one()
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v.exp() * alpha
    }
}
struct EluBackward {
    alpha: f64,
}
impl CustomOp1 for EluBackward {
    fn name(&self) -> &'static str {
        "elu-bwd"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle::map_dtype!(
            "elu-bwd", s, | s | cpu_backend::unary_map(s, l, | v | bwd(v, self.alpha)),
            (F8E4M3, BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}
struct EluWithBackward(Elu);
impl EluWithBackward {
    fn new(alpha: f64) -> Self {
        Self(Elu { alpha })
    }
}
impl CustomOp1 for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        self.0.cpu_fwd(s, l)
    }
    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<Option<Tensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}
#[test]
async fn custom_op1_with_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = candle::Var::new(&[-2f32, 0f32, 2f32], cpu)?;
    let elu_t = t.apply_op1(EluWithBackward::new(2.))?;
    assert_eq!(to_vec1_round_async(& elu_t, 4). await ?, & [- 1.7293, 0.0, 2.0]);
    let grads = elu_t.backward()?;
    let grad_x = grads.get(&t).unwrap();
    assert_eq!(to_vec1_round_async(grad_x, 4). await ?, [0.2707, 1.0, 1.0]);
    Ok(())
}
impl candle::InplaceOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        let alpha = self.alpha;
        match s {
            CpuStorage::F8E4M3(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::BF16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F32(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F64(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            _ => candle::bail!("unsupported dtype for inplace elu"),
        }
        Ok(())
    }
}
#[test]
async fn inplace_op1() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round_async(& t, 4). await ?, & [- 0.9933, - 0.9817, - 0.9502, - 0.8647,
        - 0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}
#[cfg(all(feature = "ug", any(feature = "cuda", feature = "metal")))]
#[allow(clippy::approx_constant)]
#[test]
async fn ug_op() -> Result<()> {
    let kernel = {
        use candle_ug::lang::op;
        let layout = candle_ug::Layout::from_shape(&[12]);
        let ptr = op::Arg::ptr(candle_ug::DType::F32);
        let src = op::load(ptr.id(), layout.clone(), candle_ug::DType::F32)?;
        let src = op::unary(op::UnaryOp::Exp, src)?;
        let st = op::store(ptr.id(), layout, src)?;
        let kernel = op::Kernel::new("exp".to_string(), vec![ptr], vec![st]);
        let opts: candle_ug::lower_op::Opts = Default::default();
        kernel.lower(&opts)?
    };
    let device = if candle::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if candle::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        candle::bail!("metal/cuda is mandatory for this test")
    };
    let op = candle::UgIOp1::new("test", kernel, &device)?;
    let t = Tensor::arange(0u32, 12u32, &device)?.to_dtype(DType::F32)?;
    t.inplace_op1(&op)?;
    assert_eq!(
        to_vec1_round_async(& t, 2). await ?, & [1.0, 2.72, 7.39, 20.09, 54.6, 148.41,
        403.43, 1096.63, 2980.96, 8103.08, 22026.47, 59874.13]
    );
    Ok(())
}
}pub mod display_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{DType, Device::Cpu, Tensor};
#[test]
async fn display_scalar() -> Result<()> {
    let t = Tensor::new(1234u32, &Cpu)?;
    let s = format!("{t}");
    assert_eq!(& s, "[1234]\nTensor[[], u32]");
    let t = t.to_dtype(DType::F32)?.neg()?;
    let s = format!("{}", (& t / 10.0) ?);
    assert_eq!(& s, "[-123.4000]\nTensor[[], f32]");
    let s = format!("{}", (& t / 1e8) ?);
    assert_eq!(& s, "[-1.2340e-5]\nTensor[[], f32]");
    let s = format!("{}", (& t * 1e8) ?);
    assert_eq!(& s, "[-1.2340e11]\nTensor[[], f32]");
    let s = format!("{}", (& t * 0.) ?);
    assert_eq!(& s, "[0.]\nTensor[[], f32]");
    Ok(())
}
#[test]
async fn display_vector() -> Result<()> {
    let t = Tensor::new::<&[u32; 0]>(&[], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(& s, "[]\nTensor[[0], u32]");
    let t = Tensor::new(&[0.1234567, 1.0, -1.2, 4.1, f64::NAN], &Cpu)?;
    let s = format!("{t}");
    assert_eq!(& s, "[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], f64]");
    let t = (Tensor::ones(50, DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42.]
Tensor[[50], f32]"#;
    assert_eq!(& s, expected);
    let t = (Tensor::ones(11000, DType::F32, &Cpu)? * 42.)?;
    let s = format!("{t}");
    assert_eq!(& s, "[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], f32]");
    Ok(())
}
#[test]
async fn display_multi_dim() -> Result<()> {
    let t = (Tensor::ones((200, 100), DType::F32, &Cpu)? * 42.)?;
    let s = format!("\n{t}");
    let expected = r#"
[[42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 ...
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.]]
Tensor[[200, 100], f32]"#;
    assert_eq!(& s, expected);
    let t = t.reshape(&[2, 1, 1, 100, 100])?;
    let t = format!("\n{t}");
    let expected = r#"
[[[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]],
 [[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]]]
Tensor[[2, 1, 1, 100, 100], f32]"#;
    assert_eq!(& t, expected);
    Ok(())
}
}pub mod grad_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#![allow(clippy::approx_constant)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::{Context, Result};
use candle::{test_device, test_utils, DType, Device, Shape, Tensor, Var};
async fn simple_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4.], device)?;
    let x = x.as_tensor();
    let y = (((x * x)? + x * 5f64)? + 4f64)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(x.to_vec1_async::< f32 > (). await ?, [3., 1., 4.]);
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [28., 10., 40.]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [11., 7., 13.]);
    Ok(())
}
async fn sum_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4.], device)?;
    let x = x.as_tensor();
    let y = (x.sqr()?.sum_keepdim(0)? * 2.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [52.]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, & [12., 4., 16.]);
    let y = (x.sqr()?.sum_keepdim(0)? * 2.)?.squeeze(0)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_scalar_async::< f32 > (). await ?, 52.);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, & [12., 4., 16.]);
    Ok(())
}
async fn matmul_grad(device: &Device) -> Result<()> {
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let x = Var::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let y = Var::from_slice(&data, (2, 3, 2), device)?;
    let c = x.matmul(&y)?;
    let grads = c.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    let grad_y = grads.get(&y).context("no grad for y")?;
    assert_eq!(grad_x.shape(), & Shape::from((2, 2, 3)));
    assert_eq!(grad_y.shape(), & Shape::from((2, 3, 2)));
    assert_eq!(
        &* grad_x.to_vec3_async::< f32 > (). await ?, & [[[1., 5., 9.], [1., 5., 9.]],
        [[13., 17., 21.], [13., 17., 21.]]]
    );
    assert_eq!(
        &* grad_y.to_vec3_async::< f32 > (). await ?, & [[[3., 3.], [5., 5.], [7., 7.]],
        [[15., 15.], [17., 17.], [19., 19.]]]
    );
    Ok(())
}
async fn grad_descent(device: &Device) -> Result<()> {
    let x = Var::new(0f32, device)?;
    let learning_rate = 0.1;
    for _step in 0..100 {
        let xt = x.as_tensor();
        let c = ((xt - 4.2)? * (xt - 4.2)?)?;
        let grads = c.backward()?;
        let x_grad = grads.get(&x).context("no grad for x")?;
        x.set(&(xt - x_grad * learning_rate)?)?
    }
    assert_eq!(x.to_scalar_async::< f32 > (). await ?, 4.199999);
    Ok(())
}
async fn unary_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., 4., 0.15], device)?;
    let x = x.as_tensor();
    let y = (x.log()? + 1.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [2.0986, 1.0, 2.3863, - 0.8971]);
    assert_eq!(to_vec1_round_async(grad_x, 4). await ?, [0.3333, 1.0, 0.25, 6.6667]);
    let y = x.exp()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [20.0855, 2.7183, 54.5982, 1.1618]);
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [20.0855, 2.7183, 54.5982, 1.1618]
    );
    let y = x.exp()?.sqr()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 3). await ?, [403.429, 7.389, 2980.958, 1.35]);
    assert_eq!(to_vec1_round_async(grad_x, 2). await ?, [806.86, 14.78, 5961.92, 2.7]);
    let y = x.sin()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [0.1411, 0.8415, - 0.7568, 0.1494],
    );
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [- 0.99, 0.5403, - 0.6536, 0.9888],
    );
    let y = x.cos()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 0.99, 0.5403, - 0.6536, 0.9888],
    );
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [- 0.1411, - 0.8415, 0.7568, - 0.1494],
    );
    let y = x.sqr()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [9.0, 1.0, 16.0, 0.0225]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [6.0, 2.0, 8.0, 0.3]);
    let y = x.sqr()?.sqrt()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [3.0, 1.0, 4.0, 0.15]);
    assert_eq!(to_vec1_round_async(grad_x, 4). await ?, [1.0, 1.0, 1.0, 1.0]);
    let y = x.neg()?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [- 3.0, - 1.0, - 4.0, - 0.15]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [- 1.0, - 1.0, - 1.0, - 1.0]);
    let y = x.affine(0.2, 1.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [1.6, 1.2, 1.8, 1.03]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [0.2, 0.2, 0.2, 0.2]);
    let y = Tensor::new(1f32, device)?.broadcast_div(x)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [0.3333, 1.0, 0.25, 6.6667]);
    assert_eq!(
        grad_x.to_vec1_async::< f32 > (). await ?, [- 0.11111111, - 1.0, - 0.0625, -
        44.444443],
    );
    let y = x.broadcast_div(&Tensor::new(0.5f32, device)?)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [6., 2., 8., 0.3]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [2., 2., 2., 2.]);
    let x = Var::new(&[3f32, 1., 4., 0.15], device)?;
    let y = x.powf(2.5)?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 2). await ?, [15.59, 1.0, 32.0, 0.01]);
    assert_eq!(to_vec1_round_async(grad_x, 2). await ?, [12.99, 2.5, 20.0, 0.15]);
    let y = x.tanh()?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 2). await ?, [1.0, 0.76, 1.0, 0.15]);
    assert_eq!(to_vec1_round_async(grad_x, 2). await ?, [0.01, 0.42, 0.0, 0.98],);
    let y = x.gelu()?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [2.9964, 0.8412, 3.9999, 0.0839]);
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [1.0116, 1.0830, 1.0003, 0.6188],
    );
    let y = x.erf()?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [1.0, 0.8427, 1.0, 0.168]);
    assert_eq!(to_vec1_round_async(grad_x, 4). await ?, [0.0001, 0.4151, 0.0, 1.1033],);
    let y = x.gelu_erf()?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [2.9960, 0.8413, 3.9999, 0.0839]);
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [1.0119, 1.0833, 1.0005, 0.6188],
    );
    let elu_x = Var::new(&[-1.0f32, 0., -2., 3.], device)?;
    let y = elu_x.elu(2.)?;
    let grads = y.backward()?;
    let grad_x = grads.get(&elu_x).context("no grad for x")?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 1.2642, 0.0000, - 1.7293, 3.0000]
    );
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [0.7358, 2.0000, 0.2707, 1.0000]
    );
    let y = x.silu()?;
    let grads = y.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(to_vec1_round_async(& y, 4). await ?, [2.8577, 0.7311, 3.9281, 0.0806]);
    assert_eq!(
        to_vec1_round_async(grad_x, 4). await ?, [1.0881, 0.9277, 1.0527, 0.5747],
    );
    if device.is_cpu() {
        let x = Var::new(&[[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]]], device)?;
        let y = x.interpolate1d(12)?.reshape(36)?;
        let z = Tensor::new(
            &[
                1_f32,
                02.,
                03.,
                04.,
                05.,
                06.,
                07.,
                08.,
                09.,
                10.,
                11.,
                12.,
                13.,
                14.,
                15.,
                16.,
                17.,
                18.,
                19.,
                20.,
                21.,
                22.,
                23.,
                24.,
                25.,
                26.,
                27.,
                28.,
                29.,
                30.,
                31.,
                32.,
                33.,
                34.,
                35.,
                36.,
            ],
            device,
        )?;
        let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
        let grads = loss.backward()?;
        let grad_x = grads.get(&x).context("no grad for x")?;
        assert_eq!(
            to_vec3_round_async(grad_x, 4). await ?, [[[10_f32, 26., 42.], [58., 74.,
            90.], [106., 122., 138.]]]
        );
    }
    let x = Var::new(&[[[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]]]], device)?;
    let y = x.interpolate2d(6, 6)?.reshape(36)?;
    let z = Tensor::new(
        &[
            1_f32,
            02.,
            03.,
            04.,
            05.,
            06.,
            07.,
            08.,
            09.,
            10.,
            11.,
            12.,
            13.,
            14.,
            15.,
            16.,
            17.,
            18.,
            19.,
            20.,
            21.,
            22.,
            23.,
            24.,
            25.,
            26.,
            27.,
            28.,
            29.,
            30.,
            31.,
            32.,
            33.,
            34.,
            35.,
            36.,
        ],
        device,
    )?;
    let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    let grads = loss.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(
        to_vec2_round_async(& grad_x.flatten(0, 2) ?, 4). await ?, [[18_f32, 26., 34.],
        [66., 74., 82.], [114., 122., 130.]]
    );
    let x = Var::new(&[[[[1f32, 2.], [4., 5.]]]], device)?;
    let y = x.interpolate2d(6, 6)?.reshape(36)?;
    let z = Tensor::new(
        &[
            1_f32,
            02.,
            03.,
            04.,
            05.,
            06.,
            07.,
            08.,
            09.,
            10.,
            11.,
            12.,
            13.,
            14.,
            15.,
            16.,
            17.,
            18.,
            19.,
            20.,
            21.,
            22.,
            23.,
            24.,
            25.,
            26.,
            27.,
            28.,
            29.,
            30.,
            31.,
            32.,
            33.,
            34.,
            35.,
            36.,
        ],
        device,
    )?;
    let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    let grads = loss.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(
        to_vec2_round_async(& grad_x.flatten(0, 2) ?, 4). await ?, [[72_f32, 99.], [234.,
        261.]]
    );
    let x = Var::new(&[[[[1f32, 2.], [4., 5.]], [[6f32, 7.], [8., 9.]]]], device)?;
    let y = x.interpolate2d(4, 4)?.reshape(32)?;
    #[rustfmt::skip]
    let z = Tensor::new(
        &[
            1_f32,
            02.,
            03.,
            04.,
            05.,
            06.,
            07.,
            08.,
            09.,
            10.,
            11.,
            12.,
            13.,
            14.,
            15.,
            16.,
            17.,
            18.,
            19.,
            20.,
            21.,
            22.,
            23.,
            24.,
            25.,
            26.,
            27.,
            28.,
            29.,
            30.,
            31.,
            32.,
        ],
        device,
    )?;
    let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    let grads = loss.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(
        to_vec3_round_async(& grad_x.flatten(0, 1) ?, 4). await ?, [[[14_f32, 22.], [46.,
        54.]], [[78., 86.], [110., 118.]]]
    );
    let x = Var::new(&[[[[1f32, 2.], [4., 5.]]], [[[6f32, 7.], [8., 9.]]]], device)?;
    let y = x.interpolate2d(4, 4)?.reshape(32)?;
    #[rustfmt::skip]
    let z = Tensor::new(
        &[
            1_f32,
            02.,
            03.,
            04.,
            05.,
            06.,
            07.,
            08.,
            09.,
            10.,
            11.,
            12.,
            13.,
            14.,
            15.,
            16.,
            17.,
            18.,
            19.,
            20.,
            21.,
            22.,
            23.,
            24.,
            25.,
            26.,
            27.,
            28.,
            29.,
            30.,
            31.,
            32.,
        ],
        device,
    )?;
    let loss = y.unsqueeze(1)?.transpose(0, 1)?.matmul(&z.unsqueeze(1)?)?;
    let grads = loss.backward()?;
    let grad_x = grads.get(&x).context("no grad for x")?;
    assert_eq!(
        to_vec3_round_async(& grad_x.flatten(0, 1) ?, 4). await ?, [[[14_f32, 22.], [46.,
        54.]], [[78., 86.], [110., 118.]]]
    );
    Ok(())
}
async fn binary_grad(device: &Device) -> Result<()> {
    let x = Var::new(&[3f32, 1., -4., -1.], device)?;
    let x = x.as_tensor();
    let y = x.maximum(&(x * 0.1)?)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(x.to_vec1_async::< f32 > (). await ?, [3., 1., - 4., - 1.]);
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [3., 1., - 0.4, - 0.1]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [1., 1., 0.1, 0.1]);
    let y = x.minimum(&(x * 0.1)?)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [0.3, 0.1, - 4., - 1.]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [0.1, 0.1, 1., 1.]);
    let y = x.minimum(x)?;
    let grads = y.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    assert_eq!(y.to_vec1_async::< f32 > (). await ?, [3., 1., - 4., - 1.]);
    assert_eq!(grad_x.to_vec1_async::< f32 > (). await ?, [1., 1., 1., 1.]);
    let x_var = Var::new(&[3f32, 1., -4., -1., 5., 9.], device)?;
    let x = x_var.as_tensor();
    let y_var = Var::new(&[2f32, 7., 1.], device)?;
    let y = y_var.as_tensor();
    let ss = x.reshape((2, 3))?.slice_scatter0(&y.reshape((1, 3))?, 1)?.sqr()?;
    let grads = ss.backward()?;
    let grad_x = grads.get(x).context("no grad for x")?;
    let grad_y = grads.get(y).context("no grad for y")?;
    assert_eq!(ss.to_vec2_async::< f32 > (). await ?, [[9., 1., 16.], [4., 49., 1.]]);
    assert_eq!(
        grad_x.to_vec1_async::< f32 > (). await ?, [6.0, 2.0, - 8.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(grad_y.to_vec1_async::< f32 > (). await ?, [4.0, 14.0, 2.0]);
    Ok(())
}
#[test]
async fn test_flip_backprop() -> Result<()> {
    let device = &Device::Cpu;
    let x = Var::ones((2, 2), DType::F64, device)?;
    let weights = Tensor::arange(1.0, 5.0, device)?.reshape((2, 2))?;
    let y = x.matmul(&weights)?;
    let expected_y = Tensor::from_vec(vec![4.0, 6.0, 4.0, 6.0], (2, 2), device)?;
    candle::test_utils::assert_tensor_eq(&y, &expected_y)?;
    let z = y.flip(&[1])?;
    let expected_z = Tensor::from_vec(vec![6.0, 4.0, 6.0, 4.0], (2, 2), device)?;
    candle::test_utils::assert_tensor_eq(&z, &expected_z)?;
    let loss = z.sum_all()?;
    let grad_store = loss.backward()?;
    let grad_x = grad_store.get_id(x.id()).unwrap();
    let flipped_weights = weights.flip(&[1])?;
    let dloss_dy = Tensor::ones((2, 2), DType::F64, device)?;
    let expected_grad = dloss_dy.matmul(&flipped_weights.t()?)?;
    candle::test_utils::assert_tensor_eq(grad_x, &expected_grad)?;
    Ok(())
}
candle_wasm_tests::test_device!(
    simple_grad, simple_grad_cpu, simple_grad_gpu, simple_grad_metal, simple_grad_wgpu
);
candle_wasm_tests::test_device!(
    sum_grad, sum_grad_cpu, sum_grad_gpu, sum_grad_metal, sum_grad_wgpu
);
candle_wasm_tests::test_device!(
    matmul_grad, matmul_grad_cpu, matmul_grad_gpu, matmul_grad_metal, matmul_grad_wgpu
);
candle_wasm_tests::test_device!(
    grad_descent, grad_descent_cpu, grad_descent_gpu, grad_descent_metal,
    grad_descent_wgpu
);
candle_wasm_tests::test_device!(
    unary_grad, unary_grad_cpu, unary_grad_gpu, unary_grad_metal, unary_grad_wgpu
);
candle_wasm_tests::test_device!(
    binary_grad, binary_grad_cpu, binary_grad_gpu, binary_grad_metal, binary_grad_wgpu
);
}pub mod indexing_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{Device, IndexOp, Tensor};
#[test]
async fn integer_index() -> Result<()> {
    let dev = Device::Cpu;
    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(1)?;
    assert_eq!(result.dims(), & [3]);
    assert_eq!(result.to_vec1_async::< u32 > (). await ?, & [3, 4, 5]);
    let result = tensor.i((.., 2))?;
    assert_eq!(result.dims(), & [2]);
    assert_eq!(result.to_vec1_async::< u32 > (). await ?, & [2, 5]);
    Ok(())
}
#[test]
async fn range_index() -> Result<()> {
    let dev = Device::Cpu;
    let tensor = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((2, 3))?;
    let result = tensor.i(..)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2], [3, 4, 5]]);
    let tensor = Tensor::arange(0u32, 4 * 3, &dev)?.reshape((4, 3))?;
    let result = tensor.i(1..3)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[3, 4, 5], [6, 7, 8]]);
    let result = tensor.i(2..)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[6, 7, 8], [9, 10, 11]]);
    let result = tensor.i(..2)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2], [3, 4, 5]]);
    let result = tensor.i(1..=2)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[3, 4, 5], [6, 7, 8]]);
    let result = tensor.i(..1)?;
    assert_eq!(result.dims(), & [1, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2]]);
    let result = tensor.i(..=1)?;
    assert_eq!(result.dims(), & [2, 3]);
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2], [3, 4, 5]]);
    let result = tensor.i(1..1)?;
    assert_eq!(result.dims(), & [0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & empty);
    #[allow(clippy::reversed_empty_ranges)]
    let result = tensor.i(1..0)?;
    assert_eq!(result.dims(), & [0, 3]);
    let empty: [[u32; 3]; 0] = [];
    assert_eq!(result.to_vec2_async::< u32 > (). await ?, & empty);
    Ok(())
}
#[test]
async fn index_3d() -> Result<()> {
    let tensor = Tensor::from_iter(0..24u32, &Device::Cpu)?.reshape((2, 3, 4))?;
    assert_eq!(tensor.i((0, 0, 0)) ?.to_scalar_async::< u32 > (). await ?, 0);
    assert_eq!(tensor.i((1, 0, 0)) ?.to_scalar_async::< u32 > (). await ?, 12);
    assert_eq!(tensor.i((0, 1, 0)) ?.to_scalar_async::< u32 > (). await ?, 4);
    assert_eq!(tensor.i((0, 1, 3)) ?.to_scalar_async::< u32 > (). await ?, 7);
    assert_eq!(tensor.i((0..2, 0, 0)) ?.to_vec1_async::< u32 > (). await ?, & [0, 12]);
    assert_eq!(
        tensor.i((0..2, .., 0)) ?.to_vec2_async::< u32 > (). await ?, & [[0, 4, 8], [12,
        16, 20]]
    );
    assert_eq!(
        tensor.i((..2, .., 3)) ?.to_vec2_async::< u32 > (). await ?, & [[3, 7, 11], [15,
        19, 23]]
    );
    assert_eq!(
        tensor.i((1, .., 3)) ?.to_vec1_async::< u32 > (). await ?, & [15, 19, 23]
    );
    Ok(())
}
#[test]
async fn slice_assign() -> Result<()> {
    let dev = Device::Cpu;
    let tensor = Tensor::arange(0u32, 4 * 5, &dev)?.reshape((4, 5))?;
    let src = Tensor::arange(0u32, 2 * 3, &dev)?.reshape((3, 2))?;
    let out = tensor.slice_assign(&[1..4, 3..5], &src)?;
    assert_eq!(
        out.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2, 3, 4], [5, 6, 7, 0, 1], [10,
        11, 12, 2, 3], [15, 16, 17, 4, 5]]
    );
    let out = tensor.slice_assign(&[0..3, 0..2], &src)?;
    assert_eq!(
        out.to_vec2_async::< u32 > (). await ?, & [[0, 1, 2, 3, 4], [2, 3, 7, 8, 9], [4,
        5, 12, 13, 14], [15, 16, 17, 18, 19]]
    );
    Ok(())
}
}pub mod layout_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, Device, IndexOp, Result, Tensor};
use candle as candle;
async fn contiguous(device: &Device) -> Result<()> {
    let tensor = Tensor::arange(0u32, 24u32, device)?.reshape((2, 3, 4))?;
    assert_eq!(
        tensor.to_vec3_async::< u32 > (). await ?, & [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9,
        10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.t() ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, & [[[0, 4, 8],
        [1, 5, 9], [2, 6, 10], [3, 7, 11]], [[12, 16, 20], [13, 17, 21], [14, 18, 22],
        [15, 19, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 1) ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, &
        [[[0, 1, 2, 3], [12, 13, 14, 15]], [[4, 5, 6, 7], [16, 17, 18, 19]], [[8, 9, 10,
        11], [20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 1) ?.flatten_all() ?.to_vec1_async::< u32 > (). await ?, &
        [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21,
        22, 23]
    );
    assert_eq!(
        tensor.i(1..) ? .transpose(0, 1) ? .contiguous() ? .to_vec3_async::< u32 > ().
        await ?, & [[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 2) ?.contiguous() ?.to_vec3_async::< u32 > (). await ?, &
        [[[0, 12], [4, 16], [8, 20]], [[1, 13], [5, 17], [9, 21]], [[2, 14], [6, 18],
        [10, 22]], [[3, 15], [7, 19], [11, 23]]]
    );
    Ok(())
}
candle_wasm_tests::test_device!(
    contiguous, contiguous_cpu, contiguous_gpu, contiguous_metal, contiguous_wgpu
);
#[test]
async fn strided_blocks() -> Result<()> {
    use candle::Device::Cpu;
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 26u32, &Cpu)?.i(2..)?.reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 2);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i(1)?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 12);
            assert_eq!(len, 12);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?.contiguous()?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 8);
            assert_eq!(
                tensor.to_vec2_async::< u32 > (). await ?, & [[4, 5, 6, 7], [16, 17, 18,
                19]]
            );
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_len, block_start_index } => {
            assert_eq!(block_len, 4);
            assert_eq!(block_start_index.collect::< Vec < _ >> (), & [4, 16])
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.t()?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
            assert_eq!(block_len, 1);
            assert_eq!(
                block_start_index.collect::< Vec < _ >> (), & [0, 4, 8, 1, 5, 9, 2, 6,
                10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]
            )
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.transpose(0, 1)?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => panic!("unexpected block structure"),
        candle::StridedBlocks::MultipleBlocks { block_start_index, block_len } => {
            assert_eq!(block_len, 4);
            assert_eq!(
                block_start_index.collect::< Vec < _ >> (), & [0, 12, 4, 16, 8, 20]
            )
        }
    };
    Ok(())
}
async fn layout(device: &Device) -> Result<()> {
    let rs: usize = 14;
    let a: usize = 12;
    let b: usize = 13;
    let data1 = Tensor::ones((1, b, a, rs), candle::DType::U32, &Device::Cpu)?;
    let data1 = data1.reshape((1, b, a, rs))?;
    let data2 = data1.to_device_async(device).await?;
    let index1 = data1.i((.., .., 3..6, ..4))?;
    let index2 = data2.i((.., .., 3..6, ..4))?;
    let result1 = index1.reshape((b, 3, 4))?;
    let result2 = index2.reshape((b, 3, 4))?;
    assert_eq!(
        result1.to_vec3_async::< u32 > (). await ?, result2.to_vec3_async::< u32 > ().
        await ?
    );
    let copy1 = index1.copy()?;
    let copy2 = index2.copy()?;
    let result1 = copy1.reshape((b, 3, 4))?;
    let result2 = copy2.reshape((b, 3, 4))?;
    assert_eq!(
        result1.to_vec3_async::< u32 > (). await ?, result2.to_vec3_async::< u32 > ().
        await ?
    );
    let result1 = index1.sum_all()?.to_vec0_async::<u32>().await?;
    let result2 = index2.sum_all()?.to_vec0_async::<u32>().await?;
    assert_eq!(result1, result2);
    Ok(())
}
candle_wasm_tests::test_device!(
    layout, layout_cpu, layout_gpu, layout_metal, layout_wgpu
);
}pub mod matmul_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, DType, Device, IndexOp, Result, Tensor};
async fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::< f32 > (). await ?, & [[7.0f32, 10.0], [15.0, 22.0]]);
    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::< f32 > (). await ?, & [& [3.0, 4.0], & [6.0, 8.0]]);
    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2_async::< f32 > (). await ?, & [& [16., 19.], & [52., 64.]]);
    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3_async::< f32 > (). await ?, & expected);
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(! a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), & [6, 1, 2]);
    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(! b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), & [6, 1, 3]);
    assert_eq!(a_tt.matmul(& b) ?.to_vec3_async::< f32 > (). await ?, & expected);
    assert_eq!(a.matmul(& b_tt) ?.to_vec3_async::< f32 > (). await ?, & expected);
    assert_eq!(a_tt.matmul(& b_tt) ?.to_vec3_async::< f32 > (). await ?, & expected);
    Ok(())
}
async fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2_async::< f32 > (). await ?, & [[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}
async fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), & [3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            assert!(sum_diff2.to_vec0_async::< f32 > (). await ? < 1e-6)
        }
    }
    Ok(())
}
#[test]
async fn tensor_dot() -> Result<()> {
    let lhs = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
    let rhs = Tensor::new(&[4., 5., 6.], &Device::Cpu)?;
    let expected = Tensor::new(32., &Device::Cpu)?;
    let dot_ret = lhs.dot(&rhs)?;
    candle::test_utils::assert_tensor_eq(&dot_ret, &expected)?;
    Ok(())
}
#[test]
async fn tensor_mv() -> Result<()> {
    let mat = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let vec = Tensor::new(&[1., 1., 1.], &Device::Cpu)?;
    let expected = Tensor::new(&[6., 15.], &Device::Cpu)?;
    let mv_ret = mat.mv(&vec)?;
    candle::test_utils::assert_tensor_eq(&mv_ret, &expected)?;
    Ok(())
}
async fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), & [1, 32]);
    Ok(())
}
async fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    Ok(())
}
candle_wasm_tests::test_device!(
    matmul, matmul_cpu, matmul_gpu, matmul_metal, matmul_wgpu
);
candle_wasm_tests::test_device!(
    matmul_bf16, matmul_bf16_cpu, matmul_bf16_gpu, matmul_bf16_metal, matmul_bf16_wgpu
);
candle_wasm_tests::test_device!(
    broadcast_matmul, broadcast_matmul_cpu, broadcast_matmul_gpu, broadcast_matmul_metal,
    broadcast_matmul_wgpu
);
candle_wasm_tests::test_device!(
    squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal, squeeze_mm_wgpu
);
candle_wasm_tests::test_device!(
    mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal, mm_layout_wgpu
);
#[cfg(feature = "wgpu")]
#[test]
async fn test_matmul_kernels_wgpu() -> Result<()> {
    use candle::wgpu::MatmulAlgorithm;
    let algs = vec![
        MatmulAlgorithm::Matmul32_64, MatmulAlgorithm::Matmul32_64B,
        MatmulAlgorithm::Matmul1_64B, MatmulAlgorithm::Matmul1_64_32B,
        MatmulAlgorithm::Matmul1_32_32B, MatmulAlgorithm::Matmul7,
        MatmulAlgorithm::Matmul1, MatmulAlgorithm::MatmulX, MatmulAlgorithm::Matmul16_16,
        MatmulAlgorithm::Matmul32_32, MatmulAlgorithm::Matmul64_64,
        MatmulAlgorithm::Matmul64_64_8_8, MatmulAlgorithm::Matmul24_24,
        MatmulAlgorithm::Matmul24_48, MatmulAlgorithm::Matmul24_24B,
        MatmulAlgorithm::Matmul24_48B,
    ];
    let device = Device::new_wgpu_async(0).await?;
    if let Device::Wgpu(wgpu) = &device {
        for alg in algs {
            wgpu.inner_device().set_extension(alg.clone());
            for tpa in [true, false] {
                for tpb in [true, false] {
                    for use_start_offset in [true, false] {
                        for tpb_batch in [true, false] {
                            for tpa_batch in [true, false] {
                                big_matmul_wgpu(
                                    &device,
                                    tpa,
                                    tpb,
                                    use_start_offset,
                                    tpb_batch,
                                    tpa_batch,
                                ).await?;
                            }
                        }
                    }
                }
            }
            matmul(&device).await?;
            broadcast_matmul(&device).await?;
            squeeze_mm(&device).await?;
            mm_layout(&device).await?;
        }
    }
    Ok(())
}
#[cfg(feature = "wgpu")]
async fn big_matmul_wgpu(
    device: &Device,
    tpa: bool,
    tpb: bool,
    use_start_offset: bool,
    tpb_batch: bool,
    tpa_batch: bool,
) -> Result<()> {
    use candle::D;
    let b = 1;
    let m = 63;
    let n = 63;
    let k = 63;
    let start_offset = if use_start_offset { 100 } else { 0 };
    let lhs1 = Tensor::rand(0f32, 100f32, b * k * m + start_offset, &Device::Cpu)?
        .to_dtype(DType::U32)?
        .to_dtype(DType::F32)?
        .i(start_offset..)?;
    let rhs1 = Tensor::rand(0f32, 100f32, b * k * n + start_offset, &Device::Cpu)?
        .to_dtype(DType::U32)?
        .to_dtype(DType::F32)?
        .i(start_offset..)?;
    let lhs;
    if tpa_batch {
        if tpa {
            lhs = lhs1
                .reshape((m, k, b))?
                .transpose(D::Minus1, D::Minus2)?
                .transpose(0, 1)?;
        } else {
            lhs = lhs1.reshape((k, m, b))?.transpose(0, 2)?;
        }
    } else if tpa {
        lhs = lhs1.reshape((b, k, m))?.transpose(D::Minus1, D::Minus2)?;
    } else {
        lhs = lhs1.reshape((b, m, k))?;
    }
    let rhs;
    if tpb_batch {
        if tpb {
            rhs = rhs1
                .reshape((k, n, b))?
                .transpose(D::Minus1, D::Minus2)?
                .transpose(0, 1)?;
        } else {
            rhs = rhs1.reshape((n, k, b))?.transpose(0, 2)?;
        }
    } else if tpb {
        rhs = rhs1.reshape((b, n, k))?.transpose(D::Minus1, D::Minus2)?;
    } else {
        rhs = rhs1.reshape((b, k, n))?;
    }
    let t1 = lhs.matmul(&rhs)?.reshape((b, m, n))?;
    let lhs = lhs.to_device_async(device).await?;
    let rhs = rhs.to_device_async(device).await?;
    let t2 = lhs.matmul(&rhs)?.reshape((b, m, n))?;
    let m = to_vec3_round_async(&t1, 3).await?;
    let m2 = to_vec3_round_async(&t2, 3).await?;
    assert_eq!(m, m2);
    Ok(())
}
}pub mod pool_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils, Device, IndexOp, Result, Tensor};
async fn avg_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[0.5f32, 1.], [1., 1.]]);
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 2, 8), dev)?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(
        pool.to_vec2_async::< f32 > (). await ?, [[5. / 4., 6. / 4., 6. / 4., 1.]]
    );
    Ok(())
}
async fn max_pool2d(dev: &Device) -> Result<()> {
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t = Tensor::from_vec(data, (1, 1, 4, 4), dev)?;
    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[2f32, 3.], [5., 1.]]);
    let t = t.reshape((1, 1, 2, 8))?;
    let pool = t.max_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(pool.to_vec2_async::< f32 > (). await ?, [[2.0, 3.0, 5.0, 1.0]]);
    Ok(())
}
async fn avg_pool2d_pytorch(dev: &Device) -> Result<()> {
    if dev.is_metal() {
        return Ok(());
    }
    let t = Tensor::new(
            &[
                0.4056f32,
                -0.8689,
                -0.0773,
                -1.5630,
                -2.8012,
                -1.5059,
                0.3972,
                1.0852,
                0.4997,
                3.0616,
                1.6541,
                0.0964,
                -0.8338,
                -1.6523,
                -0.8323,
                -0.1699,
                0.0823,
                0.3526,
                0.6843,
                0.2395,
                1.2279,
                -0.9287,
                -1.7030,
                0.1370,
                0.6047,
                0.3770,
                -0.6266,
                0.3529,
                2.2013,
                -0.6836,
                0.2477,
                1.3127,
            ],
            dev,
        )?
        .reshape((1, 2, 4, 4))?;
    if !dev.is_wgpu() {
        let pool = t.avg_pool2d(2)?.squeeze(0)?;
        assert_eq!(
            to_vec3_round_async(& pool, 4). await ?, [[[- 1.1926, - 0.0395], [0.2688,
            0.1871]], [[0.1835, - 0.1606], [0.6249, 0.3217]]]
        );
    }
    let pool = t.avg_pool2d(3)?.squeeze(0)?;
    assert_eq!(to_vec3_round_async(& pool, 4). await ?, [[[0.085]], [[0.0078]]]);
    let t = t.reshape((1, 1, 4, 8))?;
    let pool = t.avg_pool2d(2)?.squeeze(0)?.squeeze(0)?;
    assert_eq!(
        to_vec2_round_async(& pool, 4). await ?, [[0.7745, 0.0276, - 1.6983, 0.12],
        [0.3542, 0.1625, 0.4542, - 0.0014]]
    );
    Ok(())
}
async fn upsample_nearest2d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 6f32, dev)?.reshape((1, 1, 2, 3))?;
    let upsampled = t.upsample_nearest2d(4, 6)?.i(0)?.i(0)?;
    assert_eq!(
        t.i(0) ?.i(0) ?.to_vec2_async::< f32 > (). await ?, [[0.0, 1.0, 2.0], [3.0, 4.0,
        5.0]]
    );
    assert_eq!(
        upsampled.to_vec2_async::< f32 > (). await ?, [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 5.0, 5.0], [3.0, 3.0, 4.0,
        4.0, 5.0, 5.0]]
    );
    Ok(())
}
async fn upsample_nearest1d(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 3f32, dev)?.reshape((1, 1, 3))?;
    let upsampled = t.upsample_nearest1d(6)?.i(0)?.i(0)?;
    assert_eq!(t.i(0) ?.i(0) ?.to_vec1_async::< f32 > (). await ?, [0.0, 1.0, 2.0]);
    assert_eq!(
        upsampled.to_vec1_async::< f32 > (). await ?, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    );
    Ok(())
}
candle_wasm_tests::test_device!(
    avg_pool2d, avg_pool2d_cpu, avg_pool2d_gpu, avg_pool2d_metal, avg_pool2d_wgpu
);
candle_wasm_tests::test_device!(
    avg_pool2d_pytorch, avg_pool2d_pytorch_cpu, avg_pool2d_pytorch_gpu,
    avg_pool2d_pytorch_metal, avg_pool2d_pytorch_wgpu
);
candle_wasm_tests::test_device!(
    max_pool2d, max_pool2d_cpu, max_pool2d_gpu, max_pool2d_metal, max_pool2d_wgpu
);
candle_wasm_tests::test_device!(
    upsample_nearest1d, upsample_nearest1d_cpu, upsample_nearest1d_gpu,
    upsample_nearest1d_metal, upsample_nearest1d_wgpu
);
candle_wasm_tests::test_device!(
    upsample_nearest2d, upsample_nearest2d_cpu, upsample_nearest2d_gpu,
    upsample_nearest2d_metal, upsample_nearest2d_wgpu
);
}pub mod tensor_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils, DType, Device, IndexOp, Result, Tensor, D};
use float8::F8E4M3;
async fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}
async fn ones(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8) {
        assert_eq!(
            Tensor::ones((2, 3), DType::U8, device) ?.to_vec2_async::< u8 > (). await ?,
            [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::U32) {
        assert_eq!(
            Tensor::ones((2, 3), DType::U32, device) ?.to_vec2_async::< u32 > (). await
            ?, [[1, 1, 1], [1, 1, 1]],
        );
    }
    if device.is_dtype_available(DType::I64) {
        assert_eq!(
            Tensor::ones((2, 3), DType::I64, device) ?.to_vec2_async::< i64 > (). await
            ?, [[1, 1, 1], [1, 1, 1]],
        );
    }
    assert_eq!(
        Tensor::ones((2, 3), DType::F32, device) ?.to_vec2_async::< f32 > (). await ?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    if !device.is_metal() && device.is_dtype_available(DType::F64) {
        assert_eq!(
            Tensor::ones((2, 3), DType::F64, device) ?.to_vec2_async::< f64 > (). await
            ?, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        );
    }
    if device.is_dtype_available(DType::F16) {
        assert_eq!(
            Tensor::ones((2, 3), DType::F16, device) ?.to_vec2_async::< half::f16 > ().
            await ?, [[half::f16::from_f32(1.0), half::f16::from_f32(1.0),
            half::f16::from_f32(1.0)], [half::f16::from_f32(1.0),
            half::f16::from_f32(1.0), half::f16::from_f32(1.0)]],
        );
        assert_eq!(
            Tensor::ones((2, 3), DType::BF16, device) ?.to_vec2_async::< half::bf16 > ().
            await ?, [[half::bf16::from_f32(1.0), half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0)], [half::bf16::from_f32(1.0),
            half::bf16::from_f32(1.0), half::bf16::from_f32(1.0)]],
        );
        if !device.is_metal() {
            assert_eq!(
                Tensor::ones((2, 3), DType::F8E4M3, device) ?.to_vec2_async::< F8E4M3 >
                (). await ?, [[F8E4M3::from_f32(1.), F8E4M3::from_f32(1.),
                F8E4M3::from_f32(1.)], [F8E4M3::from_f32(1.), F8E4M3::from_f32(1.),
                F8E4M3::from_f32(1.)]],
            );
        }
    }
    Ok(())
}
async fn full(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((3, 4), DType::U32, device)?;
    tensor.const_set(42u32.into())?;
    assert_eq!(
        tensor.to_vec2_async::< u32 > (). await ?, [[42, 42, 42, 42], [42, 42, 42, 42],
        [42, 42, 42, 42]]
    );
    tensor.i((.., 2))?.const_set(1337u32.into())?;
    assert_eq!(
        tensor.to_vec2_async::< u32 > (). await ?, [[42, 42, 1337, 42], [42, 42, 1337,
        42], [42, 42, 1337, 42]]
    );
    tensor.i((2, ..))?.const_set(1u32.into())?;
    assert_eq!(
        tensor.to_vec2_async::< u32 > (). await ?, [[42, 42, 1337, 42], [42, 42, 1337,
        42], [1, 1, 1, 1]]
    );
    Ok(())
}
async fn const_set(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::full(42u32, (2, 3), device) ?.to_vec2_async::< u32 > (). await ?, [[42,
        42, 42], [42, 42, 42]],
    );
    Ok(())
}
async fn arange(device: &Device) -> Result<()> {
    if device.is_dtype_available(DType::U8) {
        assert_eq!(
            Tensor::arange(0u8, 5u8, device) ?.to_vec1_async::< u8 > (). await ?, [0, 1,
            2, 3, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 2, device) ?.to_vec1_async::< u8 > (). await ?,
            [0, 2, 4],
        );
        assert_eq!(
            Tensor::arange_step(0u8, 5u8, 3, device) ?.to_vec1_async::< u8 > (). await ?,
            [0, 3],
        );
    }
    if device.is_dtype_available(DType::I64) {
        assert_eq!(
            Tensor::arange_step(5i64, 0i64, - 1, device) ?.to_vec1_async::< i64 > ().
            await ?, [5, 4, 3, 2, 1],
        );
    }
    if !device.is_metal() && device.is_dtype_available(DType::F8E4M3) {
        assert_eq!(
            Tensor::arange_step(F8E4M3::from_f32(0.), F8E4M3::from_f32(5.),
            F8E4M3::from_f32(2.), device) ? .to_vec1_async::< F8E4M3 > (). await ?,
            [F8E4M3::from_f32(0.), F8E4M3::from_f32(2.), F8E4M3::from_f32(4.),],
        );
    }
    Ok(())
}
async fn add_mul(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], device)?;
    let dim1 = tensor.dims1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1_async().await?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}
async fn tensor_2d(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content, data);
    Ok(())
}
async fn clamp(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.clamp(1.5, 6.2)?;
    assert_eq!(
        tensor.to_vec2_async::< f32 > (). await ?, [[3.0, 1.5, 4.0, 1.5, 5.0], [2.0, 1.5,
        6.2, 6.2, 2.0]],
    );
    Ok(())
}
async fn asort(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1.1, 5.], [2.1, 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let indexes = tensor.arg_sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    let indexes = tensor.arg_sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    let (sorted, indexes) = tensor.sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    assert_eq!(
        sorted.to_vec2_async::< f32 > (). await ?, [[1.0, 1.1, 3.0, 4.0, 5.0], [1.0, 2.0,
        2.1, 7.0, 8.0]]
    );
    let (sorted, indexes) = tensor.sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2_async::< u32 > (). await ?, [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    assert_eq!(
        sorted.to_vec2_async::< f32 > (). await ?, [[5.0, 4.0, 3.0, 1.1, 1.0], [8.0, 7.0,
        2.1, 2.0, 1.0]]
    );
    Ok(())
}
/// Test sorting a large tensor that exceeds 1024 elements.
async fn asort_big(device: &Device) -> Result<()> {
    if device.is_metal() {
        return Ok(());
    }
    const SIZE: usize = 2000;
    let data: Vec<f32> = (0..SIZE).map(|x| (SIZE - x) as f32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    let indexes = tensor.arg_sort_last_dim(true)?;
    let expected_indexes: Vec<u32> = (0..SIZE).rev().map(|x| x as u32).collect();
    assert_eq!(indexes.to_vec1_async::< u32 > (). await ?, expected_indexes);
    let indexes = tensor.arg_sort_last_dim(false)?;
    let expected_indexes: Vec<u32> = (0..SIZE).map(|x| x as u32).collect();
    assert_eq!(indexes.to_vec1_async::< u32 > (). await ?, expected_indexes);
    Ok(())
}
async fn unary_op(device: &Device) -> Result<()> {
    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        to_vec2_round_async(& tensor.gelu() ?, 4). await ?, [[- 0.0036, 0.8412, 3.9999, -
        0.046, 0.3457], [2.6911, - 0.0647, - 0.1091, 1.7353, 2.7933]]
    );
    if device.is_dtype_available(DType::F16) {
        let t_f16 = tensor.to_dtype(DType::F16)?.gelu()?.to_dtype(DType::F32)?;
        let max_diff = (tensor.gelu()? - t_f16)?.flatten_all()?.max(0)?;
        assert!(max_diff.to_vec0_async::< f32 > (). await ? < 5e-3);
        assert_eq!(
            to_vec2_round_async(& tensor.gelu_erf() ?, 4). await ?, [[- 0.004, 0.8413,
            3.9999, - 0.046, 0.3457], [2.6906, - 0.0647, - 0.1091, 1.7353, 2.7928]]
        );
    }
    assert_eq!(
        to_vec2_round_async(& tensor.erf() ?, 4). await ?, [[- 1.0, 0.8427, 1.0, -
        0.1125, 0.5205], [0.9999, - 0.9891, - 0.3079, 0.9891, 0.9999]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.silu() ?, 4). await ?, [[- 0.1423, 0.7311, 3.9281, -
        0.0475, 0.3112], [2.53, - 0.2553, - 0.1205, 1.5447, 2.6395]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.ceil() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 0.0,
        1.0], [3.0, - 1.0, - 0.0, 2.0, 3.0]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.floor() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 1.0,
        0.0], [2.0, - 2.0, - 1.0, 1.0, 2.0]]
    );
    assert_eq!(
        to_vec2_round_async(& tensor.round() ?, 4). await ?, [[- 3.0, 1.0, 4.0, - 0.0,
        1.0], [3.0, - 2.0, - 0.0, 2.0, 3.0]]
    );
    let tensor = Tensor::new(&[2997.9246, 314.15926f32], device)?;
    assert_eq!(
        to_vec1_round_async(& tensor.round_to(2) ?, 4). await ?, [2997.92, 314.16]
    );
    assert_eq!(
        to_vec1_round_async(& tensor.round_to(- 2) ?, 4). await ?, [3000.0, 300.]
    );
    let tensor = Tensor::new(
        &[-1.01f32, -0.9, -0.1, 0.0, -0.0, 0.1, 0.9, 1.0, 1.1],
        device,
    )?;
    assert_eq!(
        tensor.sign() ?.to_vec1_async::< f32 > (). await ?, [- 1., - 1., - 1., 0., 0.,
        1., 1., 1., 1.]
    );
    let tensor = Tensor::new(&[-1.0f32, 0., -2., 3.], device)?;
    let y = tensor.elu(2.)?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 1.2642, 0.0000, - 1.7293, 3.0000]
    );
    let y = tensor.reshape((2, 2))?.t()?.elu(2.)?.flatten_all()?;
    assert_eq!(
        to_vec1_round_async(& y, 4). await ?, [- 1.2642, - 1.7293, 0.0000, 3.0000]
    );
    Ok(())
}
async fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2_async().await?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);
    let min = tensor1.minimum(&(&tensor2 * 0.5)?)?;
    let max = tensor1.maximum(&(&tensor2 * 0.5)?)?;
    assert_eq!(
        min.to_vec2_async::< f32 > (). await ?, [[2.5, 1.0, 2.5, 1.0, 2.5], [1.0, 0.5,
        3.5, 4.0, 1.0]],
    );
    assert_eq!(
        max.to_vec2_async::< f32 > (). await ?, [[3.0, 2.5, 4.0, 2.5, 5.0], [2.0, 1.0,
        7.0, 8.0, 2.0]]
    );
    Ok(())
}
async fn ternary_op(device: &Device) -> Result<()> {
    let data = &[[0u8, 1, 0, 1, 0], [1, 1, 1, 0, 0]];
    let ids = Tensor::new(data, device)?;
    let data = &[[0f32, 1., 2., 3., 4.], [5., 6., 7., 8., 9.]];
    let a = Tensor::new(data, device)?;
    let data = &[[10f32, 11., 12., 13., 14.], [15., 16., 17., 18., 19.]];
    let b = Tensor::new(data, device)?;
    let tensor = ids.where_cond(&a, &b)?;
    let dims = tensor.dims();
    assert_eq!(dims, [2, 5]);
    let result: Vec<f32> = tensor.flatten_all()?.to_vec1_async().await?;
    assert_eq!(result, [10., 1., 12., 3., 14., 5., 6., 7., 18., 19.]);
    Ok(())
}
async fn transpose(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?.t()?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2_async::< f32 > (). await ?, & [[3f32, 2.], [1., 1.], [4., 7.],
        [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t() ?.to_vec2_async::< f32 > (). await ?, data);
    assert_eq!(tensor.contiguous() ?.t() ?.to_vec2_async::< f32 > (). await ?, data);
    assert_eq!(((tensor + 1.) ?.t() ? - 1.) ?.to_vec2_async::< f32 > (). await ?, data);
    Ok(())
}
async fn var(device: &Device) -> Result<()> {
    let data = &[
        [0.2035f32, 1.2959, 1.8101, -0.4644],
        [1.5027, -0.3270, 0.5905, 0.6538],
        [-1.5745, 1.3330, -0.5596, -0.6548],
        [0.1264, -0.5080, 1.6420, 0.1992],
    ];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        to_vec2_round_async(& tensor.var_keepdim(1) ?, 4). await ?, & [[1.0631], [0.559],
        [1.4893], [0.8258]]
    );
    Ok(())
}
async fn sum(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.sum_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[8], [15]],
        [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[5, 2, 11], [9,
        7, 17]]],
    );
    assert_eq!(
        tensor.sum_keepdim((0, 2, 1)) ?.to_vec3_async::< u32 > (). await ?, & [[[51]]],
    );
    assert_eq!(
        tensor.t() ?.sum_keepdim(1) ?.t() ?.to_vec3_async::< u32 > (). await ?, & [[[8],
        [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim((2, 1)) ?.to_vec3_async::< u32 > (). await ?, & [[[8 + 15]],
        [[10 + 18]]]
    );
    let data: Vec<u32> = (0..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [7998000]);
    let tensor = tensor.reshape((2000, 2))?;
    assert_eq!(
        tensor.sum_keepdim((0, 1)) ?.to_vec2_async::< u32 > (). await ?, & [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.sum_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1) ?.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998000,
        4000000]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.sum_keepdim((0, 1)) ?.to_vec2_async::< u32 > (). await ?, & [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.sum_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1) ?.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998000,
        4000000]]
    );
    let t1 = tensor.reshape((200, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.sum_keepdim((0, 1, 2)) ?.to_vec3_async::< u32 > (). await ?, &
            [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ? .sum_keepdim(2) ? .sum_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ? .sum_keepdim((1, 2)) ? .to_vec3_async::< u32 > ().
            await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(1) ? .sum_keepdim((0, 2)) ? .to_vec3_async::< u32 > ().
            await ?, & [[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[398000,
            398200, 398400, 398600], [398800, 399000, 399200, 399400], [399600, 399800,
            400000, 400200], [400400, 400600, 400800, 401000], [401200, 401400, 401600,
            401800]]]
        );
    }
    Ok(())
}
async fn min(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.min_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[1], [1]], [[1],
        [2]]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[2, 1, 4], [1, 2,
        8]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.min_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [200]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.min_keepdim(0) ?.min_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1) ?.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[200, 201]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.min_keepdim(0) ?.min_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1) ?.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[200]]
    );
    assert_eq!(
        tensor.min_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[200, 201]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.min_keepdim(0) ? .min_keepdim(2) ? .min_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[200]]]
        );
        assert_eq!(
            tensor.min_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[200, 201,
            202, 203], [204, 205, 206, 207], [208, 209, 210, 211], [212, 213, 214, 215],
            [216, 217, 218, 219]]]
        );
    }
    Ok(())
}
async fn max(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.max_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[4], [9]], [[7],
        [8]]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[3, 1, 7], [8, 5,
        9]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.max_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [3999]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.max_keepdim(0) ?.max_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1) ?.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998, 3999]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.max_keepdim(0) ?.max_keepdim(1) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1) ?.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, &
        [[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[3998, 3999]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.max_keepdim(0) ? .max_keepdim(2) ? .max_keepdim(1) ? .to_vec3_async::<
            u32 > (). await ?, & [[[3999]]]
        );
        assert_eq!(
            tensor.max_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[3980, 3981,
            3982, 3983], [3984, 3985, 3986, 3987], [3988, 3989, 3990, 3991], [3992, 3993,
            3994, 3995], [3996, 3997, 3998, 3999]]]
        );
    }
    Ok(())
}
async fn argmin(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmin_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[1], [0]],
        [[1], [1]]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[1, 0, 0], [0,
        1, 1]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmin_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [0]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.argmin_keepdim(0) ? .argmin_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(1) ? .argmin_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[0, 0]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.argmin_keepdim(0) ? .argmin_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(1) ? .argmin_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[0, 0]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.argmin_keepdim(0) ? .argmin_keepdim(2) ? .argmin_keepdim(1) ?
            .to_vec3_async::< u32 > (). await ?, & [[[0]]]
        );
        assert_eq!(
            tensor.argmin_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[0, 0, 0,
            0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],]]
        );
    }
    Ok(())
}
async fn argmax(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmax_keepdim(2) ?.to_vec3_async::< u32 > (). await ?, & [[[2], [2]],
        [[2], [0]]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[0, 0, 1], [1,
        0, 0]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmax_keepdim(0) ?.to_vec1_async::< u32 > (). await ?, & [3799]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.argmax_keepdim(0) ? .argmax_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(1) ? .argmax_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[1899, 1899]]
    );
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.argmax_keepdim(0) ? .argmax_keepdim(1) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(1) ? .argmax_keepdim(0) ? .to_vec2_async::< u32 > (). await
        ?, & [[0]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0) ?.to_vec2_async::< u32 > (). await ?, & [[1899, 1899]]
    );
    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.argmax_keepdim(0) ? .argmax_keepdim(2) ? .argmax_keepdim(1) ?
            .to_vec3_async::< u32 > (). await ?, & [[[0]]]
        );
        assert_eq!(
            tensor.argmax_keepdim(0) ?.to_vec3_async::< u32 > (). await ?, & [[[189, 189,
            189, 189], [189, 189, 189, 189], [189, 189, 189, 189], [189, 189, 189, 189],
            [189, 189, 189, 189],]]
        );
    }
    Ok(())
}
async fn narrow(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.narrow(2, 1, 2) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0, 4.0],
        [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0, 5.0,
        9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[3.0, 1.0, 4.0],
        [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1) ?.to_vec3_async::< f32 > (). await ?, & [[[2.0, 1.0, 7.0],
        [8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.t() ?.narrow(1, 1, 2) ?.to_vec3_async::< f32 > (). await ?, & [[[1.0,
        5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}
async fn broadcast(device: &Device) -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.broadcast_left((3, 1)) ?.to_vec3_async::< f32 > (). await ?, & [[[3.0,
        1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}
async fn slice_set(device: &Device) -> Result<()> {
    let (b, h, max_t, d) = (2, 4, 7, 3);
    let cache = Tensor::zeros((b, h, max_t, d), DType::F32, device)?;
    let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), device)?;
    cache.slice_set(&tensor, 2, 0)?;
    let cache_t = cache.narrow(2, 0, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    cache.slice_set(&tensor, 2, 1)?;
    let cache_t = cache.narrow(2, 1, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let ones = Tensor::ones((b, h, 1, d), DType::F32, device)?;
    cache.slice_set(&ones, 2, 6)?;
    let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    let diff = (cache.narrow(2, 6, 1)? - 1.)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>()
        .await?;
    assert_eq!(diff, 0.);
    assert!(cache.slice_set(& cache, 0, 0).is_err());
    Ok(())
}
async fn cat(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[3f32, 1., 4.], device)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], device)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], device)?;
    assert_eq!(
        Tensor::cat(& [& t1], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32, 1., 4.],
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32, 1.,
        4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2, & t3], 0) ?.to_vec1_async::< f32 > (). await ?, [3f32,
        1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, device)?;
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 0) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0,
        4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0], [5.0, 5.0, 5.0, 5.0, 5.0], [2.0, 7.0,
        1.0, 8.0, 2.0]]
    );
    assert_eq!(
        Tensor::cat(& [& t1.t() ?, & t2.t() ?], 1) ? .t() ? .to_vec2_async::< f32 > ().
        await ?, [[3.0, 1.0, 4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0], [5.0, 5.0, 5.0,
        5.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0]]
    );
    assert_eq!(
        Tensor::cat(& [& t1, & t2], 1) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0,
        4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0,
        8.0, 2.0]]
    );
    if device.is_dtype_available(DType::I64) {
        let t1 = Tensor::arange(0, 48i64, device)?.reshape((2, 6, 4))?;
        let t2 = Tensor::arange(100, 124i64, device)?.reshape((2, 3, 4))?;
        let t3 = Tensor::arange(10000, 10032i64, device)?.reshape((2, 4, 4))?;
        let t_cat = Tensor::cat(&[&t1, &t2, &t3], 1)?;
        let t1 = t1.t()?.contiguous()?.t()?;
        let t2 = t2.t()?.contiguous()?.t()?;
        let t3 = t3.t()?.contiguous()?.t()?;
        let t_cat2 = Tensor::cat(&[&t1, &t2, &t3], 1)?;
        let diff = t_cat.eq(&t_cat2)?.to_dtype(DType::F32)?.sum_all()?;
        assert_eq!(diff.to_vec0_async::< f32 > (). await ?, 104.0);
        assert_eq!(t_cat.i((0, 0, 0)) ?.to_vec0_async::< i64 > (). await ?, 0);
        assert_eq!(t_cat.i((0, 4, 0)) ?.to_vec0_async::< i64 > (). await ?, 16);
        assert_eq!(t_cat.i((0, 5, 0)) ?.to_vec0_async::< i64 > (). await ?, 20);
        assert_eq!(t_cat.i((1, 5, 0)) ?.to_vec0_async::< i64 > (). await ?, 44);
        assert_eq!(t_cat.i((0, 6, 0)) ?.to_vec0_async::< i64 > (). await ?, 100);
        assert_eq!(t_cat.i((1, 6, 0)) ?.to_vec0_async::< i64 > (). await ?, 112);
        assert_eq!(t_cat.i((0, 6, 1)) ?.to_vec0_async::< i64 > (). await ?, 101);
        assert_eq!(t_cat.i((0, 7, 1)) ?.to_vec0_async::< i64 > (). await ?, 105);
        assert_eq!(t_cat.i((0, 12, 1)) ?.to_vec0_async::< i64 > (). await ?, 10013);
        assert_eq!(t_cat.i((1, 12, 3)) ?.to_vec0_async::< i64 > (). await ?, 10031);
    }
    Ok(())
}
async fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = t.embedding(&ids)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
    );
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
    );
    if device.is_dtype_available(DType::I64) {
        let hs = t.index_select(&ids.to_dtype(DType::I64)?, 0)?;
        assert_eq!(
            hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]
        );
    }
    let ids = Tensor::new(&[u32::MAX, 2u32, u32::MAX], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]]
    );
    Ok(())
}
#[test]
async fn index_select_fail() -> Result<()> {
    let ids = Tensor::new(&[4u32, 2u32, 1u32], &Device::Cpu)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &Device::Cpu)?;
    let hs = t.index_select(&ids, 0);
    assert!(hs.is_err());
    Ok(())
}
async fn cmp(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], device)?;
    assert_eq!(
        t1.eq(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 0], [0, 1], [1, 0]]
    );
    assert_eq!(
        t1.ne(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 1], [1, 0], [0, 1]]
    );
    assert_eq!(
        t1.le(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 0], [1, 1], [1, 1]]
    );
    assert_eq!(
        t1.lt(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[1, 0], [1, 0], [0, 1]]
    );
    assert_eq!(
        t1.gt(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 1], [0, 0], [0, 0]]
    );
    assert_eq!(
        t1.ge(& t2) ?.to_vec2_async::< u8 > (). await ?, & [[0, 1], [0, 1], [1, 0]]
    );
    Ok(())
}
async fn index_select(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype) {
            let ids = ids.to_dtype(dtype)?;
            let hs = t.index_select(&ids, 1)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 2.0, 1.0], [3.0, 5.0,
                4.0], [6.0, 8.0, 7.0], [9.0, 11.0, 10.0]]
            );
            let hs = t.index_select(&ids, 0)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [6.0, 7.0,
                8.0], [3.0, 4.0, 5.0]]
            );
        }
    }
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 2u32, 1u32], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0,
        4.0, 5.0], [0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0, 4.0, 5.0],]
    );
    let ids = Tensor::new(&[1u32, 0u32, 1u32], device)?;
    let t = Tensor::arange(1f32, 5f32, device)?.reshape((2, 2))?;
    assert_eq!(t.to_vec2_async::< f32 > (). await ?, & [[1.0, 2.0], [3.0, 4.0]]);
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]
    );
    Ok(())
}
async fn index_add(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 4.0], [4.0, 10.0], [7.0, 16.0],
        [10.0, 22.0]],
    );
    let init = Tensor::zeros((4, 2), DType::F32, device)?;
    let ids = Tensor::new(&[1u32, 0u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[3.0, 0.0], [9.0, 3.0], [15.0, 6.0],
        [21.0, 9.0]],
    );
    let init = Tensor::zeros((6, 3), DType::F32, device)?;
    let ids = Tensor::new(&[5u32, 0u32, 1u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[12.0, 14.0, 16.0], [6.0, 7.0, 8.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]
    );
    Ok(())
}
async fn slice_scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let src = Tensor::arange(100f32, 106f32, device)?.reshape((2, 3))?;
    assert_eq!(
        t.slice_scatter0(& src, 0) ?.to_vec2_async::< f32 > (). await ?, & [[100.0,
        101.0, 102.0], [103.0, 104.0, 105.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    assert_eq!(
        t.slice_scatter0(& src, 1) ?.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0,
        2.0], [100.0, 101.0, 102.0], [103.0, 104.0, 105.0], [9.0, 10.0, 11.0]]
    );
    assert_eq!(
        t.slice_scatter0(& src, 2) ?.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0,
        2.0], [3.0, 4.0, 5.0], [100.0, 101.0, 102.0], [103.0, 104.0, 105.0],]
    );
    Ok(())
}
async fn scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 2.0, 3.0, 1.0, 1.0], [6.0, 1.0,
        1.0, 4.0, 5.0], [1.0, 9.0, 1.0, 14.0, 1.0], [11.0, 1.0, 10.0, 1.0, 12.0]]
    );
    let hs = init.scatter(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0, 1.0, 1.0], [5.0, 1.0,
        1.0, 3.0, 4.0], [1.0, 8.0, 1.0, 7.0, 1.0], [10.0, 1.0, 9.0, 1.0, 11.0]]
    );
    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[1.0, 11.0, 6.0], [1.0, 2.0, 9.0],
        [10.0, 1.0, 3.0], [10.0, 8.0, 1.0], [1.0, 5.0, 12.0], [1.0, 1.0, 1.0]]
    );
    let hs = init.scatter(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 10.0, 5.0], [1.0, 1.0, 8.0],
        [9.0, 1.0, 2.0], [6.0, 7.0, 1.0], [1.0, 4.0, 11.0], [1.0, 1.0, 1.0]]
    );
    let hs = {
        let ids = Tensor::new(
            &[[0u32, u32::MAX, 2], [3, 4, u32::MAX], [3, 3, 1], [u32::MAX, u32::MAX, 4]],
            device,
        )?;
        init.scatter(&ids, &t, 0)?
    };
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 1.0], [1.0, 1.0, 8.0], [1.0,
        1.0, 2.0], [6.0, 7.0, 1.0], [1.0, 4.0, 11.0], [1.0, 1.0, 1.0]]
    );
    init.scatter_set(&ids, &t, 0)?;
    assert_eq!(
        init.to_vec2_async::< f32 > (). await ?, & [[0.0, 10.0, 5.0], [1.0, 1.0, 8.0],
        [9.0, 1.0, 2.0], [6.0, 7.0, 1.0], [1.0, 4.0, 11.0], [1.0, 1.0, 1.0]]
    );
    Ok(())
}
async fn gather(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[0u32], [2u32], [1u32], [0u32]], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let hs = t.gather(&ids, 1)?;
    assert_eq!(hs.to_vec2_async::< f32 > (). await ?, & [[0.0], [5.0], [7.0], [9.0]]);
    let ids = Tensor::new(
        &[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 0.0], [5.0, 3.0], [7.0, 7.0],
        [9.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 2u32, 0u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 7.0, 2.0]]);
    let ids = Tensor::new(&[[0u32, 2u32, 0u32], [0u32, 1u32, 1u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 7.0, 2.0], [0.0, 4.0, 5.0]]
    );
    let hs = {
        let ids = Tensor::new(
            &[[0u32, 0u32], [2u32, u32::MAX], [u32::MAX, 1u32], [0u32, 2u32]],
            device,
        )?;
        t.gather(&ids, 1)?
    };
    assert_eq!(
        hs.to_vec2_async::< f32 > (). await ?, & [[0.0, 0.0], [5.0, 0.0], [0.0, 7.0],
        [9.0, 11.0]]
    );
    let t = Tensor::new(
        &[
            [
                [108_f32, -47., 16., -56., -83., -130., 210.],
                [253., 95., 151., 228., -210., -123., -127.],
                [-9., -217., 2., -78., 163., 245., -204.],
                [-246., 79., -238., 88., -226., -184., 171.],
                [8., -48., -153., 234., -34., 166., -153.],
                [124., 0., -10., -61., -242., -15., -238.],
            ],
            [
                [12., -64., -199., 244., -240., 156., -128.],
                [173., -57., 4., -198., 233., -110., 238.],
                [95., 82., 0., 240., 53., -211., 209.],
                [-122., 167., -212., 227., -144., 61., 118.],
                [-63., -146., 200., 244., 168., -167., 116.],
                [-125., -147., 110., -253., -178., -250., -18.],
            ],
            [
                [57., 86., -50., 56., 92., 205., -78.],
                [-137., -156., -18., 248., -61., -239., 14.],
                [-248., -30., -50., -70., -251., 250., -83.],
                [-221., 67., 72., 59., -24., -154., 232.],
                [-144., -23., -74., 5., 93., 171., 205.],
                [46., -77., -38., -226., 246., 161., -17.],
            ],
            [
                [-153., -231., -236., 161., 126., 2., -22.],
                [-229., -41., 209., 164., 234., 160., 57.],
                [223., 254., -186., -162., -46., -160., -102.],
                [65., 30., 213., -253., 59., 224., -154.],
                [-82., -203., -177., 17., 31., -256., -246.],
                [176., -135., -65., 54., -56., 210., 76.],
            ],
            [
                [-10., -245., 168., 124., -14., -33., -178.],
                [25., -43., -39., 132., -89., 169., 179.],
                [187., -215., 32., -133., 87., -7., -168.],
                [-224., -215., -5., -230., -58., -162., 128.],
                [158., -137., -122., -100., -202., -83., 136.],
                [30., -185., -144., 250., 209., -40., 127.],
            ],
            [
                [-196., 108., -245., 122., 146., -228., 62.],
                [-1., -66., 160., 137., 13., -172., -21.],
                [244., 199., -164., 28., 119., -175., 198.],
                [-62., 253., -162., 195., -95., -230., -211.],
                [123., -72., -26., -107., -139., 64., 245.],
                [11., -126., -182., 108., -12., 184., -127.],
            ],
            [
                [-159., 126., 176., 161., 73., -111., -138.],
                [-187., 214., -217., -33., -223., -201., -212.],
                [-61., -120., -166., -172., -95., 53., 196.],
                [-33., 86., 134., -152., 154., -53., 74.],
                [186., -28., -154., -174., 141., -109., 217.],
                [82., 35., 252., 145., 181., 74., -87.],
            ],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[
            [
                [6_u32, 6, 4, 3, 4, 4, 6],
                [3, 3, 2, 4, 4, 4, 6],
                [3, 3, 0, 2, 4, 6, 4],
                [2, 5, 1, 2, 6, 6, 1],
                [2, 1, 6, 5, 3, 2, 3],
                [6, 1, 0, 1, 0, 2, 6],
            ],
            [
                [4, 6, 4, 3, 3, 3, 2],
                [4, 3, 2, 4, 4, 4, 6],
                [2, 3, 0, 2, 4, 6, 4],
                [6, 5, 1, 2, 6, 6, 1],
                [4, 1, 6, 5, 3, 2, 3],
                [1, 1, 0, 1, 0, 2, 6],
            ],
            [
                [3, 6, 4, 3, 3, 3, 2],
                [2, 3, 2, 4, 4, 4, 6],
                [4, 3, 0, 2, 4, 6, 4],
                [0, 5, 1, 2, 6, 6, 1],
                [6, 1, 6, 5, 3, 2, 3],
                [4, 1, 0, 1, 0, 2, 6],
            ],
            [
                [0, 6, 4, 3, 3, 3, 2],
                [5, 3, 2, 4, 4, 4, 6],
                [0, 3, 0, 2, 4, 6, 4],
                [3, 5, 1, 2, 6, 6, 1],
                [0, 1, 6, 5, 3, 2, 3],
                [3, 1, 0, 1, 0, 2, 6],
            ],
        ],
        device,
    )?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 159_f32, 126., 168., 161., - 14., -
        33., - 138.], [- 229., - 41., - 18., 132., - 89., 169., - 212.], [223., 254., 2.,
        - 70., 87., 53., - 168.], [- 221., 253., - 212., 59., 154., - 53., 118.], [-
        144., - 146., - 154., - 107., 31., 171., - 246.], [82., - 147., - 10., - 253., -
        242., 161., - 87.]], [[- 10., 126., 168., 161., 126., 2., - 78.], [25., - 41., -
        18., 132., - 89., 169., - 212.], [- 248., 254., 2., - 70., 87., 53., - 168.], [-
        33., 253., - 212., 59., 154., - 53., 118.], [158., - 146., - 154., - 107., 31.,
        171., - 246.], [- 125., - 147., - 10., - 253., - 242., 161., - 87.]], [[- 153.,
        126., 168., 161., 126., 2., - 78.], [- 137., - 41., - 18., 132., - 89., 169., -
        212.], [187., 254., 2., - 70., 87., 53., - 168.], [- 246., 253., - 212., 59.,
        154., - 53., 118.], [186., - 146., - 154., - 107., 31., 171., - 246.], [30., -
        147., - 10., - 253., - 242., 161., - 87.]], [[108., 126., 168., 161., 126., 2., -
        78.], [- 1., - 41., - 18., 132., - 89., 169., - 212.], [- 9., 254., 2., - 70.,
        87., 53., - 168.], [65., 253., - 212., 59., 154., - 53., 118.], [8., - 146., -
        154., - 107., 31., 171., - 246.], [176., - 147., - 10., - 253., - 242., 161., -
        87.]]]
    );
    let t = Tensor::new(
        &[
            [
                [-117_f32, -175., 69., -163.],
                [200., 242., -21., -67.],
                [179., 150., -126., -75.],
                [-118., 38., -138., -13.],
                [-221., 136., -185., 180.],
                [58., 182., -204., -149.],
            ],
            [
                [3., -148., -58., -154.],
                [-43., 45., -108., 4.],
                [-69., -249., -71., -21.],
                [80., 110., -152., -235.],
                [-88., 7., 92., -250.],
                [-186., 207., -242., 98.],
            ],
            [
                [238., 19., 64., -242.],
                [-150., -97., 218., 58.],
                [111., -233., 204., -212.],
                [-242., -232., 83., 42.],
                [153., 62., -251., 219.],
                [-117., 36., -119., 10.],
            ],
            [
                [215., 159., -169., -27.],
                [-83., 101., -88., 169.],
                [-205., 93., 225., -64.],
                [-162., 240., 214., 23.],
                [-112., 6., 21., 245.],
                [-38., 113., 93., 215.],
            ],
            [
                [91., -188., -148., 101.],
                [74., 203., -35., 55.],
                [-116., -130., -153., -96.],
                [58., 22., -45., -194.],
                [-221., -134., 73., 159.],
                [-203., -254., 31., 235.],
            ],
            [
                [105., -53., 61., 186.],
                [-195., 234., 75., -1.],
                [51., 139., 160., -108.],
                [-173., -167., 161., 19.],
                [83., -246., 156., -222.],
                [109., 39., -149., 137.],
            ],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[
            [[4_u32, 4, 4, 2]],
            [[0, 4, 4, 3]],
            [[1, 5, 3, 4]],
            [[0, 3, 3, 2]],
            [[1, 1, 5, 2]],
            [[1, 4, 5, 4]],
        ],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 221., 136., - 185., - 75.]], [[3.,
        7., 92., - 235.]], [[- 150., 36., 83., 219.]], [[215., 240., 214., - 64.]],
        [[74., 203., 31., - 96.]], [[- 195., - 246., - 149., - 222.]]]
    );
    let t = Tensor::new(
        &[
            [[-162_f32, 202.], [-126., -39.], [35., -65.], [1., 80.]],
            [[37., 248.], [-191., 89.], [117., -40.], [-217., 220.]],
        ],
        device,
    )?;
    let ids = Tensor::new(&[[[1_u32], [0], [1], [1]], [[0], [1], [0], [1]]], device)?;
    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[202.], [- 126.], [- 65.], [80.]],
        [[37.], [89.], [117.], [220.]]]
    );
    let t = Tensor::new(
        &[
            [[-21_f32, -197.], [194., 122.]],
            [[255., -106.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[-130., 238.], [-217., -92.]],
        ],
        device,
    )?;
    let ids = Tensor::new(
        &[[[0_u32, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 1]], [[1, 0], [1, 0]]],
        device,
    )?;
    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3_async::< f32 > (). await ?, & [[[- 21., - 197.], [122., 194.]], [[-
        106., 255.], [- 191., 250.]], [[33., - 117.], [43., 10.]], [[238., - 130.], [-
        92., - 217.]]]
    );
    Ok(())
}
async fn broadcasting(device: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 24f32, device)?.reshape((4, 2, 3))?;
    let t2 = Tensor::new(&[100f32, 200f32], device)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[100.0, 101.0, 102.0], [203.0, 204.0,
        205.0]], [[106.0, 107.0, 108.0], [209.0, 210.0, 211.0]], [[112.0, 113.0, 114.0],
        [215.0, 216.0, 217.0]], [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[100.0, 203.0], [101.0, 204.0], [102.0,
        205.0]], [[106.0, 209.0], [107.0, 210.0], [108.0, 211.0]], [[112.0, 215.0],
        [113.0, 216.0], [114.0, 217.0]], [[118.0, 221.0], [119.0, 222.0], [120.0,
        223.0]]]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 100.0, - 99.0, - 98.0], [- 197.0, -
        196.0, - 195.0]], [[- 94.0, - 93.0, - 92.0], [- 191.0, - 190.0, - 189.0]], [[-
        88.0, - 87.0, - 86.0], [- 185.0, - 184.0, - 183.0]], [[- 82.0, - 81.0, - 80.0],
        [- 179.0, - 178.0, - 177.0]]]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 100.0, - 197.0], [- 99.0, - 196.0],
        [- 98.0, - 195.0]], [[- 94.0, - 191.0], [- 93.0, - 190.0], [- 92.0, - 189.0]],
        [[- 88.0, - 185.0], [- 87.0, - 184.0], [- 86.0, - 183.0]], [[- 82.0, - 179.0], [-
        81.0, - 178.0], [- 80.0, - 177.0]]]
    );
    let t1 = t1.i(2..)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[112.0, 113.0, 114.0], [215.0, 216.0,
        217.0]], [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[112.0, 215.0], [113.0, 216.0], [114.0,
        217.0]], [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 88.0, - 87.0, - 86.0], [- 185.0, -
        184.0, - 183.0]], [[- 82.0, - 81.0, - 80.0], [- 179.0, - 178.0, - 177.0]]]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[- 88.0, - 185.0], [- 87.0, - 184.0],
        [- 86.0, - 183.0]], [[- 82.0, - 179.0], [- 81.0, - 178.0], [- 80.0, - 177.0]]]
    );
    let t3 = Tensor::new(1f32, device)?.broadcast_div(&t2)?;
    let s = t1.broadcast_mul(&t2.reshape((2, 1))?)?;
    let s_div = t1.broadcast_div(&t3.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[1200.0, 1300.0, 1400.0], [3000.0,
        3200.0, 3400.0]], [[1800.0, 1900.0, 2000.0], [4200.0, 4400.0, 4600.0]]]
    );
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, s_div.to_vec3_async::< f32 > (). await ?,
    );
    let s = t1.t()?.broadcast_mul(&t2)?;
    let s_div = t1.t()?.broadcast_div(&t3)?;
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, & [[[1200.0, 3000.0], [1300.0, 3200.0],
        [1400.0, 3400.0]], [[1800.0, 4200.0], [1900.0, 4400.0], [2000.0, 4600.0]]]
    );
    assert_eq!(
        s.to_vec3_async::< f32 > (). await ?, s_div.to_vec3_async::< f32 > (). await ?,
    );
    Ok(())
}
async fn randn(device: &Device) -> Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    let tensor2 = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_ne!(
        tensor.to_vec2_async::< f32 > (). await ?, tensor2.to_vec2_async::< f32 > ().
        await ?
    );
    let tensor = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    let tensor2 = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_ne!(
        tensor.to_vec2_async::< f32 > (). await ?, tensor2.to_vec2_async::< f32 > ().
        await ?
    );
    const N: usize = 2;
    
            let mut v = Vec::new();
            for _ in 0..100 {
                let t = Tensor::randn(0f32, 1f32, N, device)?;
                let vec = t.to_vec1_async::<f32>().await?;
                v.push(vec);
            }
            
    assert!(
        (0..N).all(| i | v.windows(2).any(| pair | pair[0] [i] != pair[1] [i])),
        "There are deterministic values in the randn tensors"
    );
    
            let mut v = Vec::new();
            for _ in 0..100 {
                let t = Tensor::randn(0f32, 1f32, N, device)?;
                let vec = t.to_vec1_async::<f32>().await?;
                v.push(vec);
            }
            
    assert!(
        (0..N).all(| i | v.windows(2).any(| pair | pair[0] [i] != pair[1] [i])),
        "There are deterministic values in the rand tensors"
    );
    Ok(())
}
async fn where_cond(device: &Device) -> Result<()> {
    let cond = Tensor::new(&[0u32, 2u32, 1u32, 0, 0, 0, 35, 255, 53, 0, 29, 0], device)?
        .reshape((4, 3))?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, & [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0,
        7.0, 8.0], [9.0, 10.0, 11.0]]
    );
    let t_f = Tensor::arange(12f32, 24f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t_f.to_vec2_async::< f32 > (). await ?, & [[12.0, 13.0, 14.0], [15.0, 16.0,
        17.0], [18.0, 19.0, 20.0], [21.0, 22.0, 23.0]]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        if device.is_dtype_available(dtype) {
            let cond = cond.to_dtype(dtype)?;
            let hs = cond.where_cond(&t, &t_f)?;
            assert_eq!(
                hs.to_vec2_async::< f32 > (). await ?, & [[12.0, 1.0, 2.0], [15.0, 16.0,
                17.0], [6.0, 7.0, 8.0], [21.0, 10.0, 23.0]]
            );
        }
    }
    Ok(())
}
async fn zero_dim(device: &Device) -> Result<()> {
    let t = Tensor::zeros((4, 0, 1), DType::F32, device)?;
    assert_eq!(t.dims3() ?, (4, 0, 1));
    let t2 = Tensor::zeros((4, 3, 1), DType::F32, device)?;
    let t_cat = Tensor::cat(&[&t, &t2], 1)?;
    assert_eq!(t_cat.dims3() ?, (4, 3, 1));
    let t_cat = Tensor::cat(&[&t, &t], 1)?;
    assert_eq!(t_cat.dims3() ?, (4, 0, 1));
    let t_unary = t.sqrt()?;
    assert_eq!(t_unary.dims3() ?, (4, 0, 1));
    let t_plus = (&t + 1.)?;
    assert_eq!(t_plus.dims3() ?, (4, 0, 1));
    let t_mm = t2.matmul(&t.t()?)?;
    assert_eq!(t_mm.dims3() ?, (4, 3, 0));
    let t_mm = t.matmul(&t2.t()?)?;
    assert_eq!(t_mm.dims3() ?, (4, 0, 3));
    let t_mm = t.t()?.matmul(&t)?;
    assert_eq!(t_mm.dims3() ?, (4, 1, 1));
    Ok(())
}
candle_wasm_tests::test_device!(zeros, zeros_cpu, zeros_gpu, zeros_metal, zeros_wgpu);
candle_wasm_tests::test_device!(ones, ones_cpu, ones_gpu, ones_metal, ones_wgpu);
candle_wasm_tests::test_device!(full, full_cpu, full_gpu, full_metal, full_wgpu);
candle_wasm_tests::test_device!(const_set, cs_cpu, cs_gpu, cs_metal, cs_wgpu);
candle_wasm_tests::test_device!(
    arange, arange_cpu, arange_gpu, arange_metal, arange_wgpu
);
candle_wasm_tests::test_device!(
    add_mul, add_mul_cpu, add_mul_gpu, add_mul_metal, add_mul_wgpu
);
candle_wasm_tests::test_device!(
    tensor_2d, tensor_2d_cpu, tensor_2d_gpu, tensor_2d_metal, tensor_2d_wgpu
);
candle_wasm_tests::test_device!(
    narrow, narrow_cpu, narrow_gpu, narrow_metal, narrow_wgpu
);
candle_wasm_tests::test_device!(
    broadcast, broadcast_cpu, broadcast_gpu, broadcast_metal, broadcast_wgpu
);
candle_wasm_tests::test_device!(slice_set, ss_cpu, ss_gpu, ss_metal, ss_wgpu);
candle_wasm_tests::test_device!(cat, cat_cpu, cat_gpu, cat_metal, cat_wgpu);
candle_wasm_tests::test_device!(sum, sum_cpu, sum_gpu, sum_metal, sum_wgpu);
candle_wasm_tests::test_device!(min, min_cpu, min_gpu, min_metal, min_wgpu);
candle_wasm_tests::test_device!(max, max_cpu, max_gpu, max_metal, max_wgpu);
candle_wasm_tests::test_device!(
    argmax, argmax_cpu, argmax_gpu, argmax_metal, argmax_wgpu
);
candle_wasm_tests::test_device!(
    argmin, argmin_cpu, argmin_gpu, argmin_metal, argmin_wgpu
);
candle_wasm_tests::test_device!(
    transpose, transpose_cpu, transpose_gpu, transpose_metal, transpose_wgpu
);
candle_wasm_tests::test_device!(
    unary_op, unary_op_cpu, unary_op_gpu, unary_op_metal, unary_op_wgpu
);
candle_wasm_tests::test_device!(
    binary_op, binary_op_cpu, binary_op_gpu, binary_op_metal, binary_op_wgpu
);
candle_wasm_tests::test_device!(
    ternary_op, ternary_op_cpu, ternary_op_gpu, ternary_op_metal, ternary_op_wgpu
);
candle_wasm_tests::test_device!(
    embeddings, embeddings_cpu, embeddings_gpu, embeddings_metal, embeddings_wgpu
);
candle_wasm_tests::test_device!(cmp, cmp_cpu, cmp_gpu, cmp_metal, cmp_wgpu);
candle_wasm_tests::test_device!(
    broadcasting, broadcasting_cpu, broadcasting_gpu, broadcasting_metal,
    broadcasting_wgpu
);
candle_wasm_tests::test_device!(
    index_select, index_select_cpu, index_select_gpu, index_select_metal,
    index_select_wgpu
);
candle_wasm_tests::test_device!(
    where_cond, where_cond_cpu, where_cond_gpu, where_cond_metal, where_cond_wgpu
);
candle_wasm_tests::test_device!(
    index_add, index_add_cpu, index_add_gpu, index_add_metal, index_add_wgpu
);
candle_wasm_tests::test_device!(
    gather, gather_cpu, gather_gpu, gather_metal, gather_wgpu
);
candle_wasm_tests::test_device!(
    scatter, scatter_cpu, scatter_gpu, scatter_metal, scatter_add_wgpu
);
candle_wasm_tests::test_device!(
    slice_scatter, slice_scatter_cpu, slice_scatter_gpu, slice_scatter_metal,
    slice_scatter_wgpu
);
candle_wasm_tests::test_device!(randn, randn_cpu, randn_gpu, randn_metal, randn_wgpu);
candle_wasm_tests::test_device!(clamp, clamp_cpu, clamp_gpu, clamp_metal, clamp_wgpu);
candle_wasm_tests::test_device!(asort, asort_cpu, asort_gpu, asort_metal);
candle_wasm_tests::test_device!(
    asort_big, asort_big_cpu, asort_big_gpu, asort_big_metal
);
candle_wasm_tests::test_device!(var, var_cpu, var_gpu, var_metal, var_wgpu);
candle_wasm_tests::test_device!(
    zero_dim, zero_dim_cpu, zero_dim_gpu, zero_dim_metal, zero_dim_wgpu
);
async fn tensor_send_sync(device: &Device) -> Result<()> {
    let tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], device)?;
    for _ in 0..10 {
        let tensor = tensor.clone();
        ({
            let new = tensor.add(&tensor).unwrap();
            let result: Vec<f32> = new.to_vec1_async().await.unwrap();
            assert_eq!(result, vec![2.0f32, 4.0, 6.0]);
        });
    }
    let result: Vec<f32> = tensor.to_vec1_async().await.unwrap();
    assert_eq!(result, vec![1.0f32, 2.0, 3.0]);
    let tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], device)?;
    tensor.device().synchronize_async().await.unwrap();
    let new = ({
            let new = tensor.add(&tensor).unwrap();
            new.device().synchronize_async().await.unwrap();
            new
        })
        ;
    let result: Vec<f32> = new.to_vec1_async().await.unwrap();
    assert_eq!(result, vec![2.0f32, 4.0, 6.0]);
    Ok(())
}
candle_wasm_tests::test_device!(
    tensor_send_sync, tensor_send_sync_cpu, tensor_send_sync_gpu, tensor_send_sync_metal,
    tensor_send_sync_wgpu
);
#[test]
async fn randn_hasneg() -> Result<()> {
    let t = Tensor::randn(0f32, 1f32, 200, &Device::Cpu)?.to_vec1_async::<f32>().await?;
    if t.iter().all(|&v| v >= 0.) {
        candle::bail!("all values in tensors are non-negative")
    }
    Ok(())
}
#[test]
async fn pad_with_same() -> Result<()> {
    let t = Tensor::arange(1f32, 5f32, &Device::Cpu)?.reshape((2, 2))?;
    let t0 = t.pad_with_same(0, 1, 2)?;
    assert_eq!(
        t0.to_vec2_async::< f32 > (). await ?, [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0,
        4.0], [3.0, 4.0]]
    );
    let t1 = t.pad_with_same(1, 1, 2)?;
    assert_eq!(
        t1.to_vec2_async::< f32 > (). await ?, [[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0,
        4.0, 4.0, 4.0]]
    );
    Ok(())
}
#[test]
async fn i64_abs() -> Result<()> {
    let t = Tensor::new(&[-42i64, 1337], &Device::Cpu)?;
    let t = t.abs()?;
    assert_eq!(t.to_vec1_async::< i64 > (). await ?, [42, 1337]);
    Ok(())
}
#[test]
async fn tril_triu_eye() -> Result<()> {
    let t = Tensor::tril2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0,
        0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    );
    let t = Tensor::triu2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0,
        1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    );
    let t = Tensor::eye(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2_async::< f32 > (). await ?, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0,
        0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    );
    Ok(())
}
#[test]
async fn cumsum() -> Result<()> {
    let t = &[3f32, 1., 4., 1., 5.];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(t.cumsum(0) ?.to_vec1_async::< f32 > (). await ?, [3., 4., 8., 9., 14.]);
    let t = t.unsqueeze(1)?;
    assert_eq!(
        t.cumsum(0) ?.to_vec2_async::< f32 > (). await ?, [[3.0], [4.0], [8.0], [9.0],
        [14.0]]
    );
    assert_eq!(
        t.cumsum(1) ?.to_vec2_async::< f32 > (). await ?, [[3.0], [1.0], [4.0], [1.0],
        [5.0]]
    );
    let t = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(
        t.cumsum(1) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 4.0, 8.0, 9.0, 14.0],
        [2.0, 3.0, 10.0, 18.0, 20.0]],
    );
    assert_eq!(
        t.cumsum(0) ?.to_vec2_async::< f32 > (). await ?, [[3.0, 1.0, 4.0, 1.0, 5.0],
        [5.0, 2.0, 11.0, 9.0, 7.0]]
    );
    Ok(())
}
/// A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
/// Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
async fn assert_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
    let a_vec: Vec<f64> = a.to_vec1_async().await?;
    let b_vec: Vec<f64> = b.to_vec1_async().await?;
    assert_eq!(a_vec.len(), b_vec.len());
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        assert!((a - b).abs() < epsilon);
    }
    Ok(())
}
#[test]
async fn log_sum_exp() -> Result<()> {
    let input = Tensor::new(
        &[
            [[1f64, 2., 3.], [4., 5., 6.]],
            [[-1000.0, -999.0, -1001.0], [1000.0, 999.0, 1001.0]],
        ],
        &Device::Cpu,
    )?;
    let output = input.log_sum_exp(D::Minus1)?;
    let expected = Tensor::new(
        &[[3.4076, 6.4076], [-998.5924, 1001.4076]],
        &Device::Cpu,
    )?;
    assert_eq!(output.dims(), expected.dims());
    assert_close(&output.flatten_all()?, &expected.flatten_all()?, 0.00001).await?;
    assert_eq!(
        input.log_sum_exp((0, 1)) ?.to_vec1_async::< f64 > (). await ?, [1000.0, 999.0,
        1001.0]
    );
    assert_eq!(
        input.log_sum_exp(()) ?.to_vec3_async::< f64 > (). await ?, input
        .to_vec3_async::< f64 > (). await ?
    );
    Ok(())
}
#[test]
async fn pow() -> Result<()> {
    let lhs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let rhs = (&lhs - 2.)?;
    let res = lhs.pow(&rhs)?;
    assert_eq!(
        to_vec2_round_async(& res, 3). await ?, [[1.0, 1.0, 3.0], [16.0, 125.0, 1296.0]]
    );
    Ok(())
}
#[test]
async fn test_flip_1d() -> Result<()> {
    let t = Tensor::arange(0.0, 5.0, &Device::Cpu)?.reshape((5,))?;
    let flipped = t.flip(&[0])?;
    let expected = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0, 0.0], (5,), &Device::Cpu)?;
    candle::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}
#[test]
async fn test_flip_2d() -> Result<()> {
    let t = Tensor::arange(0.0, 6.0, &Device::Cpu)?.reshape((2, 3))?;
    let flipped = t.flip(&[0, 1])?;
    let expected = Tensor::from_vec(
        vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        (2, 3),
        &Device::Cpu,
    )?;
    candle::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}
#[test]
async fn test_flip_3d_channels() -> Result<()> {
    let t = Tensor::arange(0.0, 12.0, &Device::Cpu)?.reshape((2, 2, 3))?;
    let flipped = t.flip(&[2])?;
    let expected = Tensor::from_vec(
        vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 8.0, 7.0, 6.0, 11.0, 10.0, 9.0],
        (2, 2, 3),
        &Device::Cpu,
    )?;
    candle::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}
#[test]
async fn tensor_new() -> Result<()> {
    let t1 = Tensor::new(vec![1f32, 2.0, 3.0], &Device::Cpu)?;
    assert_eq!(t1.to_vec1_async::< f32 > (). await ?, [1.0, 2.0, 3.0]);
    let t2 = Tensor::new(vec![vec![1f32, 2., 3.], vec![4., 5., 6.]], &Device::Cpu)?;
    assert_eq!(t2.to_vec2_async::< f32 > (). await ?, [[1., 2., 3.], [4., 5., 6.]]);
    let t3 = Tensor::new(
        vec![
            vec![vec![1f32, 2., 3.], vec![4., 5., 6.]], vec![vec![3f32, 1., 4.], vec![1.,
            5., 9.]],
        ],
        &Device::Cpu,
    )?;
    assert_eq!(
        t3.to_vec3_async::< f32 > (). await ?, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]]
    );
    Ok(())
}
#[test]
async fn tensor_norm() -> Result<()> {
    let t = Tensor::new(&[[3., 4.], [0., 0.]], &Device::Cpu)?;
    let norm = t.norm()?;
    assert_eq!(norm.to_scalar_async::< f64 > (). await ?, 5.);
    Ok(())
}
#[cfg(feature = "cuda")]
#[test]
async fn transfers_cuda_to_device() -> Result<()> {
    use rand::seq::SliceRandom;
    let devices = cudarc::driver::safe::CudaContext::device_count()
        .map_err(candle::cuda::CudaError::from)?;
    if devices < 2 {
        return Ok(());
    }
    let first = Device::new_cuda(0)?;
    let mut data: Vec<u32> = (0..262144).collect();
    let mut rng = rand::rng();
    data.shuffle(&mut rng);
    let t1 = Tensor::from_vec(data, (512, 512), &first)?;
    let second = Device::new_cuda(1)?;
    let t2 = t1.to_device_async(&second).await?;
    assert_ne!(t1.device().as_cuda_device() ?.id(), t2.device().as_cuda_device() ?.id());
    Ok(())
}
#[cfg(feature = "cuda")]
#[test]
async fn allocates_twice_when_transferring_to_same_device() -> Result<()> {
    use std::{ops::Deref, sync::RwLockReadGuard};
    use candle::Storage;
    use rand::seq::SliceRandom;
    let first = Device::new_cuda(0)?;
    let second = Device::new_cuda(0)?;
    let mut data: Vec<u32> = (0..262144).collect();
    let mut rng = rand::rng();
    data.shuffle(&mut rng);
    let t1 = Tensor::from_vec(data, (512, 512), &first)?;
    let t2 = t1.to_device_async(&second).await?;
    let (storage1, _) = t1.storage_and_layout();
    let (storage2, _) = t2.storage_and_layout();
    let extract = |s: RwLockReadGuard<'_, Storage>| match &s.deref() {
        Storage::Cuda(c) => {
            use cudarc::driver::DevicePtr;
            let slice = c.as_cuda_slice::<u32>().unwrap();
            let ptr = slice.device_ptr(slice.stream()).0;
            ptr
        }
        _ => unimplemented!(),
    };
    let id1 = extract(storage1);
    let id2 = extract(storage2);
    assert_ne!(id1, id2);
    Ok(())
}
}pub mod batch_norm {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{test_utils, DType, Device, Tensor};
use candle_nn::{batch_norm, BatchNorm, BatchNormConfig, VarBuilder, VarMap};
#[test]
async fn batch_norm_test() -> Result<()> {
    let running_mean = Tensor::zeros(5, DType::F32, &Device::Cpu)?;
    let running_var = Tensor::ones(5, DType::F32, &Device::Cpu)?;
    let bn = BatchNorm::new_no_bias(5, running_mean.clone(), running_var.clone(), 1e-8)?;
    let input: [f32; 120] = [
        -0.7493,
        -1.0410,
        1.6977,
        -0.6579,
        1.7982,
        -0.0087,
        0.2812,
        -0.1190,
        0.2908,
        -0.5975,
        -0.0278,
        -0.2138,
        -1.3130,
        -1.6048,
        -2.2028,
        0.9452,
        0.4002,
        0.0831,
        1.0004,
        0.1860,
        0.5004,
        0.5539,
        0.9991,
        -0.2540,
        -0.0703,
        -0.3752,
        -0.1096,
        -0.2374,
        1.0258,
        -2.2208,
        -0.0257,
        0.6073,
        -1.1627,
        -0.0964,
        -1.9718,
        1.6577,
        0.1931,
        -0.3692,
        -0.8011,
        0.9059,
        0.4797,
        0.6521,
        -0.0165,
        -0.6683,
        -0.4148,
        2.0649,
        -0.8276,
        1.7947,
        -0.2061,
        0.5812,
        -1.3598,
        1.6192,
        1.0466,
        -0.4423,
        0.4202,
        0.1749,
        0.6969,
        0.2616,
        -0.0369,
        -1.4951,
        -0.0814,
        -0.1877,
        0.0267,
        0.6150,
        0.2402,
        -1.1440,
        -2.0068,
        0.6032,
        -2.6639,
        0.8260,
        0.1085,
        -0.1693,
        1.2805,
        0.7654,
        -0.4930,
        0.3770,
        1.1309,
        0.2303,
        0.2949,
        -0.2634,
        -0.5225,
        0.4269,
        0.6341,
        1.5736,
        0.9827,
        -1.2499,
        0.3509,
        -1.6243,
        -0.8123,
        0.7634,
        -0.3047,
        0.0143,
        -0.4032,
        0.0537,
        0.7022,
        0.8405,
        -1.2221,
        -1.6847,
        -0.0714,
        -0.1608,
        0.5579,
        -1.5858,
        0.4617,
        -0.6480,
        0.1332,
        0.0419,
        -0.9784,
        0.4173,
        1.2313,
        -1.9046,
        -0.1656,
        0.1259,
        0.0763,
        1.4252,
        -0.9115,
        -0.1093,
        -0.3100,
        -0.6734,
        -1.4357,
        0.9205,
    ];
    let input = Tensor::new(&input, &Device::Cpu)?.reshape((2, 5, 3, 4))?;
    let output = bn.forward_train(&input)?;
    assert_eq!(output.dims(), & [2, 5, 3, 4]);
    let output = output.flatten_all()?;
    assert_eq!(
        to_vec1_round_async(& output, 4). await ?, & [- 0.6391, - 0.9414, 1.8965, -
        0.5444, 2.0007, 0.1283, 0.4287, 0.014, 0.4387, - 0.4818, 0.1085, - 0.0842, -
        1.6809, - 2.0057, - 2.6714, 0.8328, 0.2262, - 0.1268, 0.8943, - 0.0123, 0.3377,
        0.3973, 0.8928, - 0.5021, 0.0861, - 0.2324, 0.0451, - 0.0884, 1.2311, - 2.1603,
        0.1327, 0.7939, - 1.055, 0.0589, - 1.9002, 1.8912, 0.2918, - 0.3253, - 0.7993,
        1.0741, 0.6063, 0.7955, 0.0617, - 0.6536, - 0.3754, 2.3461, - 0.8284, 2.0495, -
        0.201, 0.6476, - 1.4446, 1.7665, 1.1493, - 0.4556, 0.4741, 0.2097, 0.7723,
        0.3031, - 0.0186, - 1.5905, 0.053, - 0.0572, 0.165, 0.7746, 0.3862, - 1.0481, -
        1.9422, 0.7624, - 2.6231, 0.9933, 0.2498, - 0.0381, 1.2061, 0.6327, - 0.7681,
        0.2004, 1.0396, 0.037, 0.109, - 0.5125, - 0.8009, 0.2559, 0.4865, 1.5324, 1.1861,
        - 1.1461, 0.5261, - 1.5372, - 0.689, 0.957, - 0.1587, 0.1745, - 0.2616, 0.2156,
        0.8931, 1.0375, - 1.2614, - 1.7691, 0.0015, - 0.0966, 0.6921, - 1.6605, 0.5866, -
        0.6313, 0.226, 0.1258, - 0.9939, 0.5378, 1.3484, - 2.0319, - 0.1574, 0.1568,
        0.1034, 1.5574, - 0.9614, - 0.0967, - 0.313, - 0.7047, - 1.5264, 1.0134]
    );
    let bn2 = BatchNorm::new(
        5,
        running_mean,
        running_var,
        Tensor::new(&[0.5f32], &Device::Cpu)?.broadcast_as(5)?,
        Tensor::new(&[-1.5f32], &Device::Cpu)?.broadcast_as(5)?,
        1e-8,
    )?;
    let output2 = bn2.forward_train(&input)?;
    assert_eq!(output2.dims(), & [2, 5, 3, 4]);
    let output2 = output2.flatten_all()?;
    let diff2 = ((output2 - (output * 0.5)?)? + 1.5)?.sqr()?;
    let sum_diff2 = diff2.sum_keepdim(0)?;
    assert_eq!(to_vec1_round_async(& sum_diff2, 4). await ?, & [0f32]);
    assert_eq!(
        to_vec1_round_async(bn.running_mean(), 4). await ?, & [- 0.0133, 0.0197, -
        0.0153, - 0.0073, - 0.0020]
    );
    assert_eq!(
        to_vec1_round_async(bn.running_var(), 4). await ?, & [0.9972, 0.9842, 0.9956,
        0.9866, 0.9898]
    );
    Ok(())
}
#[test]
async fn train_batch_norm() -> Result<()> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    let bn = batch_norm(1, BatchNormConfig::default(), vb)?;
    let original_mean = bn.running_mean().detach().copy()?;
    let var_map_mean = {
        vm.data().lock().unwrap().get("running_mean").unwrap().clone()
    };
    assert_eq!(
        to_vec1_round_async(bn.running_mean(), 4). await ?,
        to_vec1_round_async(var_map_mean.as_tensor(), 4). await ?,
    );
    let mean_plus_one = {
        let one = original_mean.ones_like()?;
        original_mean.add(&one)?.reshape((1, 1))?
    };
    bn.forward_train(&mean_plus_one)?;
    assert_ne!(
        to_vec1_round_async(bn.running_mean(), 4). await ?, to_vec1_round_async(&
        original_mean, 4). await ?,
    );
    assert_eq!(
        to_vec1_round_async(bn.running_mean(), 4). await ?,
        to_vec1_round_async(var_map_mean.as_tensor(), 4). await ?,
    );
    Ok(())
}
}pub mod cpu_flash_attn {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{DType, Device, Result, Tensor};
use candle_nn::cpu_flash_attention::run_flash_attn_cpu;
#[test]
async fn cpu_flash_attn() -> Result<()> {
    let b = 1;
    let s = 2;
    let h = 1;
    let d = 4;
    let softmax_scale = 1.0f32 / (d as f32).sqrt();
    let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu)?;
    let ground_truth = {
        let att = (q.clone() * softmax_scale as f64)?.matmul(&k.clone().t()?)?;
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        att.matmul(&v.clone())?
    };
    let out = run_flash_attn_cpu::<
        f32,
    >(
        &q.transpose(1, 2)?,
        &k.transpose(1, 2)?,
        &v.transpose(1, 2)?,
        None,
        softmax_scale,
        None,
        None,
    )?;
    let out_arr: Vec<f32> = out.flatten_all()?.to_vec1_async().await?;
    let ground_truth_arr: Vec<f32> = ground_truth.flatten_all()?.to_vec1_async().await?;
    for (a, b) in out_arr.iter().zip(ground_truth_arr.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} {b}");
    }
    Ok(())
}
}pub mod group_norm {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::test_utils::to_vec3_round;
use candle::{Device, Tensor};
use candle_nn::{GroupNorm, Module};
#[test]
async fn group_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::from_vec(vec![1f32; 6], 6, device)?;
    let b = Tensor::from_vec(vec![0f32; 6], 6, device)?;
    let gn2 = GroupNorm::new(w.clone(), b.clone(), 6, 2, 1e-5)?;
    let gn3 = GroupNorm::new(w, b, 6, 3, 1e-5)?;
    let input = Tensor::new(
        &[
            [
                [-0.3034f32, 0.2726, -0.9659],
                [-1.1845, -1.3236, 0.0172],
                [1.9507, 1.2554, -0.8625],
                [1.0682, 0.3604, 0.3985],
                [-0.4957, -0.4461, -0.9721],
                [1.5157, -0.1546, -0.5596],
            ],
            [
                [-1.6698, -0.4040, -0.7927],
                [0.3736, -0.0975, -0.1351],
                [-0.9461, 0.5461, -0.6334],
                [-1.0919, -0.1158, 0.1213],
                [-0.9535, 0.1281, 0.4372],
                [-0.2845, 0.3488, 0.5641],
            ],
        ],
        device,
    )?;
    assert_eq!(
        to_vec3_round_async(& gn2.forward(& input) ?, 4). await ?, & [[[- 0.1653, 0.3748,
        - 0.7866], [- 0.9916, - 1.1220, 0.1353], [1.9485, 1.2965, - 0.6896], [1.2769,
        0.3628, 0.4120], [- 0.7427, - 0.6786, - 1.3578], [1.8547, - 0.3022, - 0.8252]],
        [[- 1.9342, 0.0211, - 0.5793], [1.2223, 0.4945, 0.4365], [- 0.8163, 1.4887, -
        0.3333], [- 1.7960, - 0.0392, 0.3875], [- 1.5469, 0.3998, 0.9561], [- 0.3428,
        0.7970, 1.1845]]]
    );
    assert_eq!(
        to_vec3_round_async(& gn3.forward(& input) ?, 4). await ?, & [[[0.4560, 1.4014, -
        0.6313], [- 0.9901, - 1.2184, 0.9822], [1.4254, 0.6360, - 1.7682], [0.4235, -
        0.3800, - 0.3367], [- 0.3890, - 0.3268, - 0.9862], [2.1325, 0.0386, - 0.4691]],
        [[- 1.8797, 0.0777, - 0.5234], [1.2802, 0.5517, 0.4935], [- 1.0102, 1.5327, -
        0.4773], [- 1.2587, 0.4047, 0.8088], [- 1.9074, 0.1691, 0.7625], [- 0.6230,
        0.5928, 1.0061]]]
    );
    Ok(())
}
}pub mod kv_cache {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
#[test]
async fn kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::Cache::new(0, 16);
    for _ in [0, 1] {
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1_async::< f32 > (). await ?, [1., 2., 3.]);
        let t = Tensor::new(&[4f32], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(data.to_vec1_async::< f32 > (). await ?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0f32, 5., 6., 7.], &Device::Cpu)?;
        cache.append(&t)?;
        let data = cache.current_data()?.unwrap();
        assert_eq!(
            data.to_vec1_async::< f32 > (). await ?, [1., 2., 3., 4., 0., 5., 6., 7.]
        );
        assert_eq!(cache.current_seq_len(), 8);
        cache.reset();
    }
    Ok(())
}
#[test]
async fn rotating_kv_cache() -> Result<()> {
    let mut cache = candle_nn::kv_cache::RotatingCache::new(0, 6);
    for _ in [0, 1] {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.current_seq_len(), 0);
        let data = cache.current_data()?;
        assert!(data.is_none());
        assert_eq!(cache.positions(1), & [0]);
        assert_eq!(cache.positions(2), & [0, 1]);
        let t = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [1., 2., 3.]);
        assert_eq!(cache.positions(0), & [0, 1, 2]);
        assert_eq!(cache.positions(1), & [0, 1, 2, 3]);
        assert_eq!(cache.positions(2), & [0, 1, 2, 3, 4]);
        assert_eq!(cache.positions(3), & [0, 1, 2, 3, 4, 5]);
        assert_eq!(cache.positions(4), & [6, 1, 2, 3, 4, 5]);
        let t = Tensor::new(&[4.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [1., 2., 3., 4.]);
        let t = Tensor::new(&[0., 5., 6., 7.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [6., 7., 3., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(cache.offset(), 2);
        let t = Tensor::new(&[8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [6., 7., 8., 4., 0., 5.]);
        assert_eq!(cache.current_seq_len(), 9);
        assert_eq!(cache.offset(), 3);
        let t = Tensor::new(&[9., 10., 11.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [6., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(cache.offset(), 0);
        let t = Tensor::new(&[12.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [12., 7., 8., 9., 10., 11.]);
        assert_eq!(cache.current_seq_len(), 13);
        assert_eq!(cache.offset(), 1);
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2_async::< u8 > (). await ?, & [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0,
            0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2_async::< u8 > (). await ?, & [[0, 0, 1, 1, 0, 0], [0, 0, 0, 1,
            0, 0], [0, 0, 0, 0, 0, 0]],
        );
        assert_eq!(cache.positions(0), & [12, 7, 8, 9, 10, 11]);
        assert_eq!(cache.positions(2), & [12, 13, 14, 9, 10, 11]);
        assert_eq!(cache.positions(3), & [12, 13, 14, 15, 10, 11]);
        assert_eq!(cache.positions(8), & [13, 14, 15, 16, 17, 18, 19, 20]);
        let t = Tensor::new(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(
            data.to_vec1_async::< f64 > (). await ?, [0., 1., 2., 3., 4., 5., 6., 7., 8.]
        );
        assert_eq!(cache.current_seq_len(), 22);
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.positions(0), & [16, 17, 18, 19, 20, 21]);
        assert_eq!(cache.positions(1), & [22, 17, 18, 19, 20, 21]);
        let mask = cache.attn_mask(1, &Device::Cpu)?;
        assert!(mask.is_none());
        let mask = cache.attn_mask(2, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2_async::< u8 > (). await ?, & [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0,
            0, 0]]
        );
        let mask = cache.attn_mask(3, &Device::Cpu)?.unwrap();
        assert_eq!(
            mask.to_vec2_async::< u8 > (). await ?, & [[0, 1, 1, 0, 0, 0], [0, 0, 1, 0,
            0, 0], [0, 0, 0, 0, 0, 0]]
        );
        let t = Tensor::new(&[42.], &Device::Cpu)?;
        let data = cache.append(&t)?;
        assert_eq!(data.to_vec1_async::< f64 > (). await ?, [42., 4., 5., 6., 7., 8.]);
        assert_eq!(cache.current_seq_len(), 23);
        assert_eq!(cache.offset(), 1);
        cache.reset();
    }
    Ok(())
}
}pub mod layer_norm {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};
#[test]
async fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(
        Tensor::cat(&[&w, &w], 0)?,
        Tensor::cat(&[&b, &b], 0)?,
        1e-8,
    );
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1_async::< f32 > (). await ?, [0.5f32]);
    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3_async::< f32 > (). await ?, [[[3.5f32, - 2.5]]]);
    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        to_vec3_round_async(& res, 4). await ?, [[[- 3.1742, 0.5, 4.1742], [- 3.1742,
        0.5, 4.1742], [4.1742, 0.5, - 3.1742]]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    assert_eq!(to_vec3_round_async(& mean, 4). await ?, [[[0.5], [0.5], [0.5]]]);
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    assert_eq!(to_vec3_round_async(& std, 4). await ?, [[[1.7321], [1.7321], [1.7321]]]);
    Ok(())
}
}pub mod loss {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::test_utils::to_vec0_round;
use candle::{Device, Result, Tensor};
#[test]
async fn nll_and_cross_entropy() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(
        &[
            [1.1050f32, 0.3013, -1.5394, -2.1528, -0.8634],
            [1.0730, -0.9419, -0.1670, -0.6582, 0.5061],
            [0.8318, 1.1154, -0.3610, 0.5351, 1.0830],
        ],
        &cpu,
    )?;
    let target = Tensor::new(&[1u32, 0, 4], &cpu)?;
    let log_softmax = candle_nn::ops::log_softmax(&input, 1)?;
    let loss = candle_nn::loss::nll(&log_softmax, &target)?;
    assert_eq!(to_vec0_round_async(& loss, 4). await ?, 1.1312);
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round_async(& loss, 4). await ?, 1.1312);
    Ok(())
}
#[test]
async fn binary_cross_entropy_with_logit() -> Result<()> {
    let cpu = Device::Cpu;
    let inp = [
        [2.3611f32, -0.8813, -0.5006, -0.2178],
        [0.0419, 0.0763, -1.0457, -1.6692],
        [-1.0494, 0.8111, 1.5723, 1.2315],
        [1.3081, 0.6641, 1.1802, -0.2547],
        [0.5292, 0.7636, 0.3692, -0.8318],
    ];
    let target = [
        [0.0f32, 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
    ];
    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;
    let loss = candle_nn::loss::binary_cross_entropy_with_logit(&inp, &target)?;
    assert_eq!(to_vec0_round_async(& loss, 4). await ?, 0.8224);
    Ok(())
}
#[test]
async fn huber_loss() -> Result<()> {
    let cpu = Device::Cpu;
    let inp = [
        [2.3611f32, -0.8813, -0.5006, -0.2178],
        [0.0419, 0.0763, -1.0457, -1.6692],
        [-1.0494, 0.8111, 1.5723, 1.2315],
        [1.3081, 0.6641, 1.1802, -0.2547],
        [0.5292, 0.7636, 0.3692, -0.8318],
    ];
    let target = [
        [0.0f32, 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
    ];
    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;
    let loss = candle_nn::loss::huber(&inp, &target, 1.0)?;
    assert_eq!(to_vec0_round_async(& loss, 4). await ?, 0.4734);
    let loss = candle_nn::loss::huber(&inp, &target, 0.88)?;
    assert_eq!(to_vec0_round_async(& loss, 4). await ?, 0.4483);
    Ok(())
}
}pub mod one_hot {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Result, Shape, Tensor};
use candle_nn::encoding::one_hot_async;
#[test]
async fn test_i64_one_hot() -> Result<()> {
    let device = candle::Device::Cpu;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let depth = 4;
    let on_value = 1.0;
    let off_value = 0.0;
    let one_hot = one_hot_async::<f32>(indices, depth, on_value, off_value).await?;
    let expected_matrix = [
        [[1., 0., 0., 0.], [0., 0., 1., 0.]],
        [[0., 1., 0., 0.], [0., 0., 0., 0.]],
    ];
    assert_eq!(one_hot.shape(), & Shape::from((2, 2, depth)));
    let matrix = one_hot.to_vec3_async::<f32>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_rank_3_one_hot() -> Result<()> {
    let device = candle::Device::Cpu;
    let indices = Tensor::new(
        vec![vec![vec![0i64, 1], vec![2, 3]], vec![vec![3, 1], vec![1, - 1]],],
        &device,
    )?;
    let depth = 4;
    let on_value = 1.0;
    let off_value = 0.0;
    let one_hot = one_hot_async::<f32>(indices, depth, on_value, off_value).await?;
    let expected_matrix = Tensor::new(
        vec![
            vec![vec![vec![1f32, 0., 0., 0.], vec![0., 1., 0., 0.]], vec![vec![0., 0.,
            1., 0.], vec![0., 0., 0., 1.]],], vec![vec![vec![0., 0., 0., 1.], vec![0.,
            1., 0., 0.]], vec![vec![0., 1., 0., 0.], vec![0., 0., 0., 0.]],],
        ],
        &device,
    )?;
    assert_eq!(one_hot.shape(), expected_matrix.shape());
    assert_eq!(one_hot.dims(), expected_matrix.dims());
    let matrix = one_hot.get(1)?.to_vec3_async::<f32>().await?;
    let expected_matrix = expected_matrix.get(1)?.to_vec3_async::<f32>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_u8_one_cold() -> Result<()> {
    let device = candle::Device::Cpu;
    let depth = 4;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let on_value = 0u8;
    let off_value = 1;
    let one_cold = one_hot_async(indices, depth, on_value, off_value).await?;
    let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]];
    assert_eq!(one_cold.shape(), & Shape::from((2, 2, depth)));
    let matrix = one_cold.to_vec3_async::<u8>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_iter() -> Result<()> {
    let device = candle::Device::Cpu;
    let depth = 4;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let matrix = indices.to_vec2_async::<i64>().await?;
    let (dim1, dim2) = indices.dims2()?;
    let iter = (0..dim1).flat_map(|i| (0..dim2).map(move |j| (i, j)));
    let mut v = vec![0; depth * dim1 * dim2];
    for (i, j) in iter {
        let idx = i * depth * dim2 + j * depth;
        v[idx] = matrix[i][j];
    }
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            let idx = i * depth * dim2 + j * depth;
            assert_eq!(v[idx], value);
        }
    }
    Ok(())
}
}pub mod ops {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_device, test_utils::to_vec3_round, Device, IndexOp, Result, Tensor};
async fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round_async(& t0, 4). await ?, & [[[0.6, 0.5, 0.3636], [0.1111, 0.7143,
        0.5294]], [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]]
    );
    assert_eq!(
        to_vec3_round_async(& t1, 4). await ?, & [[[0.75, 0.1667, 0.3077], [0.25, 0.8333,
        0.6923]], [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]]
    );
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.375, 0.125, 0.5], [0.0667, 0.3333,
        0.6]], [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.375, 0.125, 0.5], [0.0667, 0.3333,
        0.6]], [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]]
    );
    Ok(())
}
async fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t, 4). await ?, & [[[1.019, 0.6794, 4.0762], [0.1674,
        1.6744, 4.521]], [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[1.019, 0.6794, 4.0762], [0.1674,
        1.6744, 4.521]], [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t, 2). await ?, to_vec3_round_async(& t2, 2). await ?
    );
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0_async::<f32>()
        .await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t, 4). await ?, & [[[0.7673, - 2.6726, 3.0071], [- 0.7247,
        0.0, 3.4742]], [[- 0.008, - 1.778, 3.991], [1.2071, - 2.8284, 1.9213]]]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round_async(& t2, 4). await ?, & [[[0.7673, - 2.6726, 3.0071], [- 0.7247,
        0.0, 3.4742]], [[- 0.008, - 1.778, 3.991], [1.2071, - 2.8284, 1.9213]]]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert!(diff < 1e-5);
    Ok(())
}
async fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0_async::<f32>()
        .await?;
    assert!(diff < 1e-5);
    Ok(())
}
#[test]
async fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1_async::< f32 > (). await ?, & [1f32, 0.]);
    Ok(())
}
async fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i(&src.i(1..2)?, &cos2, &sin2)?;
    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope_i(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>()
        .await?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}
async fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope(&src.i(1..2)?, &cos2, &sin2)?;
    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>()
        .await?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}
async fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(0..1)?, &cos, &sin)?
    };
    let rope2 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(1..2)?, &cos2, &sin2)?
    };
    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &both_cos, &both_sin)?
    };
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0_async::<f32>()
        .await?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}
async fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0_async::<f32>().await?;
    assert_eq!(diff, 0.);
    Ok(())
}
candle_wasm_tests::test_device!(ropei, ropei_cpu, ropei_gpu, ropei_metal, ropei_wgpu);
candle_wasm_tests::test_device!(rope, rope_cpu, rope_gpu, rope_metal, rope_wgpu);
candle_wasm_tests::test_device!(
    rope_thd, rope_thd_cpu, rope_thd_gpu, rope_thd_metal, rope_thd_wgpu
);
candle_wasm_tests::test_device!(
    softmax, softmax_cpu, softmax_gpu, softmax_metal, softmax_wgpu
);
candle_wasm_tests::test_device!(
    rms_norm, rms_norm_cpu, rms_norm_gpu, rms_norm_metal, rms_norm_wgpu
);
candle_wasm_tests::test_device!(
    rms_norml, rms_norml_cpu, rms_norml_gpu, rms_norml_metal, rms_norml_wgpu
);
candle_wasm_tests::test_device!(layer_norm, ln_cpu, ln_gpu, ln_meta, ln_wgpu);
candle_wasm_tests::test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal, lnl_wgpu);
candle_wasm_tests::test_device!(
    sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal, sigmoid_wgpu
);
}pub mod optim {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::test_utils::{to_vec0_round, to_vec2_round};
use anyhow::Result;
use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, SGD};
#[test]
async fn sgd_optim() -> Result<()> {
    let x = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![x.clone()], 0.1)?;
    let xt = x.as_tensor();
    for _step in 0..100 {
        let loss = ((xt - 4.2)? * (xt - 4.2)?)?;
        sgd.backward_step(&loss)?
    }
    assert_eq!(x.to_scalar_async::< f32 > (). await ?, 4.199999);
    Ok(())
}
#[test]
async fn sgd_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![w.clone(), b.clone()], 0.004)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        sgd.backward_step(&loss)?;
    }
    assert_eq!(w.to_vec2_async::< f32 > (). await ?, & [[2.9983196, 0.99790204]]);
    assert_eq!(b.to_scalar_async::< f32 > (). await ?, - 1.9796902);
    Ok(())
}
#[test]
async fn adamw_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round_async(w.as_tensor(), 4). await ?, & [[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round_async(b.as_tensor(), 4). await ?, 0.7873);
    Ok(())
}
#[test]
async fn adamw_linear_regression_varmap() -> Result<()> {
    use candle_nn::Init::Const;
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(
        &[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]],
        &Device::Cpu,
    )?;
    let sample_ys = gen.forward(&sample_xs)?;
    let mut var_map = candle_nn::VarMap::new();
    let w = var_map.get((1, 2), "w", Const(0.), DType::F32, &Device::Cpu)?;
    let b = var_map.get((), "b", Const(0.), DType::F32, &Device::Cpu)?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(var_map.all_vars(), params)?;
    let lin = Linear::new(w, Some(b));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round_async(lin.weight(), 4). await ?, & [[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round_async(lin.bias().unwrap(), 4). await ?, 0.7873);
    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;
    assert_eq!(to_vec2_round_async(lin.weight(), 4). await ?, & [[0., 0.]]);
    assert_eq!(to_vec0_round_async(lin.bias().unwrap(), 4). await ?, 1.);
    Ok(())
}
}pub mod rnn {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{test_utils::to_vec2_round, DType, Device, Result, Tensor};
use candle_nn::RNN;
#[test]
async fn lstm() -> Result<()> {
    let cpu = &Device::Cpu;
    let w_ih = Tensor::arange(0f32, 24f32, cpu)?.reshape((12, 2))?;
    let w_ih = w_ih.cos()?;
    let w_hh = Tensor::arange(0f32, 36f32, cpu)?.reshape((12, 3))?;
    let w_hh = w_hh.sin()?;
    let b_ih = Tensor::new(
        &[-1f32, 1., -0.5, 2., -1., 1., -0.5, 2., -1., 1., -0.5, 2.],
        cpu,
    )?;
    let b_hh = b_ih.cos()?;
    let tensors: std::collections::HashMap<_, _> = [
        ("weight_ih_l0".to_string(), w_ih),
        ("weight_hh_l0".to_string(), w_hh),
        ("bias_ih_l0".to_string(), b_ih),
        ("bias_hh_l0".to_string(), b_hh),
    ]
        .into_iter()
        .collect();
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, cpu);
    let lstm = candle_nn::lstm(2, 3, Default::default(), vb)?;
    let mut state = lstm.zero_state(1)?;
    for inp in [3f32, 1., 4., 1., 5., 9., 2.] {
        let inp = Tensor::new(&[[inp, inp * 0.5]], cpu)?;
        state = lstm.step(&inp, &state)?;
    }
    let h = state.h();
    let c = state.c();
    assert_eq!(to_vec2_round_async(h, 4). await ?, & [[0.9919, 0.1738, - 0.1451]]);
    assert_eq!(to_vec2_round_async(c, 4). await ?, & [[5.725, 0.4458, - 0.2908]]);
    Ok(())
}
#[test]
async fn gru() -> Result<()> {
    let cpu = &Device::Cpu;
    let w_ih = Tensor::arange(0f32, 18f32, cpu)?.reshape((9, 2))?;
    let w_ih = w_ih.cos()?;
    let w_hh = Tensor::arange(0f32, 27f32, cpu)?.reshape((9, 3))?;
    let w_hh = w_hh.sin()?;
    let b_ih = Tensor::new(&[-1f32, 1., -0.5, 2., -1., 1., -0.5, 2., -1.], cpu)?;
    let b_hh = b_ih.cos()?;
    let tensors: std::collections::HashMap<_, _> = [
        ("weight_ih_l0".to_string(), w_ih),
        ("weight_hh_l0".to_string(), w_hh),
        ("bias_ih_l0".to_string(), b_ih),
        ("bias_hh_l0".to_string(), b_hh),
    ]
        .into_iter()
        .collect();
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, cpu);
    let gru = candle_nn::gru(2, 3, Default::default(), vb)?;
    let mut state = gru.zero_state(1)?;
    for inp in [3f32, 1., 4., 1., 5., 9., 2.] {
        let inp = Tensor::new(&[[inp, inp * 0.5]], cpu)?;
        state = gru.step(&inp, &state)?;
    }
    let h = state.h();
    assert_eq!(to_vec2_round_async(h, 4). await ?, & [[0.0579, 0.8836, - 0.9991]]);
    Ok(())
}
}pub mod sdpa {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;
    #[cfg(not(target_arch = "wasm32"))]
    use tokio::test as test;
    use candle_wasm_tests::{
        to_vec0_round_async, to_vec1_round_async, to_vec2_round_async,
        to_vec3_round_async,
    };
    use candle::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;
    use std::ops::{Div, Mul};
    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let vs: Vec<f32> = (0..elem_count)
            .map(|_| normal.sample_async(rng).await)
            .collect();
        Tensor::from_vec(vs, &shape, dev)
    }
    #[test]
    fn sdpa_full() -> Result<()> {
        const BS: usize = 4;
        const R: usize = 16;
        const L: usize = 16;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            scale as f32,
            1.,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.02, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector() -> Result<()> {
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            scale as f32,
            1.,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.000, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_full_softcapping() -> Result<()> {
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                    &att.to_dtype(DType::F32)?.div(SOFTCAP)?.tanh()?.mul(SOFTCAP)?,
                )?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            scale as f32,
            SOFTCAP as f32,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.002, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector_softcapping() -> Result<()> {
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                    &att.to_dtype(DType::F32)?.div(SOFTCAP)?.tanh()?.mul(SOFTCAP)?,
                )?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            scale as f32,
            SOFTCAP as f32,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0001, "{}", error);
        Ok(())
    }
    #[test]
    fn sdpa_vector_cross() -> Result<()> {
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;
        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            scale as f32,
            1.,
        )?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar_async()
            .await?;
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }
}
}pub mod generation_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
#[test]
async fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    Ok(())
}
#[test]
async fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 0);
    Ok(())
}
#[test]
async fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 2);
    Ok(())
}
#[test]
async fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 3);
    let token = logits_process.sample_async(&logits).await?;
    assert_eq!(token, 2);
    Ok(())
}
#[test]
async fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::GumbelSoftmax {
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[-1.0, 0.0, 0.2, 1.0], &Device::Cpu)?;
    let sm = candle_nn::ops::softmax(&logits, 0)?.to_vec1_async::<f64>().await?;
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample_async(&logits).await?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}
}pub mod nms_tests {
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::Result;
use candle_transformers::object_detection::{
    non_maximum_suppression, soft_non_maximum_suppression, Bbox,
};
#[test]
async fn nms_basic() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 245.0, ymin : 305.0, xmax : 575.0, ymax : 490.0, confidence :
        0.9, data : (), }, Bbox { xmin : 235.0, ymin : 300.0, xmax : 485.0, ymax : 515.0,
        confidence : 0.8, data : (), }, Bbox { xmin : 305.0, ymin : 270.0, xmax : 540.0,
        ymax : 500.0, confidence : 0.6, data : (), },]
    ];
    non_maximum_suppression(&mut bboxes, 0.5);
    let bboxes = bboxes.into_iter().next().unwrap();
    assert_eq!(bboxes.len(), 1);
    assert_eq!(bboxes[0].confidence, 0.9);
    Ok(())
}
#[test]
async fn softnms_basic_functionality() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.5,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.9, data : (), }, Bbox { xmin : 0.2, ymin : 0.2, xmax : 1.2, ymax : 1.2,
        confidence : 0.6, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes[0] [0].confidence == 0.9);
    assert!(bboxes[0] [1].confidence < 0.5);
    assert!(bboxes[0] [2].confidence < 0.6);
    Ok(())
}
#[test]
async fn softnms_confidence_decay() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.8, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes[0] [0].confidence == 0.9);
    assert!(bboxes[0] [1].confidence < 0.8);
    Ok(())
}
#[test]
async fn softnms_confidence_threshold() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.05, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert_eq!(bboxes[0] [0].confidence, 0.9);
    assert_eq!(bboxes[0] [1].confidence, 0.00);
    Ok(())
}
#[test]
async fn softnms_no_overlap() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 2.0, ymin : 2.0, xmax : 3.0, ymax : 3.0, confidence :
        0.8, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert_eq!(bboxes[0] [0].confidence, 0.9);
    assert_eq!(bboxes[0] [1].confidence, 0.8);
    Ok(())
}
#[test]
async fn softnms_no_bbox() -> Result<()> {
    let mut bboxes: Vec<Vec<Bbox<()>>> = vec![];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes.is_empty());
    Ok(())
}
#[test]
async fn softnms_single_bbox() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 1);
    Ok(())
}
#[test]
async fn softnms_equal_confidence_overlap() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.5,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.5, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert!(bboxes[0] [0].confidence == 0.5);
    assert!(bboxes[0] [1].confidence < 0.5);
    Ok(())
}
}