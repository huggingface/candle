use candle_core::{Device, Result, Tensor};

// ============================================================================
// PyTorch Exact Comparison Tests
// ============================================================================
// These tests compare against exact PyTorch outputs to ensure correctness

/* Test corresponds to PyTorch:
import torch
import torch.nn.functional as F
input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
output = F.interpolate(input, size=(8, 8), mode='bicubic', align_corners=False)
*/
#[test]
fn bicubic_case1_arange4x4_to_8x8_ac_false() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bicubic2d(8, 8, false)?;

    #[rustfmt::skip]
    let expected = vec![
        -0.52734375f32, -0.23046875f32, 0.24609375f32, 0.875_f32, 1.281_25_f32, 1.910_156_3_f32, 2.386_718_8_f32, 2.683_593_8_f32,
        0.66015625f32, 0.95703125f32, 1.433_593_8_f32, 2.062_5_f32, 2.468_75_f32, 3.097_656_3_f32, 3.574_218_8_f32, 3.871_093_8_f32,
        2.566_406_3_f32, 2.863_281_3_f32, 3.339_843_8_f32, 3.968_75_f32, 4.375_f32, 5.003_906_3_f32, 5.480_468_8_f32, 5.777_343_8_f32,
        5.082_031_3_f32, 5.378_906_3_f32, 5.855_468_8_f32, 6.484_375_f32, 6.890_625_f32, 7.519_531_3_f32, 7.996_093_8_f32, 8.292_969_f32,
        6.707_031_3_f32, 7.003_906_3_f32, 7.480_468_8_f32, 8.109_375_f32, 8.515_625_f32, 9.144_531_f32, 9.621_094_f32, 9.917_969_f32,
        9.222_656_f32, 9.519_531_f32, 9.996_094_f32, 10.625_f32, 11.031_25_f32, 11.660_156_f32, 12.136_719_f32, 12.433_594_f32,
        11.128_906_f32, 11.425_781_f32, 11.902_344_f32, 12.531_25_f32, 12.937_5_f32, 13.566_406_f32, 14.042_969_f32, 14.339_844_f32,
        12.316_406_f32, 12.613_281_f32, 13.089_844_f32, 13.718_75_f32, 14.125_f32, 14.753_906_f32, 15.230_469_f32, 15.527_344_f32,
    ];
    let expected = Tensor::new(expected, dev)?.reshape((1, 1, 8, 8))?;

    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;

    assert!(
        max_diff < 1e-4,
        "Max difference {} exceeds threshold 1e-4",
        max_diff
    );
    Ok(())
}

/* Test corresponds to PyTorch:
import torch
import torch.nn.functional as F
input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
output = F.interpolate(input, size=(8, 8), mode='bicubic', align_corners=True)
*/
#[test]
fn bicubic_case2_arange4x4_to_8x8_ac_true() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = input.upsample_bicubic2d(8, 8, true)?;

    #[rustfmt::skip]
    let expected = vec![
        0.00000000f32, 0.34110737f32, 0.800_291_f32, 1.329_445_6_f32, 1.670_554_6_f32, 2.199_71_f32, 2.658_892_2_f32, 3.00000000f32,
        1.364_429_5_f32, 1.705_536_7_f32, 2.164_720_8_f32, 2.693_875_f32, 3.034_984_f32, 3.564_139_4_f32, 4.023_321_6_f32, 4.364_429_5_f32,
        3.201_164_f32, 3.542_271_4_f32, 4.001_456_3_f32, 4.530_610_6_f32, 4.871_72_f32, 5.400_875_f32, 5.860_057_f32, 6.201_165_f32,
        5.317_782_4_f32, 5.658_889_3_f32, 6.118_075_f32, 6.647_228_2_f32, 6.988_337_5_f32, 7.517_493_7_f32, 7.976_674_6_f32, 8.317_782_f32,
        6.682_218_6_f32, 7.023_324_5_f32, 7.482_510_6_f32, 8.011_664_f32, 8.352_774_f32, 8.881_93_f32, 9.341_11_f32, 9.682_219_f32,
        8.798_84_f32, 9.139_946_f32, 9.599_133_f32, 10.128_286_f32, 10.469_394_f32, 10.998_551_f32, 11.457_732_f32, 11.798_840_5_f32,
        10.635_569_f32, 10.976_675_f32, 11.435_863_5_f32, 11.965_015_f32, 12.306_125_f32, 12.835_281_f32, 13.294_459_f32, 13.635_569_f32,
        12.00000000f32, 12.341_106_f32, 12.800_295_f32, 13.329_447_f32, 13.670_555_f32, 14.199_714_f32, 14.658_89_f32, 15.00000000f32,
    ];
    let expected = Tensor::new(expected, dev)?.reshape((1, 1, 8, 8))?;

    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;

    assert!(
        max_diff < 1e-4,
        "Max difference {} exceeds threshold 1e-4",
        max_diff
    );
    Ok(())
}

/* Test corresponds to PyTorch:
import torch
import torch.nn.functional as F
input = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
output = F.interpolate(input, size=(7, 5), mode='bicubic', align_corners=False)
*/
#[test]
fn bicubic_case4_3x3_to_7x5_ac_false() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 9f32, dev)?.reshape((1, 1, 3, 3))?;
    let output = input.upsample_bicubic2d(7, 5, false)?;

    #[rustfmt::skip]
    let expected = vec![
        -0.42398766f32, -0.01198716f32, 0.672_012_3_f32, 1.356_012_5_f32, 1.768_012_8_f32, 0.22761855f32, 0.639_619_1_f32, 1.323_619_1_f32,
        2.007_619_6_f32, 2.419_62_f32, 1.329_657_f32, 1.741_657_f32, 2.425_656_3_f32, 3.109_656_f32, 3.521_656_8_f32, 2.904_000_3_f32,
        3.316_000_2_f32, 4.00000000f32, 4.684_f32, 5.096_000_7_f32, 4.478_343_f32, 4.890_342_7_f32, 5.574_342_7_f32, 6.258_343_f32,
        6.670_343_4_f32, 5.580_384_f32, 5.992_383_5_f32, 6.676_384_f32, 7.360_385_f32, 7.772_384_6_f32, 6.231_989_4_f32, 6.643_989_f32,
        7.327_989_f32, 8.011_99_f32, 8.423_99_f32,
    ];
    let expected = Tensor::new(expected, dev)?.reshape((1, 1, 7, 5))?;

    let diff = (&output - &expected)?.abs()?.flatten_all()?.max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;

    assert!(
        max_diff < 1e-4,
        "Max difference {} exceeds threshold 1e-4",
        max_diff
    );
    Ok(())
}

// ============================================================================
// Dimension and Shape Tests (Consolidated)
// ============================================================================
// These tests verify correct output dimensions for various input configurations

#[test]
fn bicubic_output_dimensions() -> Result<()> {
    let dev = &Device::Cpu;
    // Test 1: Non-square dimensions
    let t1 = Tensor::arange(0f32, 32f32, dev)?.reshape((1, 1, 4, 8))?;
    let out1 = t1.upsample_bicubic2d(6, 12, false)?;
    assert_eq!(out1.dims(), &[1, 1, 6, 12], "Non-square upscale failed");

    // Test 2: Batch processing
    let t2 = Tensor::arange(0f32, 192f32, dev)?.reshape((4, 3, 4, 4))?;
    let out2 = t2.upsample_bicubic2d(8, 8, false)?;
    assert_eq!(out2.dims(), &[4, 3, 8, 8], "Batch processing failed");

    // Test 3: Asymmetric scale factors
    let t3 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out3 = t3.upsample_bicubic2d_with_scale(2.0, 3.0, false)?;
    assert_eq!(out3.dims(), &[1, 1, 8, 12], "Asymmetric scale failed");

    // Test 4: Fractional scale factors
    let t4 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out4 = t4.upsample_bicubic2d_with_scale(1.5, 1.5, false)?;
    assert_eq!(out4.dims(), &[1, 1, 6, 6], "Fractional scale failed");

    // Test 5: Single pixel output
    let t5 = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let out5 = t5.upsample_bicubic2d(1, 1, false)?;
    assert_eq!(out5.dims(), &[1, 1, 1, 1], "Single pixel output failed");
    let val = out5.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(val.is_finite(), "Single pixel value is not finite");

    // Test 6: Large scale factor
    let t6 = Tensor::arange(0f32, 4f32, dev)?.reshape((1, 1, 2, 2))?;
    let out6 = t6.upsample_bicubic2d_with_scale(5.0, 5.0, false)?;
    assert_eq!(out6.dims(), &[1, 1, 10, 10], "Large scale factor failed");

    Ok(())
}

// ============================================================================
// Special Behavior Tests
// ============================================================================

#[test]
fn bicubic_identity() -> Result<()> {
    let dev = &Device::Cpu;
    // Test that upsampling to the same size returns an identical tensor
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bicubic2d(4, 4, false)?;

    let diff = (&t - &output)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0::<f32>()? < 1e-6);
    Ok(())
}

#[test]
fn bicubic_align_corners_difference() -> Result<()> {
    let dev = &Device::Cpu;
    // Test that align_corners parameter produces different results
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;

    let output_false = t.upsample_bicubic2d(8, 8, false)?;
    let output_true = t.upsample_bicubic2d(8, 8, true)?;

    // Results should be different between align_corners modes
    let diff = (&output_false - &output_true)?.abs()?.sum_all()?;
    assert!(diff.to_vec0::<f32>()? > 0.01);
    Ok(())
}

#[test]
fn bicubic_scale_factor() -> Result<()> {
    let dev = &Device::Cpu;
    let input = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output_scale = input.upsample_bicubic2d_with_scale(2.0, 2.0, false)?;
    let output_size = input.upsample_bicubic2d(8, 8, false)?;

    // scale_factor=2.0 should produce identical results to size=(8, 8)
    let diff = (&output_scale - &output_size)?
        .abs()?
        .flatten_all()?
        .max(0)?;
    let max_diff = diff.to_vec0::<f32>()?;

    assert!(
        max_diff < 1e-6,
        "scale_factor and size methods differ by {}",
        max_diff
    );

    Ok(())
}

#[test]
fn bicubic_large_64x64_to_128x128() -> Result<()> {
    let dev = &Device::Cpu;
    use candle_core::DType;

    let input = Tensor::randn(0f32, 1f32, (1, 1, 64, 64), dev)?;
    let output = input.upsample_bicubic2d(128, 128, false)?;

    assert_eq!(output.dims(), &[1, 1, 128, 128]);
    assert_eq!(output.dtype(), DType::F32);

    // Verify all values are finite
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    for &val in &output_vec {
        assert!(
            val.is_finite(),
            "Large tensor output contains non-finite value"
        );
    }

    Ok(())
}
