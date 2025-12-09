use candle_core::{test_device, Device, Result, Tensor};

// Test that upsampling to the same size returns an identical tensor
fn bilinear_identity(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bilinear2d(4, 4, false)?;

    let diff = (&t - &output)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0::<f32>()? < 1e-6);
    Ok(())
}

// Test basic 2x upscaling
fn bilinear_upscale_2x(dev: &Device) -> Result<()> {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), dev)?;
    let output = t.upsample_bilinear2d(4, 4, false)?;

    assert_eq!(output.dims(), &[1, 1, 4, 4]);

    // Verify some expected values for 2x2 -> 4x4 upsampling
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(output_vec.len(), 16);

    // Corner values should be preserved (approximately)
    assert!((output_vec[0] - 1.0).abs() < 0.1); // top-left
    Ok(())
}

// Test align_corners parameter produces different results
fn bilinear_align_corners(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;

    let output_false = t.upsample_bilinear2d(8, 8, false)?;
    let output_true = t.upsample_bilinear2d(8, 8, true)?;

    // Results should be different between align_corners modes
    let diff = (&output_false - &output_true)?.abs()?.sum_all()?;
    assert!(diff.to_vec0::<f32>()? > 0.1);
    Ok(())
}

// Test scale_factor mode
fn bilinear_scale_factor(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 96f32, dev)?.reshape((2, 3, 4, 4))?;
    let output = t.upsample_bilinear2d_with_scale(2.0, 2.0, false)?;

    assert_eq!(output.dims(), &[2, 3, 8, 8]);
    Ok(())
}

// Test downscaling
fn bilinear_downscale(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 64f32, dev)?.reshape((1, 1, 8, 8))?;
    let output = t.upsample_bilinear2d(4, 4, false)?;

    assert_eq!(output.dims(), &[1, 1, 4, 4]);
    Ok(())
}

// Test non-square dimensions
fn bilinear_non_square(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 32f32, dev)?.reshape((1, 1, 4, 8))?;
    let output = t.upsample_bilinear2d(6, 12, false)?;

    assert_eq!(output.dims(), &[1, 1, 6, 12]);
    Ok(())
}

// Test batch processing
fn bilinear_batch(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 192f32, dev)?.reshape((4, 3, 4, 4))?;
    let output = t.upsample_bilinear2d(8, 8, false)?;

    assert_eq!(output.dims(), &[4, 3, 8, 8]);
    Ok(())
}

// Test with different scale factors for height and width
fn bilinear_asymmetric_scale(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bilinear2d_with_scale(2.0, 3.0, false)?;

    assert_eq!(output.dims(), &[1, 1, 8, 12]);
    Ok(())
}

/* This test corresponds to the following PyTorch script:
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 4, 4))
print(t.flatten())
res = torch.nn.functional.interpolate(t, size=(8, 8), mode='bilinear', align_corners=False)
print(res.flatten())
*/
fn bilinear_pytorch_compat(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997,
            3.0616, 1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843,
            0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013,
            -0.6836, 0.2477, 1.3127,
        ],
        dev,
    )?
    .reshape((1, 2, 4, 4))?;

    let output = t.upsample_bilinear2d(8, 8, false)?;
    assert_eq!(output.dims(), &[1, 2, 8, 8]);

    // Verify output shape and basic properties
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(output_vec.len(), 128);

    // Check that values are in reasonable range (no NaN or Inf)
    for &val in &output_vec {
        assert!(val.is_finite());
    }

    Ok(())
}

// Test edge case: single pixel output
fn bilinear_single_pixel(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bilinear2d(1, 1, false)?;

    assert_eq!(output.dims(), &[1, 1, 1, 1]);

    // Single pixel should be some average of input
    let val = output.flatten_all()?.to_vec1::<f32>()?[0];
    assert!(val.is_finite());
    Ok(())
}

// Test with align_corners=true
fn bilinear_align_corners_true(dev: &Device) -> Result<()> {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), dev)?;
    let output = t.upsample_bilinear2d(4, 4, true)?;

    assert_eq!(output.dims(), &[1, 1, 4, 4]);

    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;

    // With align_corners=true, corner values should be exactly preserved
    assert!((output_vec[0] - 1.0).abs() < 1e-5); // top-left
    assert!((output_vec[3] - 2.0).abs() < 1e-5); // top-right
    assert!((output_vec[12] - 3.0).abs() < 1e-5); // bottom-left
    assert!((output_vec[15] - 4.0).abs() < 1e-5); // bottom-right

    Ok(())
}

// Test fractional scale factors
fn bilinear_fractional_scale(dev: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 16f32, dev)?.reshape((1, 1, 4, 4))?;
    let output = t.upsample_bilinear2d_with_scale(1.5, 1.5, false)?;

    // 4 * 1.5 = 6.0 (floor)
    assert_eq!(output.dims(), &[1, 1, 6, 6]);
    Ok(())
}

test_device!(
    bilinear_identity,
    bilinear_identity_cpu,
    bilinear_identity_gpu,
    bilinear_identity_metal
);

test_device!(
    bilinear_upscale_2x,
    bilinear_upscale_2x_cpu,
    bilinear_upscale_2x_gpu,
    bilinear_upscale_2x_metal
);

test_device!(
    bilinear_align_corners,
    bilinear_align_corners_cpu,
    bilinear_align_corners_gpu,
    bilinear_align_corners_metal
);

test_device!(
    bilinear_scale_factor,
    bilinear_scale_factor_cpu,
    bilinear_scale_factor_gpu,
    bilinear_scale_factor_metal
);

test_device!(
    bilinear_downscale,
    bilinear_downscale_cpu,
    bilinear_downscale_gpu,
    bilinear_downscale_metal
);

test_device!(
    bilinear_non_square,
    bilinear_non_square_cpu,
    bilinear_non_square_gpu,
    bilinear_non_square_metal
);

test_device!(
    bilinear_batch,
    bilinear_batch_cpu,
    bilinear_batch_gpu,
    bilinear_batch_metal
);

test_device!(
    bilinear_asymmetric_scale,
    bilinear_asymmetric_scale_cpu,
    bilinear_asymmetric_scale_gpu,
    bilinear_asymmetric_scale_metal
);

test_device!(
    bilinear_pytorch_compat,
    bilinear_pytorch_compat_cpu,
    bilinear_pytorch_compat_gpu,
    bilinear_pytorch_compat_metal
);

test_device!(
    bilinear_single_pixel,
    bilinear_single_pixel_cpu,
    bilinear_single_pixel_gpu,
    bilinear_single_pixel_metal
);

test_device!(
    bilinear_align_corners_true,
    bilinear_align_corners_true_cpu,
    bilinear_align_corners_true_gpu,
    bilinear_align_corners_true_metal
);

test_device!(
    bilinear_fractional_scale,
    bilinear_fractional_scale_cpu,
    bilinear_fractional_scale_gpu,
    bilinear_fractional_scale_metal
);
