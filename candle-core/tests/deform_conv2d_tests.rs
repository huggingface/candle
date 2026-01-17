// Allow excessive precision for test data generated from PyTorch
#![allow(clippy::excessive_precision)]

use anyhow::Result;
use candle_core::{test_device, Device, Tensor};

/* ============================================================================
 * Deformable Convolution 2D (deform_conv2d) Data Consistency Tests
 * ============================================================================
 * These tests verify numerical consistency between Candle's deform_conv2d
 * implementation and PyTorch's torchvision.ops.deform_conv2d.
 *
 * Test data generated with:
 *   import torch
 *   from torchvision.ops import deform_conv2d
 *   torch.manual_seed(42)
 *   # batch=1, in_c=2, out_c=2, h=w=4, k=3, stride=1, padding=1
 *   input = torch.randn(1, 2, 4, 4)
 *   weight = torch.randn(2, 2, 3, 3) * 0.1
 *   offset = torch.randn(1, 18, 4, 4) * 0.5
 *   output = deform_conv2d(input, offset, weight, stride=1, padding=1)
 * ============================================================================
 */

// Input data: [1, 2, 4, 4] - torch.manual_seed(42)
#[rustfmt::skip]
const INPUT_DATA: &[f32] = &[
    1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667,
    -0.752136, 1.648723, -0.392479, -1.403607, -0.727881, -0.559430, -0.768839, 0.762445,
    1.642317, -0.159597, -0.497397, 0.439589, -0.758131, 1.078318, 0.800801, 1.680621,
    1.279124, 1.296423, 0.610466, 1.334738, -0.231624, 0.041759, -0.251575, 0.859859,
];

// Weight data: [2, 2, 3, 3] - torch.randn(...) * 0.1
#[rustfmt::skip]
const WEIGHT_DATA: &[f32] = &[
    -0.138467, -0.087124, -0.022337, 0.171736, 0.031888, -0.042452,
    0.030572, -0.077459, -0.155757, 0.099564, -0.087979, -0.060114,
    -0.127415, 0.212279, -0.123465, -0.048791, -0.091382, -0.065814,
    0.007802, 0.052581, 0.034665, -0.019733, -0.105459, 0.127800,
    0.014534, 0.023105, 0.000865, -0.014229, 0.057501, -0.064172,
    -0.220640, -0.075080, 0.281403, 0.035979, -0.008981, 0.045844,
];

// Offset data: [1, 18, 4, 4] - torch.randn(...) * 0.5
#[rustfmt::skip]
const OFFSET_DATA: &[f32] = &[
    0.268094, 0.262311, 0.570601, 0.025822, 0.364055, -0.355321, -0.301034, 0.480224,
    -0.861148, -0.413884, 0.667351, 0.241770, -0.098781, 0.634156, 0.612131, 0.049059,
    0.320378, 0.291624, 0.533463, -0.225077, -0.339376, 0.287158, 0.093875, -0.178812,
    0.132455, 0.636584, -0.000655, -0.151802, -0.493219, 0.061650, 0.174934, 0.308640,
    0.363089, 0.045576, -0.194533, 0.263958, 0.515546, -0.352383, 0.506574, -0.165409,
    0.547505, 0.169945, 0.359984, 0.205704, -0.286660, 0.253432, -0.237605, -0.246013,
    -0.068017, 0.817705, 0.327370, 0.288002, -0.180455, -0.030295, 0.036627, 0.409326,
    -0.187672, 0.516544, -0.343326, 0.318407, 0.108777, -0.023328, -0.716761, -0.283263,
    0.134740, -0.105188, -0.366401, 0.052149, 0.520701, -0.199865, -1.146667, 0.248781,
    -1.240060, -0.208772, -0.597727, 0.406168, -0.153139, -0.165079, -0.490401, 0.097367,
    0.143416, -0.365421, 0.087410, -0.546964, 0.481669, -0.154765, 0.285601, 0.558955,
    -0.773425, 0.378353, 0.387760, 1.013268, 0.490604, -0.320060, -0.245420, 0.104007,
    -0.465974, -0.795483, -0.567988, -0.261299, 0.358268, 0.766734, -0.725489, -0.393068,
    0.511457, -0.277897, 0.352136, 0.354938, -0.766309, -0.362566, 0.233202, 0.333362,
    -0.587667, 0.179029, 0.239384, 0.676850, -0.079666, -0.212472, 0.472115, -0.092467,
    0.092581, 0.534346, 0.653267, 0.229917, 0.130889, -0.379967, -1.023069, -0.764727,
    0.077158, 0.220383, -0.074146, -1.159222, 0.651603, 0.243934, 0.566996, -0.177780,
    0.154714, -0.250155, 0.517502, 0.844824, 0.010637, -0.414635, -0.540429, -0.391927,
    -0.286329, 0.041784, 0.199953, 0.994604, -0.230563, -0.031942, -0.683367, 0.164908,
    0.008812, 0.039113, 0.096579, 0.204837, -0.787705, 1.125419, 0.500615, 0.682119,
    -0.435945, -0.013559, -0.176623, 0.731929, 0.086451, 0.525681, 0.003746, -0.038683,
    0.269866, 0.282753, 0.252896, 0.111227, -0.457162, 0.741984, -0.455453, -0.264550,
    -0.739523, 0.216137, -0.062513, 0.391059, 0.281754, 0.929109, 0.522036, -0.431908,
    0.652930, 0.123296, -0.988795, 0.008948, -0.706438, -0.939528, -0.089917, 0.395193,
    -0.061096, -0.373498, 0.854654, 0.028961, 0.431870, -0.294500, -0.517001, -0.108933,
    0.207296, 0.578282, 0.134527, -0.018315, -0.240380, 0.158151, 0.193286, 0.366847,
    -0.156809, -0.064627, -0.357481, -0.023781, 0.261479, 0.485865, -0.138927, -0.305799,
    -0.015883, 0.050820, 0.671652, 0.356635, 0.173142, -0.270082, 0.428430, -0.336025,
    0.226817, 0.623046, -1.153254, -0.643446, 0.106842, -0.617553, 0.929599, 0.028063,
    -0.382361, -0.027641, 0.602425, -0.491237, 0.151984, 0.466948, -0.986292, -0.705993,
    0.232575, 0.185696, -0.002328, 0.039775, -0.228022, -0.030957, -0.111094, -0.623480,
    0.217572, 0.132944, -0.293549, 0.041344, 0.092882, -0.484898, 0.946577, 0.222345,
    0.283333, -0.354881, -0.243753, 0.025048, 0.163653, 0.064610, 1.425987, -0.371785,
    0.473883, -0.338315, -0.286508, -0.165159, -0.153564, -0.357748, 0.038084, -0.106354,
    -0.160127, -0.422189, -0.275673, 0.994481, 0.423933, -0.347669, 0.152808, 0.145453,
    -0.886748, -0.352321, -0.197326, 0.943406, 0.089339, -0.019253, -0.043442, -0.590140,
];

// Expected output for basic test (no mask, no bias): [1, 2, 4, 4]
#[rustfmt::skip]
const EXPECTED_BASIC: &[f32] = &[
    -0.006874, -0.079786, 0.030247, -0.109367, -0.341144, -0.428116, 0.013985, 0.142769,
    0.057068, 0.157177, -0.140938, 0.024959, -0.195104, -0.123534, -0.354833, 0.082347,
    0.220001, -0.175575, 0.021028, -0.046995, 0.259748, 0.220893, 0.092009, 0.055896,
    0.136941, -0.403170, -0.050020, -0.211272, 0.072600, 0.038178, 0.268262, -0.008896,
];

// Mask data for DCNv2 test: [1, 9, 4, 4] - torch.rand(...)
#[rustfmt::skip]
const MASK_DATA: &[f32] = &[
    0.208883, 0.435096, 0.131408, 0.258788, 0.590549, 0.772269, 0.914185, 0.040947,
    0.834308, 0.147354, 0.687234, 0.923123, 0.507021, 0.954904, 0.073974, 0.309020,
    0.791626, 0.391066, 0.397650, 0.291604, 0.844653, 0.745252, 0.660225, 0.219018,
    0.094125, 0.554080, 0.648139, 0.269144, 0.360101, 0.837684, 0.539830, 0.522559,
    0.376950, 0.047205, 0.029871, 0.260992, 0.245839, 0.655777, 0.354445, 0.304389,
    0.976715, 0.674161, 0.856451, 0.257944, 0.295767, 0.683770, 0.166862, 0.173148,
    0.475850, 0.317120, 0.125171, 0.796579, 0.902081, 0.581112, 0.412943, 0.036864,
    0.317881, 0.627293, 0.735765, 0.436791, 0.302324, 0.778613, 0.101800, 0.816009,
    0.306023, 0.507653, 0.401192, 0.560619, 0.348901, 0.863563, 0.487001, 0.890300,
    0.980740, 0.256405, 0.135245, 0.901151, 0.891807, 0.118226, 0.461348, 0.006937,
    0.090700, 0.596571, 0.633017, 0.605991, 0.363918, 0.961289, 0.571489, 0.204958,
    0.471693, 0.620073, 0.675096, 0.146460, 0.687395, 0.244559, 0.084530, 0.226896,
    0.982205, 0.927429, 0.947742, 0.793506, 0.877725, 0.433075, 0.224886, 0.749828,
    0.240909, 0.162567, 0.340333, 0.604930, 0.757398, 0.305795, 0.205717, 0.567447,
    0.205283, 0.174469, 0.760626, 0.416008, 0.956892, 0.986391, 0.649553, 0.672079,
    0.615142, 0.507830, 0.463634, 0.506872, 0.686712, 0.964885, 0.370420, 0.288642,
    0.378918, 0.258438, 0.585019, 0.873224, 0.890989, 0.729563, 0.132034, 0.231648,
    0.390144, 0.407838, 0.541124, 0.041014, 0.655622, 0.118564, 0.183628, 0.084309,
];

// Expected output with mask (DCNv2): [1, 2, 4, 4]
#[rustfmt::skip]
const EXPECTED_WITH_MASK: &[f32] = &[
    -0.025398, 0.024227, 0.009798, -0.123565, -0.281637, -0.308128, -0.026529, 0.152766,
    0.083494, -0.014510, -0.113616, 0.134171, -0.083259, -0.134015, -0.090958, -0.027379,
    0.034732, -0.077605, 0.021479, -0.008897, 0.141371, 0.212019, 0.046503, 0.068724,
    0.052497, -0.185963, -0.076206, -0.143023, 0.070934, 0.023770, 0.058840, 0.011445,
];

// Expected output with bias [0.5, -0.3]: [1, 2, 4, 4]
#[rustfmt::skip]
const EXPECTED_WITH_BIAS: &[f32] = &[
    0.493126, 0.420214, 0.530247, 0.390633, 0.158856, 0.071884, 0.513985, 0.642769,
    0.557068, 0.657177, 0.359062, 0.524959, 0.304896, 0.376466, 0.145167, 0.582347,
    -0.079999, -0.475575, -0.278972, -0.346995, -0.040252, -0.079107, -0.207991, -0.244104,
    -0.163059, -0.703170, -0.350020, -0.511272, -0.227400, -0.261822, -0.031738, -0.308896,
];

/// Test basic deform_conv2d against PyTorch expected output
fn deform_conv2d_basic(dev: &Device) -> Result<()> {
    let input = Tensor::new(INPUT_DATA, dev)?.reshape((1, 2, 4, 4))?;
    let weight = Tensor::new(WEIGHT_DATA, dev)?.reshape((2, 2, 3, 3))?;
    let offset = Tensor::new(OFFSET_DATA, dev)?.reshape((1, 18, 4, 4))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None,   // mask
        None,   // bias
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 2, 4, 4]);

    // Compare with PyTorch expected output
    let res_vec: Vec<f32> = res.flatten_all()?.to_vec1()?;

    // Calculate error statistics
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut sum_sq_diff: f32 = 0.0;

    for (got, exp) in res_vec.iter().zip(EXPECTED_BASIC.iter()) {
        let diff = (got - exp).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }

    let n = res_vec.len() as f32;
    let mean_diff = sum_diff / n;
    let rmse = (sum_sq_diff / n).sqrt();

    println!("\n=== deform_conv2d_basic ({:?}) ===", dev);
    println!("  Elements: {}", res_vec.len());
    println!("  Max absolute error: {:.2e}", max_diff);
    println!("  Mean absolute error: {:.2e}", mean_diff);
    println!("  RMSE: {:.2e}", rmse);

    assert!(
        max_diff < 1e-4,
        "Max diff {} exceeds tolerance 1e-4",
        max_diff
    );

    Ok(())
}

/// Test deform_conv2d with mask (DCNv2) against PyTorch expected output
fn deform_conv2d_with_mask(dev: &Device) -> Result<()> {
    let input = Tensor::new(INPUT_DATA, dev)?.reshape((1, 2, 4, 4))?;
    let weight = Tensor::new(WEIGHT_DATA, dev)?.reshape((2, 2, 3, 3))?;
    let offset = Tensor::new(OFFSET_DATA, dev)?.reshape((1, 18, 4, 4))?;
    let mask = Tensor::new(MASK_DATA, dev)?.reshape((1, 9, 4, 4))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        Some(&mask),
        None,   // bias
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 2, 4, 4]);

    let res_vec: Vec<f32> = res.flatten_all()?.to_vec1()?;

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut sum_sq_diff: f32 = 0.0;

    for (got, exp) in res_vec.iter().zip(EXPECTED_WITH_MASK.iter()) {
        let diff = (got - exp).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }

    let n = res_vec.len() as f32;
    let mean_diff = sum_diff / n;
    let rmse = (sum_sq_diff / n).sqrt();

    println!("\n=== deform_conv2d_with_mask ({:?}) ===", dev);
    println!("  Elements: {}", res_vec.len());
    println!("  Max absolute error: {:.2e}", max_diff);
    println!("  Mean absolute error: {:.2e}", mean_diff);
    println!("  RMSE: {:.2e}", rmse);

    assert!(
        max_diff < 1e-4,
        "Max diff {} exceeds tolerance 1e-4",
        max_diff
    );

    Ok(())
}

/// Test deform_conv2d with bias against PyTorch expected output
fn deform_conv2d_with_bias(dev: &Device) -> Result<()> {
    let input = Tensor::new(INPUT_DATA, dev)?.reshape((1, 2, 4, 4))?;
    let weight = Tensor::new(WEIGHT_DATA, dev)?.reshape((2, 2, 3, 3))?;
    let offset = Tensor::new(OFFSET_DATA, dev)?.reshape((1, 18, 4, 4))?;
    let bias = Tensor::new(&[0.5f32, -0.3f32], dev)?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None, // mask
        Some(&bias),
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 2, 4, 4]);

    let res_vec: Vec<f32> = res.flatten_all()?.to_vec1()?;

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut sum_sq_diff: f32 = 0.0;

    for (got, exp) in res_vec.iter().zip(EXPECTED_WITH_BIAS.iter()) {
        let diff = (got - exp).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        sum_sq_diff += diff * diff;
    }

    let n = res_vec.len() as f32;
    let mean_diff = sum_diff / n;
    let rmse = (sum_sq_diff / n).sqrt();

    println!("\n=== deform_conv2d_with_bias ({:?}) ===", dev);
    println!("  Elements: {}", res_vec.len());
    println!("  Max absolute error: {:.2e}", max_diff);
    println!("  Mean absolute error: {:.2e}", mean_diff);
    println!("  RMSE: {:.2e}", rmse);

    assert!(
        max_diff < 1e-4,
        "Max diff {} exceeds tolerance 1e-4",
        max_diff
    );

    Ok(())
}

/// Test deform_conv2d with stride=2
fn deform_conv2d_with_stride(dev: &Device) -> Result<()> {
    // Input: [1, 2, 8, 8]
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01) - 0.64).collect();
    let input = Tensor::new(&input_data[..], dev)?.reshape((1, 2, 8, 8))?;

    // Weight: [2, 2, 3, 3]
    let weight_data: Vec<f32> = (0..36).map(|i| (i as f32 * 0.01) - 0.18).collect();
    let weight = Tensor::new(&weight_data[..], dev)?.reshape((2, 2, 3, 3))?;

    // Output size with stride=2, padding=1: (8 + 2*1 - 3) / 2 + 1 = 4
    // Offset: [1, 18, 4, 4]
    let offset_data: Vec<f32> = (0..288).map(|i| ((i as f32 * 0.01) - 1.44) * 0.3).collect();
    let offset = Tensor::new(&offset_data[..], dev)?.reshape((1, 18, 4, 4))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None,   // mask
        None,   // bias
        (2, 2), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 2, 4, 4]);

    // Verify output is not all zeros
    let res_sum = res.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(res_sum > 0.0, "Output should not be all zeros");

    Ok(())
}

/// Test deform_conv2d with dilation=2
fn deform_conv2d_with_dilation(dev: &Device) -> Result<()> {
    // Input: [1, 2, 8, 8]
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01) - 0.64).collect();
    let input = Tensor::new(&input_data[..], dev)?.reshape((1, 2, 8, 8))?;

    // Weight: [2, 2, 3, 3]
    let weight_data: Vec<f32> = (0..36).map(|i| (i as f32 * 0.01) - 0.18).collect();
    let weight = Tensor::new(&weight_data[..], dev)?.reshape((2, 2, 3, 3))?;

    // Output size with dilation=2, padding=2: (8 + 2*2 - (2*(3-1)+1)) / 1 + 1 = 8
    // Offset: [1, 18, 8, 8]
    let offset_data: Vec<f32> = (0..1152)
        .map(|i| ((i as f32 * 0.001) - 0.576) * 0.3)
        .collect();
    let offset = Tensor::new(&offset_data[..], dev)?.reshape((1, 18, 8, 8))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None,   // mask
        None,   // bias
        (1, 1), // stride
        (2, 2), // padding
        (2, 2), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 2, 8, 8]);

    let res_sum = res.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(res_sum > 0.0, "Output should not be all zeros");

    Ok(())
}

/// Test deform_conv2d with multiple offset groups
fn deform_conv2d_offset_groups(dev: &Device) -> Result<()> {
    // Input: [1, 4, 6, 6]
    let input_data: Vec<f32> = (0..144).map(|i| (i as f32 * 0.01) - 0.72).collect();
    let input = Tensor::new(&input_data[..], dev)?.reshape((1, 4, 6, 6))?;

    // Weight: [4, 4, 3, 3]
    let weight_data: Vec<f32> = (0..144).map(|i| (i as f32 * 0.001) - 0.072).collect();
    let weight = Tensor::new(&weight_data[..], dev)?.reshape((4, 4, 3, 3))?;

    // Offset: [1, 2*offset_groups*k*k, 6, 6] = [1, 36, 6, 6]
    let offset_groups = 2;
    let offset_data: Vec<f32> = (0..1296)
        .map(|i| ((i as f32 * 0.001) - 0.648) * 0.3)
        .collect();
    let offset = Tensor::new(&offset_data[..], dev)?.reshape((1, 36, 6, 6))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None,   // mask
        None,   // bias
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        offset_groups,
    )?;

    assert_eq!(res.dims(), [1, 4, 6, 6]);

    Ok(())
}

/// Test deform_conv2d with batch size > 1
fn deform_conv2d_batch(dev: &Device) -> Result<()> {
    // Input: [2, 2, 4, 4]
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01) - 0.32).collect();
    let input = Tensor::new(&input_data[..], dev)?.reshape((2, 2, 4, 4))?;

    // Weight: [2, 2, 3, 3]
    let weight_data: Vec<f32> = (0..36).map(|i| (i as f32 * 0.01) - 0.18).collect();
    let weight = Tensor::new(&weight_data[..], dev)?.reshape((2, 2, 3, 3))?;

    // Offset: [2, 18, 4, 4]
    let offset_data: Vec<f32> = (0..576)
        .map(|i| ((i as f32 * 0.001) - 0.288) * 0.3)
        .collect();
    let offset = Tensor::new(&offset_data[..], dev)?.reshape((2, 18, 4, 4))?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        None,   // mask
        None,   // bias
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [2, 2, 4, 4]);

    Ok(())
}

/// Test full config (mask + bias)
fn deform_conv2d_full(dev: &Device) -> Result<()> {
    // Input: [1, 3, 6, 6]
    let input_data: Vec<f32> = (0..108).map(|i| (i as f32 * 0.01) - 0.54).collect();
    let input = Tensor::new(&input_data[..], dev)?.reshape((1, 3, 6, 6))?;

    // Weight: [4, 3, 3, 3]
    let weight_data: Vec<f32> = (0..108).map(|i| (i as f32 * 0.001) - 0.054).collect();
    let weight = Tensor::new(&weight_data[..], dev)?.reshape((4, 3, 3, 3))?;

    // Offset: [1, 18, 6, 6]
    let offset_data: Vec<f32> = (0..648)
        .map(|i| ((i as f32 * 0.001) - 0.324) * 0.3)
        .collect();
    let offset = Tensor::new(&offset_data[..], dev)?.reshape((1, 18, 6, 6))?;

    // Mask: [1, 9, 6, 6]
    let mask_data: Vec<f32> = (0..324).map(|i| i as f32 / 324.0).collect();
    let mask = Tensor::new(&mask_data[..], dev)?.reshape((1, 9, 6, 6))?;

    // Bias: [4]
    let bias = Tensor::new(&[0.1f32, 0.2, -0.1, -0.2], dev)?;

    let res = input.deform_conv2d(
        &offset,
        &weight,
        Some(&mask),
        Some(&bias),
        (1, 1), // stride
        (1, 1), // padding
        (1, 1), // dilation
        1,      // groups
        1,      // offset_groups
    )?;

    assert_eq!(res.dims(), [1, 4, 6, 6]);

    Ok(())
}

// Register tests using test_device! macro
test_device!(
    deform_conv2d_basic,
    deform_conv2d_basic_cpu,
    deform_conv2d_basic_gpu,
    deform_conv2d_basic_metal
);
test_device!(
    deform_conv2d_with_mask,
    deform_conv2d_with_mask_cpu,
    deform_conv2d_with_mask_gpu,
    deform_conv2d_with_mask_metal
);
test_device!(
    deform_conv2d_with_bias,
    deform_conv2d_with_bias_cpu,
    deform_conv2d_with_bias_gpu,
    deform_conv2d_with_bias_metal
);
test_device!(
    deform_conv2d_with_stride,
    deform_conv2d_with_stride_cpu,
    deform_conv2d_with_stride_gpu,
    deform_conv2d_with_stride_metal
);
test_device!(
    deform_conv2d_with_dilation,
    deform_conv2d_with_dilation_cpu,
    deform_conv2d_with_dilation_gpu,
    deform_conv2d_with_dilation_metal
);
test_device!(
    deform_conv2d_offset_groups,
    deform_conv2d_offset_groups_cpu,
    deform_conv2d_offset_groups_gpu,
    deform_conv2d_offset_groups_metal
);
test_device!(
    deform_conv2d_batch,
    deform_conv2d_batch_cpu,
    deform_conv2d_batch_gpu,
    deform_conv2d_batch_metal
);
test_device!(
    deform_conv2d_full,
    deform_conv2d_full_cpu,
    deform_conv2d_full_gpu,
    deform_conv2d_full_metal
);
