#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, DType, Device, Tensor};
use candle_nn::{batch_norm, BatchNorm, BatchNormConfig, VarBuilder, VarMap};

/* The test below has been generated using the following PyTorch code:
import torch
torch.manual_seed(19551105)
m = torch.nn.BatchNorm2d(5, affine=False)
input = torch.randn(2, 5, 3, 4)
output = m(input)
print(input.flatten())
print(output.flatten())
print(m.running_mean)
print(m.running_var)
*/
#[test]
fn batch_norm_test() -> Result<()> {
    let running_mean = Tensor::zeros(5, DType::F32, &Device::Cpu)?;
    let running_var = Tensor::ones(5, DType::F32, &Device::Cpu)?;
    let bn = BatchNorm::new_no_bias(5, running_mean.clone(), running_var.clone(), 1e-8)?;
    let input: [f32; 120] = [
        -0.7493, -1.0410, 1.6977, -0.6579, 1.7982, -0.0087, 0.2812, -0.1190, 0.2908, -0.5975,
        -0.0278, -0.2138, -1.3130, -1.6048, -2.2028, 0.9452, 0.4002, 0.0831, 1.0004, 0.1860,
        0.5004, 0.5539, 0.9991, -0.2540, -0.0703, -0.3752, -0.1096, -0.2374, 1.0258, -2.2208,
        -0.0257, 0.6073, -1.1627, -0.0964, -1.9718, 1.6577, 0.1931, -0.3692, -0.8011, 0.9059,
        0.4797, 0.6521, -0.0165, -0.6683, -0.4148, 2.0649, -0.8276, 1.7947, -0.2061, 0.5812,
        -1.3598, 1.6192, 1.0466, -0.4423, 0.4202, 0.1749, 0.6969, 0.2616, -0.0369, -1.4951,
        -0.0814, -0.1877, 0.0267, 0.6150, 0.2402, -1.1440, -2.0068, 0.6032, -2.6639, 0.8260,
        0.1085, -0.1693, 1.2805, 0.7654, -0.4930, 0.3770, 1.1309, 0.2303, 0.2949, -0.2634, -0.5225,
        0.4269, 0.6341, 1.5736, 0.9827, -1.2499, 0.3509, -1.6243, -0.8123, 0.7634, -0.3047, 0.0143,
        -0.4032, 0.0537, 0.7022, 0.8405, -1.2221, -1.6847, -0.0714, -0.1608, 0.5579, -1.5858,
        0.4617, -0.6480, 0.1332, 0.0419, -0.9784, 0.4173, 1.2313, -1.9046, -0.1656, 0.1259, 0.0763,
        1.4252, -0.9115, -0.1093, -0.3100, -0.6734, -1.4357, 0.9205,
    ];
    let input = Tensor::new(&input, &Device::Cpu)?.reshape((2, 5, 3, 4))?;
    let output = bn.forward_train(&input)?;
    assert_eq!(output.dims(), &[2, 5, 3, 4]);
    let output = output.flatten_all()?;
    assert_eq!(
        test_utils::to_vec1_round(&output, 4)?,
        &[
            -0.6391, -0.9414, 1.8965, -0.5444, 2.0007, 0.1283, 0.4287, 0.014, 0.4387, -0.4818,
            0.1085, -0.0842, -1.6809, -2.0057, -2.6714, 0.8328, 0.2262, -0.1268, 0.8943, -0.0123,
            0.3377, 0.3973, 0.8928, -0.5021, 0.0861, -0.2324, 0.0451, -0.0884, 1.2311, -2.1603,
            0.1327, 0.7939, -1.055, 0.0589, -1.9002, 1.8912, 0.2918, -0.3253, -0.7993, 1.0741,
            0.6063, 0.7955, 0.0617, -0.6536, -0.3754, 2.3461, -0.8284, 2.0495, -0.201, 0.6476,
            -1.4446, 1.7665, 1.1493, -0.4556, 0.4741, 0.2097, 0.7723, 0.3031, -0.0186, -1.5905,
            0.053, -0.0572, 0.165, 0.7746, 0.3862, -1.0481, -1.9422, 0.7624, -2.6231, 0.9933,
            0.2498, -0.0381, 1.2061, 0.6327, -0.7681, 0.2004, 1.0396, 0.037, 0.109, -0.5125,
            -0.8009, 0.2559, 0.4865, 1.5324, 1.1861, -1.1461, 0.5261, -1.5372, -0.689, 0.957,
            -0.1587, 0.1745, -0.2616, 0.2156, 0.8931, 1.0375, -1.2614, -1.7691, 0.0015, -0.0966,
            0.6921, -1.6605, 0.5866, -0.6313, 0.226, 0.1258, -0.9939, 0.5378, 1.3484, -2.0319,
            -0.1574, 0.1568, 0.1034, 1.5574, -0.9614, -0.0967, -0.313, -0.7047, -1.5264, 1.0134
        ]
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
    assert_eq!(output2.dims(), &[2, 5, 3, 4]);
    let output2 = output2.flatten_all()?;
    let diff2 = ((output2 - (output * 0.5)?)? + 1.5)?.sqr()?;
    let sum_diff2 = diff2.sum_keepdim(0)?;
    assert_eq!(test_utils::to_vec1_round(&sum_diff2, 4)?, &[0f32]);

    assert_eq!(
        test_utils::to_vec1_round(bn.running_mean(), 4)?,
        &[-0.0133, 0.0197, -0.0153, -0.0073, -0.0020]
    );
    assert_eq!(
        test_utils::to_vec1_round(bn.running_var(), 4)?,
        &[0.9972, 0.9842, 0.9956, 0.9866, 0.9898]
    );
    Ok(())
}

// This test makes sure that we can train a batch norm layer using a VarMap.
#[test]
fn train_batch_norm() -> Result<()> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    let bn = batch_norm(1, BatchNormConfig::default(), vb)?;
    // Get a copy of the original mean to ensure it is being updated.
    let original_mean = bn.running_mean().detach().copy()?;
    let var_map_mean = {
        vm.data()
            .lock()
            .unwrap()
            .get("running_mean")
            .unwrap()
            .clone()
    };
    // Ensure the var map mean is the same as the running mean.
    assert_eq!(
        test_utils::to_vec1_round(bn.running_mean(), 4)?,
        test_utils::to_vec1_round(var_map_mean.as_tensor(), 4)?,
    );
    // Train with a something guaranteed to be different from the running mean.
    let mean_plus_one = {
        let one = original_mean.ones_like()?;
        original_mean.add(&one)?.reshape((1, 1))?
    };

    bn.forward_train(&mean_plus_one)?;
    // Assert that the running mean has been updated.
    assert_ne!(
        test_utils::to_vec1_round(bn.running_mean(), 4)?,
        test_utils::to_vec1_round(&original_mean, 4)?,
    );

    // Assert that the var map mean has been updated.
    assert_eq!(
        test_utils::to_vec1_round(bn.running_mean(), 4)?,
        test_utils::to_vec1_round(var_map_mean.as_tensor(), 4)?,
    );
    Ok(())
}
