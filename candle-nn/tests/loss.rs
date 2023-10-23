#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::to_vec0_round;
use candle::{Device, Result, Tensor};

/* Equivalent python code:
import torch
import torch.nn.functional as F
input = torch.tensor([
    [ 1.1050,  0.3013, -1.5394, -2.1528, -0.8634],
    [ 1.0730, -0.9419, -0.1670, -0.6582,  0.5061],
    [ 0.8318,  1.1154, -0.3610,  0.5351,  1.0830]])

target = torch.tensor([1, 0, 4])
print(F.nll_loss(F.log_softmax(input, dim=1), target))
print(F.cross_entropy(input, target))
*/
#[test]
fn nll_and_cross_entropy() -> Result<()> {
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
    assert_eq!(to_vec0_round(&loss, 4)?, 1.1312);
    let loss = candle_nn::loss::cross_entropy(&input, &target)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 1.1312);
    Ok(())
}

#[test]
fn binary_cross_entropy_with_logit() -> Result<()> {
    let cpu = Device::Cpu;

    let inp = [[ 2.3611f64, -0.8813, -0.5006, -0.2178],
                                [ 0.0419,  0.0763, -1.0457, -1.6692],
                                [-1.0494,  0.8111,  1.5723,  1.2315],
                                [ 1.3081,  0.6641,  1.1802, -0.2547],
                                [ 0.5292,  0.7636,  0.3692, -0.8318],
                                [ 0.5100,  0.9849, -1.2905,  0.2821],
                                [ 1.4662,  0.4550,  0.9875,  0.3143],
                                [-1.2121,  0.1262,  0.0598, -1.6363],
                                [ 0.3214, -0.8689,  0.0689, -2.5094],
                                [ 1.1320, -0.6824,  0.1657, -0.0687]];

    let target = [[0.0f64, 1., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 0., 1.],
                                [1., 0., 0., 0.],
                                [0., 0., 1., 0.],
                                [1., 0., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 1., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.]];

    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;

    let loss = candle_nn::loss::binary_cross_entropy_with_logit(&inp, &target)?;

    assert_eq!(to_vec0_round(&loss, 4)?, 0.7739);


}
