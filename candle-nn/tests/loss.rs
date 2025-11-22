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

/* Equivalent python code:
import torch
import torch.nn.functional as F

inp = torch.Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],
        [ 0.0419,  0.0763, -1.0457, -1.6692],
        [-1.0494,  0.8111,  1.5723,  1.2315],
        [ 1.3081,  0.6641,  1.1802, -0.2547],
        [ 0.5292,  0.7636,  0.3692, -0.8318]])

target = torch.Tensor([[0., 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.]])

print(F.binary_cross_entropy_with_logits(inp, target))
*/
#[test]
fn binary_cross_entropy_with_logit() -> Result<()> {
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

    assert_eq!(to_vec0_round(&loss, 4)?, 0.8224);
    Ok(())
}

#[test]
fn cross_entropy_with_ignore_matches_subset() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(
        &[
            [1.1050f32, 0.3013, -1.5394],
            [1.0730, -0.9419, -0.1670],
            [0.8318, 1.1154, -0.3610],
        ],
        &cpu,
    )?;
    let target = Tensor::new(&[1u32, 0, 999], &cpu)?;

    let loss_ignore = candle_nn::loss::cross_entropy_with_ignore(&input, &target, 999)?;

    let input_first2 = input.narrow(0, 0, 2)?;
    let target_first2 = target.narrow(0, 0, 2)?;
    let loss_subset = candle_nn::loss::cross_entropy(&input_first2, &target_first2)?;

    let a = to_vec0_round(&loss_ignore, 4)?;
    let b = to_vec0_round(&loss_subset, 4)?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn cross_entropy_with_ignore_all_ignored_is_zero() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(&[[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]], &cpu)?;
    let target = Tensor::new(&[999u32, 999u32], &cpu)?;

    let loss = candle_nn::loss::cross_entropy_with_ignore(&input, &target, 999)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.0);
    Ok(())
}

#[test]
fn cross_entropy_with_ignore_none_ignored_equals_regular() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(
        &[
            [0.5f32, -0.2, 0.1, 1.0],
            [1.2, 0.3, -0.7, 0.0],
            [-0.1, 0.4, 0.6, -0.3],
        ],
        &cpu,
    )?;
    let target = Tensor::new(&[1u32, 0, 2], &cpu)?;

    let loss_regular = candle_nn::loss::cross_entropy(&input, &target)?;
    let loss_ignore = candle_nn::loss::cross_entropy_with_ignore(&input, &target, 999)?;

    let a = to_vec0_round(&loss_regular, 4)?;
    let b = to_vec0_round(&loss_ignore, 4)?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn nll_with_ignore_matches_cross_entropy_with_ignore() -> Result<()> {
    let cpu = Device::Cpu;
    let input = Tensor::new(
        &[
            [1.1050f32, 0.3013, -1.5394, -2.1528],
            [1.0730, -0.9419, -0.1670, -0.6582],
            [0.8318, 1.1154, -0.3610, 0.5351],
            [0.1, -0.2, 0.3, 0.4],
        ],
        &cpu,
    )?;
    let target = Tensor::new(&[1u32, 0, 999, 2], &cpu)?;

    let log_softmax = candle_nn::ops::log_softmax(&input, 1)?;
    let loss_nll = candle_nn::loss::nll_with_ignore(&log_softmax, &target, 999)?;
    let loss_ce = candle_nn::loss::cross_entropy_with_ignore(&input, &target, 999)?;

    let a = to_vec0_round(&loss_nll, 4)?;
    let b = to_vec0_round(&loss_ce, 4)?;
    assert_eq!(a, b);
    Ok(())
}
