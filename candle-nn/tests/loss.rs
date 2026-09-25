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

print(F.huber_loss(inp, target))
print(F.huber_loss(inp,target,delta=0.88))
*/
#[test]
fn huber_loss() -> Result<()> {
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
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4734);
    let loss = candle_nn::loss::huber(&inp, &target, 0.88)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4483);
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

print(F.smooth_l1_loss(inp, target))
print(F.smooth_l1_loss(inp,target,beta=0.77))
*/
#[test]
fn smoothl1_loss() -> Result<()> {
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
    let loss = candle_nn::loss::smoothl1(&inp, &target, 1.0)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4734);
    let loss = candle_nn::loss::smoothl1(&inp, &target, 0.77)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.5457);
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

target = torch.Tensor([[-1., 1., -1., -1.],
        [-1., 1., -1., -1.],
        [-1., -1., -1., 1.],
        [1., -1., -1., -1.],
        [-1., -1., 1., -1.]])

print(F.hinge_embedding_loss(inp, target,margin=1.0))
print(F.hinge_embedding_loss(inp,target,margin=0.15926))
*/

#[test]
fn hinge_margin_loss() -> Result<()> {
    let cpu = Device::Cpu;
    let inp = [
        [2.3611f32, -0.8813, -0.5006, -0.2178],
        [0.0419, 0.0763, -1.0457, -1.6692],
        [-1.0494, 0.8111, 1.5723, 1.2315],
        [1.3081, 0.6641, 1.1802, -0.2547],
        [0.5292, 0.7636, 0.3692, -0.8318],
    ];

    let target = [
        [-1.0f32, 1., -1., -1.],
        [-1., 1., -1., -1.],
        [-1., -1., -1., 1.],
        [1., -1., -1., -1.],
        [-1., -1., 1., -1.],
    ];

    let inp = Tensor::new(&inp, &cpu)?;
    let target = Tensor::new(&target, &cpu)?;

    let loss = candle_nn::loss::hinge_embedding(&inp, &target, 1.0)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.8432);

    let loss = candle_nn::loss::hinge_embedding(&inp, &target, 0.15926)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.4453);

    Ok(())
}

/* Equivalent python code:
import torch
import torch.nn.functional as F

inp_log_probs = torch.tensor([
    [-0.1000, -3.0000, -2.5000, -2.0000],
    [-1.3863, -1.3863, -1.3863, -1.3863],
    [-0.5000, -1.0000, -1.5000, -2.0000],
    [-0.0100, -5.0000, -4.0000, -4.5000],
    [-0.3000, -0.5000, -3.0000, -3.5000],
])

target_probs = torch.tensor([
    [0.7000, 0.1000, 0.1500, 0.0500],
    [0.2500, 0.2500, 0.2500, 0.2500],
    [0.4000, 0.3000, 0.2000, 0.1000],
    [0.9900, 0.0050, 0.0030, 0.0020],
    [0.0500, 0.0500, 0.4500, 0.4500],
])

target_log_probs = torch.log(target_probs.clamp(min=1e-8))

print(F.kl_div(inp_log_probs, target_probs,log_target=False,reduction='batchmean'))
print(F.kl_div(inp_log_probs, target_log_probs,log_target=True,reduction='batchmean'))
*/
#[test]
fn kl_div_loss() -> Result<()> {
    let cpu = Device::Cpu;
    let inp_log_probs = [
        [-0.1000_f32, -3.0000, -2.5000, -2.0000],
        [-1.3863, -1.3863, -1.3863, -1.3863],
        [-0.5000, -1.0000, -1.5000, -2.0000],
        [-0.0100, -5.0000, -4.0000, -4.5000],
        [-0.3000, -0.5000, -3.0000, -3.5000],
    ];

    let target_probs = [
        [0.7000_f32, 0.1000, 0.1500, 0.0500],
        [0.2500, 0.2500, 0.2500, 0.2500],
        [0.4000, 0.3000, 0.2000, 0.1000],
        [0.9900, 0.0050, 0.0030, 0.0020],
        [0.0500, 0.0500, 0.4500, 0.4500],
    ];

    let inp_log_probs = Tensor::new(&inp_log_probs, &cpu)?;
    let target_probs = Tensor::new(&target_probs, &cpu)?;
    let target_log_probs = target_probs.clamp(1e-8, 1.0)?.log()?;
    let loss = candle_nn::loss::kl_div(&inp_log_probs, &target_probs, false)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.3174);

    let loss = candle_nn::loss::kl_div(&inp_log_probs, &target_log_probs, true)?;
    assert_eq!(to_vec0_round(&loss, 4)?, 0.3174);

    Ok(())
}
