#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_utils::to_vec2_round, DType, Device, Result, Tensor};
use candle_nn::RNN;

/* The following test can be verified against PyTorch using the following snippet.
import torch
from torch import nn
lstm = nn.LSTM(2, 3, 1)
lstm.weight_ih_l0 = torch.nn.Parameter(torch.arange(0., 24.).reshape(12, 2).cos())
lstm.weight_hh_l0 = torch.nn.Parameter(torch.arange(0., 36.).reshape(12, 3).sin())
lstm.bias_ih_l0 = torch.nn.Parameter(torch.tensor([-1., 1., -0.5, 2, -1, 1, -0.5, 2, -1, 1, -0.5, 2]))
lstm.bias_hh_l0 = torch.nn.Parameter(torch.tensor([-1., 1., -0.5, 2, -1, 1, -0.5, 2, -1, 1, -0.5, 2]).cos())
state = torch.zeros((1, 3)), torch.zeros((1, 3))
for inp in [3., 1., 4., 1., 5., 9., 2.]:
    inp = torch.tensor([[inp, inp * 0.5]])
    _out, state = lstm(inp, state)
print(state)
# (tensor([[ 0.9919,  0.1738, -0.1451]], grad_fn=...), tensor([[ 5.7250,  0.4458, -0.2908]], grad_fn=...))
*/
#[test]
fn lstm() -> Result<()> {
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
        state = lstm.step(&inp, &state)?
    }
    let h = state.h();
    let c = state.c();
    assert_eq!(to_vec2_round(h, 4)?, &[[0.9919, 0.1738, -0.1451]]);
    assert_eq!(to_vec2_round(c, 4)?, &[[5.725, 0.4458, -0.2908]]);
    Ok(())
}

/* The following test can be verified against PyTorch using the following snippet.
import torch
from torch import nn
gru = nn.GRU(2, 3, 1)
gru.weight_ih_l0 = torch.nn.Parameter(torch.arange(0., 18.).reshape(9, 2).cos())
gru.weight_hh_l0 = torch.nn.Parameter(torch.arange(0., 27.).reshape(9, 3).sin())
gru.bias_ih_l0 = torch.nn.Parameter(torch.tensor([-1., 1., -0.5, 2, -1, 1, -0.5, 2, -1]))
gru.bias_hh_l0 = torch.nn.Parameter(torch.tensor([-1., 1., -0.5, 2, -1, 1, -0.5, 2, -1]).cos())
state = torch.zeros((1, 3))
for inp in [3., 1., 4., 1., 5., 9., 2.]:
    inp = torch.tensor([[inp, inp * 0.5]])
    _out, state = gru(inp, state)
print(state)
# tensor([[ 0.0579,  0.8836, -0.9991]], grad_fn=<SqueezeBackward1>)
*/
#[test]
fn gru() -> Result<()> {
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
        state = gru.step(&inp, &state)?
    }
    let h = state.h();
    assert_eq!(to_vec2_round(h, 4)?, &[[0.0579, 0.8836, -0.9991]]);
    Ok(())
}
