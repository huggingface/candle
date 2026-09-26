#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, SGD};

#[test]
fn sgd_optim() -> Result<()> {
    let x = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![x.clone()], 0.1)?;
    let xt = x.as_tensor();
    for _step in 0..100 {
        let loss = ((xt - 4.2)? * (xt - 4.2)?)?;
        sgd.backward_step(&loss)?
    }
    assert_eq!(x.to_scalar::<f32>()?, 4.199999);
    Ok(())
}

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.)
    for _step in range(1000):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimizer.step()
    print(m.weight)
    print(m.bias)
*/
#[test]
fn sgd_linear_regression() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut sgd = SGD::new(vec![w.clone(), b.clone()], 0.004)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        sgd.backward_step(&loss)?;
    }
    assert_eq!(w.to_vec2::<f32>()?, &[[2.9983196, 0.99790204]]);
    assert_eq!(b.to_scalar::<f32>()?, -1.9796902);
    Ok(())
}

/* The following test returns the same values as the PyTorch code below.
import torch
from torch import optim

w_gen = torch.tensor([[3., 1.]])
b_gen = torch.tensor([-2.])

sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

m = torch.nn.Linear(2, 1)
with torch.no_grad():
    m.weight.zero_()
    m.bias.zero_()
optimizer = optim.AdamW(m.parameters(), lr=0.1)
for _step in range(100):
    optimizer.zero_grad()
    ys = m(sample_xs)
    loss = ((ys - sample_ys)**2).sum()
    loss.backward()
    optimizer.step()
print(m.weight)
print(m.bias)
*/
#[test]
fn adamw_linear_regression() -> Result<()> {
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    // Now use backprop to run a linear regression between samples and get the coefficients back.
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
    assert_eq!(to_vec2_round(w.as_tensor(), 4)?, &[[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round(b.as_tensor(), 4)?, 0.7873);
    Ok(())
}

#[test]
fn adamw_linear_regression_varmap() -> Result<()> {
    use candle_nn::Init::Const;

    // Similar as the previous test but using a VarMap.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
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
    assert_eq!(to_vec2_round(lin.weight(), 4)?, &[[2.7257, 0.7097]]);
    assert_eq!(to_vec0_round(lin.bias().unwrap(), 4)?, 0.7873);

    var_map.set([("w", Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)].into_iter())?;
    var_map.set([("b", Tensor::ones((), DType::F32, &Device::Cpu)?)].into_iter())?;

    assert_eq!(to_vec2_round(lin.weight(), 4)?, &[[0., 0.]]);
    assert_eq!(to_vec0_round(lin.bias().unwrap(), 4)?, 1.);
    Ok(())
}

/// Six AdamW steps on `loss = sum(w^2)`, optionally checkpointing the optimizer
/// state after `before` steps and restoring it into a fresh `AdamW` for the rest,
/// the way a new process would.
fn adamw_staged(before: usize, after: usize, restore_moments: bool) -> Result<Vec<f32>> {
    let dev = Device::Cpu;
    let w = Var::from_slice(&[1.0f32, 2.0, 3.0], (3,), &dev)?;

    let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW::default())?;
    for _ in 0..before {
        let loss = w.as_tensor().sqr()?.sum_all()?;
        opt.step(&loss.backward()?)?;
    }

    if after > 0 {
        let step_t = opt.step_t();
        let saved: Vec<(Tensor, Tensor)> = opt
            .moments()
            .map(|(_, m, v)| (m.as_tensor().clone(), v.as_tensor().clone()))
            .collect();

        let mut opt = AdamW::new(vec![w.clone()], ParamsAdamW::default())?;
        opt.set_step_t(step_t);
        if restore_moments {
            for ((_, m, v), (saved_m, saved_v)) in opt.moments().zip(saved.iter()) {
                m.set(saved_m)?;
                v.set(saved_v)?;
            }
        }
        for _ in 0..after {
            let loss = w.as_tensor().sqr()?.sum_all()?;
            opt.step(&loss.backward()?)?;
        }
    }
    Ok(w.as_tensor().to_vec1::<f32>()?)
}

/// A run checkpointed and resumed through `step_t` / `moments` must land exactly
/// where an uninterrupted one does.
#[test]
fn adamw_resume_matches_uninterrupted_run() -> Result<()> {
    assert_eq!(adamw_staged(6, 0, true)?, adamw_staged(3, 3, true)?);
    Ok(())
}

/// Control for the test above: restoring the step counter but *not* the moments
/// must change the trajectory. Without this, a resume that silently dropped the
/// moments would still pass.
#[test]
fn adamw_losing_the_moments_is_visible() -> Result<()> {
    assert_ne!(adamw_staged(6, 0, true)?, adamw_staged(3, 3, false)?);
    Ok(())
}
