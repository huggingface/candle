#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle::{Device, Tensor, Var};
use candle_nn::{Linear, Module, Optimizer};
use candle_optim::esgd::{MomentumEnhancedSGD, ParamsMESGD};

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
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.1, nesterov=True)
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
fn nesterov_sgd_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsMESGD {
        lr: 0.004,
        weight_decay: 0.0,
        momentum: 0.1,
        dampening: 0.0,
        nesterov: true,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[1.0750, -9.9042]]);
    assert_eq!(to_vec0_round(&b, 4)?, -1.8961);
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
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.1, nesterov=True, weight_decay = 0.1)
    # optimizer.zero_grad()
    for _step in range(100):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimizer.step()
        # print("Optimizer state begin")
        # print(optimizer.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn nesterov_decay_sgd_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsMESGD {
        lr: 0.004,
        weight_decay: 0.1,
        momentum: 0.1,
        dampening: 0.0,
        nesterov: true,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }

    assert_eq!(to_vec2_round(&w, 4)?, &[[0.9921, -10.3803]]);
    assert_eq!(to_vec0_round(&b, 4)?, -1.9331);
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
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.1, nesterov=False, weight_decay = 0.0)
    # optimizer.zero_grad()
    for _step in range(100):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimizer.step()
        # print("Optimizer state begin")
        # print(optimizer.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn momentum_sgd_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsMESGD {
        lr: 0.004,
        weight_decay: 0.0,
        momentum: 0.1,
        dampening: 0.0,
        nesterov: false,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }

    assert_eq!(to_vec2_round(&w, 4)?, &[[2.8870, 0.8589]]);
    assert_eq!(to_vec0_round(&b, 4)?, -0.6341);
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
    optimizer = optim.SGD(m.parameters(), lr=0.004, momentum=0.1, nesterov=False, weight_decay = 0.0, dampening = 0.2)
    # optimizer.zero_grad()
    for _step in range(100):
        optimizer.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimizer.step()
        # print("Optimizer state begin")
        # print(optimizer.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn momentum_sgd_dampened_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsMESGD {
        lr: 0.004,
        weight_decay: 0.0,
        momentum: 0.1,
        dampening: 0.2,
        nesterov: false,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }

    assert_eq!(to_vec2_round(&w, 4)?, &[[2.8746, 0.8434]]);
    assert_eq!(to_vec0_round(&b, 4)?, -0.4838);
    Ok(())
}
