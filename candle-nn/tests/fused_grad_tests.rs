//! The fused `softmax_last_dim` and `layer_norm` now carry a hand-written backward.
//!
//! These tests hold that backward to the gradients the AUTOGRAD produces when the same
//! function is built from primitive ops — the composed forms candle records node by node. A
//! silent-wrong-gradient here is the worst failure mode this change can have (the loss still
//! goes down, plateauing at the wrong place), so parity is asserted element-wise on every
//! gradient the op produces, not spot-checked.

use candle::test_utils::to_vec1_round;
use candle::{DType, Device, Result, Tensor, Var, D};

/// Deterministic, sign-varied, non-trivial values — no RNG dependency in the test.
fn seeded(shape: (usize, usize, usize), scale: f64, dev: &Device) -> Result<Tensor> {
    let (a, b, c) = shape;
    let n = a * b * c;
    let xs: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37 + 0.1).sin() * scale as f32)
        .collect();
    Tensor::from_vec(xs, shape, dev)
}

/// The composed softmax, built from primitives so autograd differentiates it node by node.
fn softmax_composed(xs: &Tensor) -> Result<Tensor> {
    let max = xs.max_keepdim(D::Minus1)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(D::Minus1)?;
    num.broadcast_div(&den)
}

/// The composed layer norm, built from primitives — the same formula `candle_nn`'s own
/// non-fused fall-through uses.
fn layer_norm_composed(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden = xs.dim(D::Minus1)? as f64;
    let mu = (xs.sum_keepdim(D::Minus1)? / hidden)?;
    let xc = xs.broadcast_sub(&mu)?;
    let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let x_hat = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    x_hat.broadcast_mul(alpha)?.broadcast_add(beta)
}

/// A fixed, sign-varied cotangent so the tested gradient is not the all-ones special case.
fn cotangent(y: &Tensor) -> Result<Tensor> {
    let n: usize = y.dims().iter().product();
    let c: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.73 - 1.0).cos()).collect();
    Tensor::from_vec(c, y.shape(), y.device())
}

#[test]
fn fused_softmax_backward_matches_autograd() -> Result<()> {
    let dev = Device::Cpu;
    let x0 = seeded((2, 3, 8), 3.0, &dev)?;

    let grad_of = |fused: bool| -> Result<Vec<f32>> {
        let v = Var::from_tensor(&x0)?;
        let y = if fused {
            candle_nn::ops::softmax_last_dim(v.as_tensor())?
        } else {
            softmax_composed(v.as_tensor())?
        };
        let loss = (y * cotangent(&x0)?)?.sum_all()?;
        let grads = loss.backward()?;
        let g = grads.get(&v).expect("softmax dropped its input gradient");
        g.flatten_all()?.to_vec1::<f32>()
    };

    let (fused, composed) = (grad_of(true)?, grad_of(false)?);
    assert_eq!(fused.len(), composed.len());
    for (i, (a, b)) in fused.iter().zip(composed.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6,
            "softmax dx[{i}] diverges: fused {a} vs autograd {b}"
        );
    }
    Ok(())
}

#[test]
fn fused_layer_norm_backward_matches_autograd() -> Result<()> {
    let dev = Device::Cpu;
    let hidden = 8;
    let x0 = seeded((2, 3, hidden), 2.0, &dev)?;
    let gamma0: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.11).collect();
    let beta0: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.07 - 0.2).collect();
    let eps = 1e-5_f32;

    let grads_of = |fused: bool| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let v = Var::from_tensor(&x0)?;
        let g = Var::from_tensor(&Tensor::from_vec(gamma0.clone(), hidden, &dev)?)?;
        let b = Var::from_tensor(&Tensor::from_vec(beta0.clone(), hidden, &dev)?)?;
        let y = if fused {
            candle_nn::ops::layer_norm(v.as_tensor(), g.as_tensor(), b.as_tensor(), eps)?
        } else {
            layer_norm_composed(v.as_tensor(), g.as_tensor(), b.as_tensor(), eps as f64)?
        };
        let loss = (y * cotangent(&x0)?)?.sum_all()?;
        let grads = loss.backward()?;
        let dx = grads
            .get(&v)
            .expect("layer_norm dropped dx")
            .flatten_all()?
            .to_vec1()?;
        let dg = grads
            .get(&g)
            .expect("layer_norm dropped dgamma")
            .to_vec1()?;
        let db = grads.get(&b).expect("layer_norm dropped dbeta").to_vec1()?;
        Ok((dx, dg, db))
    };

    let (fx, fg, fb) = grads_of(true)?;
    let (cx, cg, cb) = grads_of(false)?;
    for (name, f, c) in [("dx", &fx, &cx), ("dgamma", &fg, &cg), ("dbeta", &fb, &cb)] {
        assert_eq!(f.len(), c.len());
        for (i, (a, b)) in f.iter().zip(c.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5,
                "layer_norm {name}[{i}] diverges: fused {a} vs autograd {b}"
            );
        }
    }
    Ok(())
}

#[test]
fn fused_backward_exists_at_all() -> Result<()> {
    // The regression this whole change exists to prevent: apply_op*_no_bwd silently DROPS the
    // gradient — backward() returns Ok and the input never appears in the store. This is the
    // cheapest possible canary, and it is the probe downstream dispatch logic relies on.
    let dev = Device::Cpu;
    let v = Var::from_tensor(&Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4]], &dev)?)?;
    let y = candle_nn::ops::softmax_last_dim(v.as_tensor())?;
    let grads = y.sum_all()?.backward()?;
    assert!(
        grads.get(&v).is_some(),
        "fused softmax records no backward op"
    );

    let g = Var::from_tensor(&Tensor::ones(4, DType::F32, &dev)?)?;
    let b = Var::from_tensor(&Tensor::zeros(4, DType::F32, &dev)?)?;
    let y = candle_nn::ops::layer_norm(v.as_tensor(), g.as_tensor(), b.as_tensor(), 1e-5)?;
    let grads = y.sum_all()?.backward()?;
    assert!(
        grads.get(&v).is_some(),
        "fused layer_norm records no backward op"
    );
    assert!(
        grads.get(&g).is_some(),
        "fused layer_norm records no dgamma"
    );
    assert!(grads.get(&b).is_some(), "fused layer_norm records no dbeta");

    // Keep the linter honest about the unused helper import.
    let _ = to_vec1_round(&Tensor::zeros(1, DType::F32, &dev)?, 4)?;
    Ok(())
}
