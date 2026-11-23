//! Layout parity and coordinate-wise central differences, independent of candle-nn.
use candle_core::{Device, Result, Tensor, Var};

fn data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 17 % 31) as f64 - 15.) / 19.).collect()
}
fn flat(t: &Tensor) -> Result<Vec<f64>> {
    t.flatten_all()?.to_vec1()
}
fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= tol * (1. + a.abs().max(b.abs())),
            "coordinate {i}: {a} != {b}"
        );
    }
}

#[test]
fn strided_kernels_match_materialized_kernels() -> Result<()> {
    let dev = &Device::Cpu;
    for dims in [vec![2, 3, 3], vec![2, 3, 2, 3]] {
        let w = Tensor::from_vec(data(dims.iter().product()), dims, dev)?.transpose(0, 1)?;
        assert!(!w.is_contiguous());
        let dense = w.contiguous()?;
        if w.rank() == 3 {
            let x = Tensor::from_vec(data(2 * 2 * 7), (2, 2, 7), dev)?;
            close(
                &flat(&x.conv1d(&w, 1, 1, 1, 1)?)?,
                &flat(&x.conv1d(&dense, 1, 1, 1, 1)?)?,
                1e-12,
            );
            let x = Tensor::from_vec(data(2 * 3 * 5), (2, 3, 5), dev)?;
            close(
                &flat(&x.conv_transpose1d(&w, 1, 0, 1, 1, 1)?)?,
                &flat(&x.conv_transpose1d(&dense, 1, 0, 1, 1, 1)?)?,
                1e-12,
            );
        } else {
            let x = Tensor::from_vec(data(2 * 2 * 5 * 6), (2, 2, 5, 6), dev)?;
            close(
                &flat(&x.conv2d(&w, 1, 1, 1, 1)?)?,
                &flat(&x.conv2d(&dense, 1, 1, 1, 1)?)?,
                1e-12,
            );
            let x = Tensor::from_vec(data(2 * 3 * 4 * 5), (2, 3, 4, 5), dev)?;
            close(
                &flat(&x.conv_transpose2d(&w, 1, 0, 1, 1)?)?,
                &flat(&x.conv_transpose2d(&dense, 1, 0, 1, 1)?)?,
                1e-12,
            );
        }
    }
    Ok(())
}

fn gradients(
    xdims: &[usize],
    wdims: &[usize],
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<()> {
    let dev = &Device::Cpu;
    let x = data(xdims.iter().product());
    let w = data(wdims.iter().product());
    let conv = |x: &Tensor, w: &Tensor| {
        if x.rank() == 3 {
            x.conv1d(w, padding, stride, dilation, 1)
        } else {
            x.conv2d(w, padding, stride, dilation, 1)
        }
    };
    let vx = Var::from_tensor(&Tensor::from_vec(x.clone(), xdims, dev)?)?;
    let vw = Var::from_tensor(&Tensor::from_vec(w.clone(), wdims, dev)?)?;
    let loss = |x: &[f64], w: &[f64]| -> Result<f64> {
        conv(
            &Tensor::from_vec(x.to_vec(), xdims, dev)?,
            &Tensor::from_vec(w.to_vec(), wdims, dev)?,
        )?
        .sqr()?
        .sum_all()?
        .to_scalar()
    };
    let grads = conv(&vx, &vw)?.sqr()?.sum_all()?.backward()?;
    for (is_x, analytic) in [
        (true, flat(grads.get(&vx).expect("input gradient"))?),
        (false, flat(grads.get(&vw).expect("weight gradient"))?),
    ] {
        assert!(analytic.iter().any(|v| v.abs() > 1e-8));
        let mut xp = x.clone();
        let mut wp = w.clone();
        let mut numeric = Vec::new();
        for i in 0..analytic.len() {
            let base = if is_x { x[i] } else { w[i] };
            if is_x {
                xp[i] = base + 1e-5;
            } else {
                wp[i] = base + 1e-5;
            }
            let plus = loss(&xp, &wp)?;
            if is_x {
                xp[i] = base - 1e-5;
            } else {
                wp[i] = base - 1e-5;
            }
            let minus = loss(&xp, &wp)?;
            if is_x {
                xp[i] = base;
            } else {
                wp[i] = base;
            }
            numeric.push((plus - minus) / 2e-5);
        }
        close(&analytic, &numeric, 2e-7);
    }
    Ok(())
}
#[test]
fn conv1d_input_and_weight_gradients() -> Result<()> {
    gradients(&[2, 2, 7], &[3, 2, 3], 1, 1, 1)?;
    gradients(&[2, 2, 7], &[3, 2, 3], 1, 2, 1)?;
    // An even input length requires output padding in the input-gradient op.
    gradients(&[2, 2, 8], &[3, 2, 3], 1, 2, 1)?;
    gradients(&[2, 2, 9], &[3, 2, 3], 2, 1, 2)
}
#[test]
fn conv2d_input_and_weight_gradients() -> Result<()> {
    gradients(&[2, 2, 4, 5], &[3, 2, 2, 3], 1, 1, 1)
}
#[test]
fn short_padded_conv1d_backward_does_not_underflow() -> Result<()> {
    gradients(&[1, 2, 2], &[3, 2, 3], 1, 1, 1)
}
