use candle_core::{test_device, Device, Result, Tensor, Var};

fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % 23) as f32 - 11.) / 20.).collect()
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a as f64 - e).abs() <= 2e-4 + 2e-4 * e.abs(),
            "element {i}: actual {a}, expected {e}"
        );
    }
}

// Independent scalar cross-correlation, with no Candle convolution operations.
fn reference(
    x: &[f64],
    w: &[f64],
    h: usize,
    width: usize,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    is_2d: bool,
) -> Vec<f64> {
    let ph = if is_2d { padding } else { 0 };
    let oh = (h + 2 * ph - dilation * (kh - 1) - 1) / stride + 1;
    let ow = (width + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    let mut y = vec![0.; 2 * 3 * oh * ow];
    for b in 0..2 {
        for o in 0..3 {
            for a in 0..oh {
                for z in 0..ow {
                    for c in 0..2 {
                        for r in 0..kh {
                            for s in 0..kw {
                                let ih = (a * stride + r * dilation) as isize - ph as isize;
                                let iw = (z * stride + s * dilation) as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < width as isize {
                                    y[((b * 3 + o) * oh + a) * ow + z] += x
                                        [((b * 2 + c) * h + ih as usize) * width + iw as usize]
                                        * w[((o * 2 + c) * kh + r) * kw + s];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    y
}

fn gradients(dev: &Device, is_2d: bool) -> Result<()> {
    // Batch > 1 makes the transposed upstream gradient non-contiguous.
    for (size, padding, stride, dilation) in
        [(5, 0, 1, 1), (5, 1, 2, 1), (7, 2, 1, 2), (2, 1, 1, 1)]
    {
        let h = if is_2d { size } else { 1 };
        let kh = if is_2d { 3 } else { 1 };
        let xs = values(2 * 2 * h * size);
        let ws = values(3 * 2 * kh * 3);
        let xd = if is_2d {
            vec![2, 2, h, size]
        } else {
            vec![2, 2, size]
        };
        let wd = if is_2d {
            vec![3, 2, kh, 3]
        } else {
            vec![3, 2, 3]
        };
        let x = Var::from_vec(xs.clone(), xd, dev)?;
        let w = Var::from_vec(ws.clone(), wd, dev)?;
        let y = if is_2d {
            x.conv2d(&w, padding, stride, dilation, 1)?
        } else {
            x.conv1d(&w, padding, stride, dilation, 1)?
        };
        let mut xr: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
        let mut wr: Vec<f64> = ws.iter().map(|&v| v as f64).collect();
        let eval = |x: &[f64], w: &[f64]| {
            reference(x, w, h, size, kh, 3, padding, stride, dilation, is_2d)
        };
        close(&y.flatten_all()?.to_vec1::<f32>()?, &eval(&xr, &wr));
        let loss = |x: &[f64], w: &[f64]| eval(x, w).iter().map(|v| v * v).sum::<f64>();
        let grads = y.sqr()?.sum_all()?.backward()?;
        for input in [true, false] {
            let n = if input { xr.len() } else { wr.len() };
            let mut expected = Vec::with_capacity(n);
            for i in 0..n {
                let old = if input { xr[i] } else { wr[i] };
                let eps = 1e-5;
                if input {
                    xr[i] = old + eps;
                } else {
                    wr[i] = old + eps;
                }
                let plus = loss(&xr, &wr);
                if input {
                    xr[i] = old - eps;
                } else {
                    wr[i] = old - eps;
                }
                let minus = loss(&xr, &wr);
                if input {
                    xr[i] = old;
                } else {
                    wr[i] = old;
                }
                expected.push((plus - minus) / (2. * eps));
            }
            let var = if input { &x } else { &w };
            let grad = grads.get(var).expect("missing gradient");
            assert_eq!(grad.dims(), var.dims());
            close(&grad.flatten_all()?.to_vec1::<f32>()?, &expected);
        }
    }
    Ok(())
}

fn conv1d_grad_reference(dev: &Device) -> Result<()> {
    gradients(dev, false)
}
fn conv2d_grad_reference(dev: &Device) -> Result<()> {
    gradients(dev, true)
}

fn conv_kernel_views(dev: &Device) -> Result<()> {
    for is_2d in [false, true] {
        let xd = if is_2d {
            vec![2, 2, 5, 5]
        } else {
            vec![2, 2, 5]
        };
        let x = Tensor::from_vec(values(xd.iter().product()), xd, dev)?;
        for transposed in [false, true] {
            let (a, b) = if transposed { (3, 2) } else { (2, 3) };
            let kd = if is_2d {
                vec![a, b, 3, 3]
            } else {
                vec![a, b, 3]
            };
            let storage = Tensor::from_vec(values(kd.iter().product()), kd, dev)?;
            let kernel = storage.transpose(0, 1)?;
            assert!(!kernel.is_contiguous());
            let run = |k: &Tensor| match (is_2d, transposed) {
                (false, false) => x.conv1d(k, 1, 1, 1, 1),
                (true, false) => x.conv2d(k, 1, 1, 1, 1),
                (false, true) => x.conv_transpose1d(k, 1, 0, 1, 1, 1),
                (true, true) => x.conv_transpose2d(k, 1, 0, 1, 1),
            };
            let actual = run(&kernel)?;
            let expected = run(&kernel.contiguous()?)?;
            assert_eq!(actual.dims(), expected.dims());
            close(
                &actual.flatten_all()?.to_vec1::<f32>()?,
                &expected
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
            );
        }
    }
    Ok(())
}

test_device!(
    conv1d_grad_reference,
    conv1d_grad_reference_cpu,
    conv1d_grad_reference_cuda,
    conv1d_grad_reference_metal
);
test_device!(
    conv2d_grad_reference,
    conv2d_grad_reference_cpu,
    conv2d_grad_reference_cuda,
    conv2d_grad_reference_metal
);
test_device!(
    conv_kernel_views,
    conv_kernel_views_cpu,
    conv_kernel_views_cuda,
    conv_kernel_views_metal
);
