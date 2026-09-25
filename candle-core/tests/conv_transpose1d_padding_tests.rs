use candle_core::{DType, Device, Result, Tensor, Var};

#[test]
fn conv_transpose1d_padding_cpu() -> Result<()> {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        for (batch, groups) in [(1, 1), (2, 1), (2, 2)] {
            for (len, kernel, stride, padding, output_padding, dilation) in [
                (7, 3, 1, 1, 0, 1),
                (7, 3, 2, 1, 1, 1),
                (7, 3, 5, 2, 3, 1),
                (5, 8, 4, 2, 0, 1),
                (5, 10, 5, 3, 1, 1),
                (4, 1, 3, 0, 2, 1),
                (9, 2, 1, 3, 0, 1),
                (6, 3, 1, 2, 0, 1),
                (9, 3, 2, 2, 1, 2),
                (5, 4, 3, 0, 2, 2),
            ] {
                let (ci, co) = (4, 3);
                let out_len =
                    (len - 1) * stride + dilation * (kernel - 1) + output_padding + 1 - 2 * padding;
                let x = Tensor::from_vec(
                    (0..(batch + 1) * ci * (len + 2))
                        .map(|i| ((i * 7 % 17) as f32 - 8.) / 16.)
                        .collect::<Vec<_>>(),
                    (batch + 1, ci, len + 2),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, batch)?
                .narrow(2, 1, len)?;
                let w = Tensor::from_vec(
                    (0..(ci + 1) * co * (kernel + 2))
                        .map(|i| ((i * 11 % 19) as f32 - 9.) / 16.)
                        .collect::<Vec<_>>(),
                    (ci + 1, co, kernel + 2),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, ci)?
                .narrow(2, 1, kernel)?;
                let x_values = x.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
                let w_values = w.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
                let mut expected = vec![0f64; batch * groups * co * out_len];
                for b in 0..batch {
                    for c in 0..ci {
                        let group = c / (ci / groups);
                        for pos in 0..len {
                            for o in 0..co {
                                for tap in 0..kernel {
                                    let dst =
                                        (pos * stride + tap * dilation) as isize - padding as isize;
                                    if dst >= 0 && dst < out_len as isize {
                                        expected[((b * groups + group) * co + o) * out_len
                                            + dst as usize] += x_values[(b * ci + c) * len + pos]
                                            * w_values[(c * co + o) * kernel + tap];
                                    }
                                }
                            }
                        }
                    }
                }
                for input in [
                    &x,
                    &x.contiguous()?,
                    &x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?,
                ] {
                    for weights in [&w, &w.contiguous()?] {
                        let actual = input.conv_transpose1d(
                            weights,
                            padding,
                            output_padding,
                            stride,
                            dilation,
                            groups,
                        )?;
                        assert_eq!(actual.dims(), [batch, groups * co, out_len]);
                        let actual = actual
                            .to_dtype(DType::F64)?
                            .flatten_all()?
                            .to_vec1::<f64>()?;
                        let tolerance = match dtype {
                            DType::BF16 => 0.04,
                            DType::F16 => 0.005,
                            DType::F32 => 1e-5,
                            _ => 1e-12,
                        };
                        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                            assert!((a - e).abs() <= tolerance * (1. + e.abs()),
                                "{dtype:?} b={batch} g={groups} l={len} k={kernel} s={stride} p={padding} op={output_padding} d={dilation} index={i}: {a} != {e}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn conv1d_padded_input_gradient_cpu() -> Result<()> {
    for len in [13, 14] {
        let (ci, co, kernel, stride, padding) = (3, 4, 3, 2, 1);
        let out_len = (len + 2 * padding - kernel) / stride + 1;
        let x = Var::from_vec(vec![0.125f64; ci * len], (1, ci, len), &Device::Cpu)?;
        let w_values: Vec<_> = (0..co * ci * kernel)
            .map(|i| (i % 11) as f64 / 8. - 0.5)
            .collect();
        let g_values: Vec<_> = (0..co * out_len)
            .map(|i| (i % 7) as f64 / 4. - 0.5)
            .collect();
        let w = Tensor::from_vec(w_values.clone(), (co, ci, kernel), &Device::Cpu)?;
        let g = Tensor::from_vec(g_values.clone(), (1, co, out_len), &Device::Cpu)?;
        let loss = x.conv1d(&w, padding, stride, 1, 1)?.mul(&g)?.sum_all()?;
        let grads = loss.backward()?;
        let actual = grads.get(&x).unwrap().flatten_all()?.to_vec1::<f64>()?;
        let mut expected = vec![0.; ci * len];
        for c in 0..ci {
            for o in 0..co {
                for pos in 0..out_len {
                    for tap in 0..kernel {
                        let dst = (pos * stride + tap) as isize - padding as isize;
                        if dst >= 0 && dst < len as isize {
                            expected[c * len + dst as usize] +=
                                g_values[o * out_len + pos] * w_values[(o * ci + c) * kernel + tap];
                        }
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }
    Ok(())
}
