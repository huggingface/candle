use anyhow::Result;
use candle::{test_utils, DType, Device, Module, Tensor, Var};
use candle_nn::{Conv3d, Conv3dConfig};

fn assert_close_tensors(
    got: &Tensor,
    expected: &Tensor,
    tolerance: f32,
    label: &str,
) -> Result<()> {
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch, got {}, expected {}",
        got.len(),
        expected.len()
    );
    let mut max_diff = 0f32;
    let mut max_idx = 0usize;
    for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.is_finite() && expected.is_finite(),
            "{label}: nonfinite value at {idx}"
        );
        let diff = (got - expected).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
        }
    }
    assert!(
        max_diff <= tolerance,
        "{label}: max diff {max_diff} at {max_idx}, got {}, expected {}, tolerance {tolerance}",
        got[max_idx],
        expected[max_idx]
    );
    Ok(())
}

fn var_from_vec_dtype<S: Into<candle::Shape>>(
    data: Vec<f32>,
    shape: S,
    dtype: DType,
    dev: &Device,
) -> Result<Var> {
    let tensor = Tensor::from_vec(data, shape, dev)?.to_dtype(dtype)?;
    Ok(Var::from_tensor(&tensor)?)
}

fn tensor_from_vec_dtype<S: Into<candle::Shape>>(
    data: Vec<f32>,
    shape: S,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dtype)?)
}

/* This test is based on the following script.
import torch
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

x = torch.arange(1 * 1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 1, 2, 2, 3) / 4 - 1
w = torch.arange(2 * 1 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 1, 1, 2, 2) / 3 - 0.5
b = torch.tensor([0.25, -0.75], dtype=torch.float32, requires_grad=True)
y = torch.nn.functional.conv3d(x, w, b, padding=(0, 1, 0))
print(y.shape)
print(y.flatten())
y.sum().backward()
print(b.grad)
*/
#[test]
#[allow(clippy::excessive_precision)]
fn conv3d_module_bias_cpu() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = (0..12).map(|v| v as f32 / 4. - 1.).collect::<Vec<_>>();
    let xs = Var::from_vec(xs, (1, 1, 2, 2, 3), dev)?;
    let ws = (0..8).map(|v| v as f32 / 3. - 0.5).collect::<Vec<_>>();
    let ws = Var::from_vec(ws, (2, 1, 1, 2, 2), dev)?;
    let bias = Var::from_slice(&[0.25f32, -0.75], 2, dev)?;
    let cfg = Conv3dConfig {
        padding: [0, 1, 0],
        ..Default::default()
    };
    let conv = Conv3d::new(ws.as_tensor().clone(), Some(bias.as_tensor().clone()), cfg);

    let out = conv.forward(&xs)?;
    assert_eq!(out.dims(), [1, 2, 2, 3, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&out.flatten_all()?, 4)?,
        [
            -0.2917, -0.125, 0.8333, 0.8333, 0.375, 0.2083, 0.7083, 0.875, 0.8333, 0.8333, -0.625,
            -0.7917, -3.625, -2.7917, -2.8333, -1.5, -0.9583, -0.4583, 1.375, 2.2083, 5.1667, 6.5,
            2.0417, 2.5417
        ]
    );

    let grads = out.sum_all()?.backward()?;
    let bias_grad = grads.get(&bias).unwrap();
    assert_eq!(
        test_utils::to_vec1_round(&bias_grad.flatten_all()?, 4)?,
        [12., 12.]
    );
    assert_eq!(grads.get(&xs).unwrap().dtype(), DType::F32);
    Ok(())
}

fn conv3d_module_low_precision_case(dtype: DType, tolerance: f32) -> Result<()> {
    let dev = &Device::Cpu;
    let input_shape = (1, 2, 3, 4, 4);
    let weight_shape = (2, 1, 2, 2, 2);
    let output_shape = (1, 2, 4, 2, 5);
    let cfg = Conv3dConfig {
        padding: [1, 0, 1],
        stride: [1, 2, 1],
        groups: 2,
        ..Default::default()
    };

    let xs = (0..(2 * 3 * 4 * 4))
        .map(|v| (v % 19) as f32 * 0.014 - 0.126)
        .collect::<Vec<_>>();
    let ws = (0..(2 * 2 * 2 * 2))
        .map(|v| (v % 7) as f32 * 0.016 - 0.048)
        .collect::<Vec<_>>();
    let bias = vec![0.02f32, -0.03];
    let grad_output = (0..(2 * 4 * 2 * 5))
        .map(|v| (v % 9) as f32 * 0.015 - 0.06)
        .collect::<Vec<_>>();

    let xs_f32 = Var::from_vec(xs.clone(), input_shape, dev)?;
    let ws_f32 = Var::from_vec(ws.clone(), weight_shape, dev)?;
    let bias_f32 = Var::from_vec(bias.clone(), 2, dev)?;
    let conv_f32 = Conv3d::new(
        ws_f32.as_tensor().clone(),
        Some(bias_f32.as_tensor().clone()),
        cfg,
    );
    let out_f32 = conv_f32.forward(&xs_f32)?;
    assert_eq!(out_f32.dims(), [1, 2, 4, 2, 5]);
    let grad_output_f32 = Tensor::from_vec(grad_output.clone(), output_shape, dev)?;
    let loss_f32 = out_f32.mul(&grad_output_f32)?.sum_all()?;
    let grads_f32 = loss_f32.backward()?;

    let xs_lp = var_from_vec_dtype(xs, input_shape, dtype, dev)?;
    let ws_lp = var_from_vec_dtype(ws, weight_shape, dtype, dev)?;
    let bias_lp = var_from_vec_dtype(bias, 2, dtype, dev)?;
    let conv_lp = Conv3d::new(
        ws_lp.as_tensor().clone(),
        Some(bias_lp.as_tensor().clone()),
        cfg,
    );
    let out_lp = conv_lp.forward(&xs_lp)?;
    assert_eq!(out_lp.dims(), out_f32.dims());
    assert_eq!(out_lp.dtype(), dtype);
    assert_close_tensors(
        &out_lp,
        &out_f32,
        tolerance,
        "conv3d module low precision output",
    )?;

    let grad_output_lp = tensor_from_vec_dtype(grad_output, output_shape, dtype, dev)?;
    let loss_lp = out_lp.mul(&grad_output_lp)?.sum_all()?;
    let grads_lp = loss_lp.backward()?;
    let grad_xs_lp = grads_lp.get(&xs_lp).unwrap();
    let grad_ws_lp = grads_lp.get(&ws_lp).unwrap();
    let grad_bias_lp = grads_lp.get(&bias_lp).unwrap();
    assert_eq!(grad_xs_lp.dtype(), dtype);
    assert_eq!(grad_ws_lp.dtype(), dtype);
    assert_eq!(grad_bias_lp.dtype(), dtype);
    assert_close_tensors(
        grad_xs_lp,
        grads_f32.get(&xs_f32).unwrap(),
        tolerance,
        "conv3d module low precision input grad",
    )?;
    assert_close_tensors(
        grad_ws_lp,
        grads_f32.get(&ws_f32).unwrap(),
        tolerance,
        "conv3d module low precision weight grad",
    )?;
    assert_close_tensors(
        grad_bias_lp,
        grads_f32.get(&bias_f32).unwrap(),
        tolerance,
        "conv3d module low precision bias grad",
    )?;
    Ok(())
}

#[test]
fn conv3d_module_f16_bf16_cpu() -> Result<()> {
    conv3d_module_low_precision_case(DType::F16, 2e-2)?;
    conv3d_module_low_precision_case(DType::BF16, 1e-1)
}

#[test]
fn conv_transpose3d_module_cpu() -> Result<()> {
    use candle_nn::{
        conv_transpose3d, conv_transpose3d_no_bias, ConvTranspose3dConfig, VarBuilder,
    };
    let dev = &Device::Cpu;
    let cfg = ConvTranspose3dConfig {
        stride: [2, 1, 1],
        output_padding: [1, 0, 0],
        groups: 2,
        ..Default::default()
    };
    // PyTorch: F.conv_transpose3d(x, w, b, stride=(2,1,1),
    // output_padding=(1,0,0), groups=2), followed by y.sum().backward().
    for dtype in [DType::F64, DType::F32, DType::F16, DType::BF16] {
        let x = var_from_vec_dtype(vec![1., 2., 3., 4.], (1, 2, 2, 1, 1), dtype, dev)?;
        let w = var_from_vec_dtype(vec![2., 3.], (2, 1, 1, 1, 1), dtype, dev)?;
        let b = var_from_vec_dtype(vec![0.5, -0.5], 2, dtype, dev)?;
        let tensors = std::collections::HashMap::from([
            ("proj.weight".to_string(), w.as_tensor().clone()),
            ("proj.bias".to_string(), b.as_tensor().clone()),
        ]);
        let vb = VarBuilder::from_tensors(tensors, dtype, dev).pp("proj");
        let layer = conv_transpose3d(2, 2, [1; 3], cfg, vb.clone())?;
        assert_eq!(layer.config(), &cfg);
        assert_eq!(layer.weight().dims(), [2, 1, 1, 1, 1]);
        assert_eq!(layer.bias().unwrap().dims(), [2]);
        let y = layer.forward(&x)?;
        assert_eq!(y.dims(), [1, 2, 4, 1, 1]);
        let expected = Tensor::from_slice(&[2.5f32, 0.5, 4.5, 0.5, 8.5, -0.5, 11.5, -0.5], 8, dev)?;
        assert_close_tensors(&y, &expected, 0., "transpose module output")?;
        let grads = y.sum_all()?.backward()?;
        for (got, expected) in [
            (grads.get(&x).unwrap(), vec![2f32, 2., 3., 3.]),
            (grads.get(&w).unwrap(), vec![3f32, 7.]),
            (grads.get(&b).unwrap(), vec![4f32, 4.]),
        ] {
            assert_eq!(got.dtype(), dtype);
            assert_close_tensors(
                got,
                &Tensor::from_slice(&expected, expected.len(), dev)?,
                0.,
                "transpose module gradient",
            )?;
        }
        let no_bias = conv_transpose3d_no_bias(2, 2, [1; 3], cfg, vb)?;
        assert!(no_bias.bias().is_none());
        let y_no_bias = no_bias.forward(&x)?;
        assert_close_tensors(
            &y_no_bias.broadcast_add(&b.reshape((1, 2, 1, 1, 1))?)?,
            &y,
            0.,
            "transpose no bias",
        )?;
    }
    Ok(())
}

#[test]
fn conv_transpose3d_builder_validation_cpu() -> Result<()> {
    use candle_nn::{
        conv_transpose3d, conv_transpose3d_no_bias, ConvTranspose3dConfig, VarBuilder, VarMap,
    };
    let dev = &Device::Cpu;
    let cfg = ConvTranspose3dConfig {
        groups: 2,
        ..Default::default()
    };
    let vars = VarMap::new();
    let layer = conv_transpose3d(
        4,
        6,
        [2, 1, 3],
        cfg,
        VarBuilder::from_varmap(&vars, DType::F32, dev),
    )?;
    assert_eq!(layer.weight().dims(), [4, 3, 2, 1, 3]);
    assert_eq!(layer.bias().unwrap().dims(), [6]);
    let bound = 1f32 / 18f32.sqrt();
    for t in [layer.weight(), layer.bias().unwrap()] {
        assert!(t
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&v| v.is_finite() && v.abs() <= bound));
    }
    let vb = VarBuilder::zeros(DType::F32, dev);
    for (input, output, groups, kernel) in [
        (3, 6, 2, [1; 3]),
        (4, 5, 2, [1; 3]),
        (0, 2, 1, [1; 3]),
        (2, 0, 1, [1; 3]),
        (2, 2, 0, [1; 3]),
        (2, 2, 1, [1, 0, 1]),
        (2, 2, 1, [usize::MAX, 2, 1]),
    ] {
        let cfg = ConvTranspose3dConfig {
            groups,
            ..Default::default()
        };
        assert!(conv_transpose3d(input, output, kernel, cfg, vb.clone()).is_err());
        assert!(conv_transpose3d_no_bias(input, output, kernel, cfg, vb.clone()).is_err());
    }
    for axis in 0..3 {
        let mut cfg = ConvTranspose3dConfig::default();
        cfg.stride[axis] = 0;
        assert!(conv_transpose3d(2, 2, [1; 3], cfg, vb.clone()).is_err());
        cfg = ConvTranspose3dConfig::default();
        cfg.dilation[axis] = 0;
        assert!(conv_transpose3d_no_bias(2, 2, [1; 3], cfg, vb.clone()).is_err());
        cfg = ConvTranspose3dConfig::default();
        cfg.output_padding[axis] = 1;
        assert!(conv_transpose3d(2, 2, [1; 3], cfg, vb.clone()).is_err());
    }
    Ok(())
}
