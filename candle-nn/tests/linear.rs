#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor, Var};
use candle_nn::{linear_b, linear_no_bias, Module, VarBuilder, VarMap};

#[test]
fn transposed_linear_weights_preserve_shape_and_values() -> Result<()> {
    let device = &Device::Cpu;
    let weights = [[1f32, 2., 3.], [-2., 4., 1.]];
    let vb = VarBuilder::from_tensors(
        [
            ("dense.weight".to_string(), Tensor::new(&weights, device)?),
            ("dense.bias".to_string(), Tensor::new(&[5f32, -3.], device)?),
        ]
        .into_iter()
        .collect(),
        DType::F32,
        device,
    );
    let input = Tensor::new(&[[2f32, 0., -1.], [1., -1., 2.]], device)?;

    for (bias, expected) in [(false, [-1f32, -5., 5., -4.]), (true, [4., -8., 10., -7.])] {
        let standard = linear_b(3, 2, bias, vb.pp("dense"))?;
        let prepared = linear_b(
            3,
            2,
            bias,
            vb.clone()
                .with_transposed_linear_weights()
                .rename_f(str::to_string)
                .pp("dense"),
        )?;
        assert!(standard.weight().is_contiguous());
        assert!(!standard.weight().t()?.is_contiguous());
        assert!(prepared.weight().t()?.is_contiguous());
        assert_eq!(prepared.weight().to_vec2::<f32>()?, weights);

        let inputs = [
            (input.clone(), 1),
            (input.broadcast_as((2, 2, 3))?, 2),
            (input.broadcast_as((2, 2, 2, 3))?, 4),
        ];
        for (input, batches) in inputs {
            let baseline = standard.forward(&input)?;
            let output = prepared.forward(&input)?;
            assert_eq!(output.dims(), baseline.dims());
            assert_eq!(
                output.flatten_all()?.to_vec1::<f32>()?,
                expected.repeat(batches)
            );
        }
    }
    Ok(())
}

#[test]
fn transposed_linear_weights_keep_variables_live() -> Result<()> {
    let device = &Device::Cpu;
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, device);
    let standard = linear_no_bias(3, 2, vb.clone())?;
    let opted_in = linear_no_bias(3, 2, vb.with_transposed_linear_weights())?;
    let input = Tensor::new(&[[2f32, 0., -1.]], device)?;

    vars.set_one("weight", Tensor::zeros((2, 3), DType::F32, device)?)?;
    assert_eq!(opted_in.forward(&input)?.to_vec2::<f32>()?, [[0., 0.]]);
    vars.set_one("weight", Tensor::ones((2, 3), DType::F32, device)?)?;
    let output = opted_in.forward(&input)?;
    assert_eq!(output.to_vec2::<f32>()?, [[1., 1.]]);
    let gradients = output.sum_all()?.backward()?;
    assert_eq!(
        gradients.get(standard.weight()).unwrap().to_vec2::<f32>()?,
        [[2., 0., -1.], [2., 0., -1.]]
    );
    Ok(())
}

#[test]
fn transposed_linear_weights_skip_tracked_views_and_other_dtypes() -> Result<()> {
    let device = &Device::Cpu;
    let var = Var::ones((2, 3), DType::F32, device)?;
    let tracked = var.t()?.t()?;
    assert!(tracked.track_op());
    assert!(!tracked.is_variable());
    let f64_weight = Tensor::ones((2, 3), DType::F64, device)?;

    for weight in [tracked, f64_weight] {
        let vb = VarBuilder::from_tensors(
            [("weight".to_string(), weight.clone())]
                .into_iter()
                .collect(),
            weight.dtype(),
            device,
        );
        let layer = linear_no_bias(3, 2, vb.with_transposed_linear_weights())?;
        assert_eq!(layer.weight().id(), weight.id());
    }
    Ok(())
}
