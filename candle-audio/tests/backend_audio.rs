#![cfg(feature = "vulkan")]
//! Small regressions for backend operations exercised by native audio models.
use candle::{DType, Device, IndexOp, Module, Tensor};

#[test]
#[ignore = "requires a real Vulkan device"]
fn vulkan_embedding_materializes_i64_id_views_without_reusing_offsets_or_strides() {
    let device = Device::new_vulkan(0).unwrap();
    let table = Tensor::new(&[[10f32, 11.], [20., 21.], [30., 31.]], &device).unwrap();
    for dtype in [DType::I64, DType::U32] {
        let ids = Tensor::new(&[99i64, 0, 2, 1, 99], &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let view = ids.narrow(0, 1, 3).unwrap();
        let result = table
            .index_select(&view, 0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(result, vec![vec![10., 11.], vec![30., 31.], vec![20., 21.]]);
        let ids = Tensor::new(&[[0i64, 99], [2, 99], [1, 99]], &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let view = ids.i((.., 0)).unwrap();
        let result = table
            .index_select(&view, 0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(result, vec![vec![10., 11.], vec![30., 31.], vec![20., 21.]]);
    }
}

#[test]
#[ignore = "requires a real Vulkan device"]
fn vulkan_dac_snake_matches_cpu_with_broadcast_channel_parameters() {
    use candle_nn::VarBuilder;
    use candle_transformers::models::dac::Snake1d;
    use std::collections::HashMap;
    let input: Vec<f32> = (0..51).map(|i| i as f32 * 0.031 - 0.7).collect();
    let mut outputs = Vec::new();
    for device in [Device::Cpu, Device::new_vulkan(0).unwrap()] {
        let alpha = Tensor::new(&[1f32, 1.5, 2.], &device)
            .unwrap()
            .reshape((1, 3, 1))
            .unwrap();
        let vb = VarBuilder::from_tensors(
            HashMap::from([("alpha".to_owned(), alpha)]),
            DType::F32,
            &device,
        );
        let model = Snake1d::new(3, vb).unwrap();
        let x = Tensor::from_vec(input.clone(), (1, 3, 17), &device).unwrap();
        outputs.push(
            model
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }
    for (cpu, gpu) in outputs[0].iter().zip(&outputs[1]) {
        assert!((cpu - gpu).abs() < 1e-5, "{cpu} != {gpu}");
    }
}
