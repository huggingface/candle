use candle_core::{test_device, test_utils, DType, Device, IndexOp, Result, Tensor, D};
use float8::F8E4M3;

fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}

fn ones(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::ones((2, 3), DType::U8, device)?.to_vec2::<u8>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::U32, device)?.to_vec2::<u32>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::I64, device)?.to_vec2::<i64>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::F32, device)?.to_vec2::<f32>()?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    if !device.is_metal() {
        assert_eq!(
            Tensor::ones((2, 3), DType::F64, device)?.to_vec2::<f64>()?,
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        );
    }
    assert_eq!(
        Tensor::ones((2, 3), DType::F16, device)?.to_vec2::<half::f16>()?,
        [
            [
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0)
            ],
            [
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0),
                half::f16::from_f32(1.0)
            ]
        ],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::BF16, device)?.to_vec2::<half::bf16>()?,
        [
            [
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(1.0)
            ],
            [
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(1.0)
            ]
        ],
    );

    if !device.is_metal() {
        assert_eq!(
            Tensor::ones((2, 3), DType::F8E4M3, device)?.to_vec2::<F8E4M3>()?,
            [
                [
                    F8E4M3::from_f32(1.),
                    F8E4M3::from_f32(1.),
                    F8E4M3::from_f32(1.)
                ],
                [
                    F8E4M3::from_f32(1.),
                    F8E4M3::from_f32(1.),
                    F8E4M3::from_f32(1.)
                ]
            ],
        );
    }
    Ok(())
}

fn full(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((3, 4), DType::U32, device)?;
    tensor.const_set(42u32.into())?;
    assert_eq!(
        tensor.to_vec2::<u32>()?,
        [[42, 42, 42, 42], [42, 42, 42, 42], [42, 42, 42, 42]]
    );
    tensor.i((.., 2))?.const_set(1337u32.into())?;
    assert_eq!(
        tensor.to_vec2::<u32>()?,
        [[42, 42, 1337, 42], [42, 42, 1337, 42], [42, 42, 1337, 42]]
    );
    tensor.i((2, ..))?.const_set(1u32.into())?;
    assert_eq!(
        tensor.to_vec2::<u32>()?,
        [[42, 42, 1337, 42], [42, 42, 1337, 42], [1, 1, 1, 1]]
    );
    Ok(())
}

fn const_set(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::full(42u32, (2, 3), device)?.to_vec2::<u32>()?,
        [[42, 42, 42], [42, 42, 42]],
    );
    Ok(())
}

fn arange(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::arange(0u8, 5u8, device)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4],
    );
    assert_eq!(
        Tensor::arange_step(0u8, 5u8, 2, device)?.to_vec1::<u8>()?,
        [0, 2, 4],
    );
    assert_eq!(
        Tensor::arange_step(0u8, 5u8, 3, device)?.to_vec1::<u8>()?,
        [0, 3],
    );
    assert_eq!(
        Tensor::arange_step(5i64, 0i64, -1, device)?.to_vec1::<i64>()?,
        [5, 4, 3, 2, 1],
    );

    if !device.is_metal() {
        assert_eq!(
            Tensor::arange_step(
                F8E4M3::from_f32(0.),
                F8E4M3::from_f32(5.),
                F8E4M3::from_f32(2.),
                device
            )?
            .to_vec1::<F8E4M3>()?,
            [
                F8E4M3::from_f32(0.),
                F8E4M3::from_f32(2.),
                F8E4M3::from_f32(4.),
            ],
        );
    }

    Ok(())
}

fn add_mul(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], device)?;
    let dim1 = tensor.dims1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}

fn tensor_2d(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content, data);
    Ok(())
}

fn clamp(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.clamp(1.5, 6.2)?;
    assert_eq!(
        tensor.to_vec2::<f32>()?,
        [[3.0, 1.5, 4.0, 1.5, 5.0], [2.0, 1.5, 6.2, 6.2, 2.0]],
    );
    Ok(())
}

fn asort(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1.1, 5.], [2.1, 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let indexes = tensor.arg_sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2::<u32>()?,
        [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    let indexes = tensor.arg_sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2::<u32>()?,
        [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    let (sorted, indexes) = tensor.sort_last_dim(true)?;
    assert_eq!(
        indexes.to_vec2::<u32>()?,
        [[1, 3, 0, 2, 4], [1, 4, 0, 2, 3]],
    );
    assert_eq!(
        sorted.to_vec2::<f32>()?,
        [[1.0, 1.1, 3.0, 4.0, 5.0], [1.0, 2.0, 2.1, 7.0, 8.0]]
    );
    let (sorted, indexes) = tensor.sort_last_dim(false)?;
    assert_eq!(
        indexes.to_vec2::<u32>()?,
        [[4, 2, 0, 3, 1], [3, 2, 0, 4, 1]],
    );
    assert_eq!(
        sorted.to_vec2::<f32>()?,
        [[5.0, 4.0, 3.0, 1.1, 1.0], [8.0, 7.0, 2.1, 2.0, 1.0]]
    );

    let offset_view = Tensor::new(&[[1f32, 2., 3.], [30., 10., 20.]], device)?.narrow(0, 1, 1)?;
    let (sorted, indexes) = offset_view.sort_last_dim(true)?;
    assert_eq!(indexes.to_vec2::<u32>()?, [[1, 2, 0]]);
    assert_eq!(sorted.to_vec2::<f32>()?, [[10., 20., 30.]]);

    let (sorted, indexes) = offset_view.sort_last_dim(false)?;
    assert_eq!(indexes.to_vec2::<u32>()?, [[0, 2, 1]]);
    assert_eq!(sorted.to_vec2::<f32>()?, [[30., 20., 10.]]);
    Ok(())
}

/// Test sorting a large tensor that exceeds 1024 elements.
fn asort_big(device: &Device) -> Result<()> {
    // Skip on metal for now
    if device.is_metal() {
        return Ok(());
    }
    const SIZE: usize = 2000;
    let data: Vec<f32> = (0..SIZE).map(|x| (SIZE - x) as f32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;

    let indexes = tensor.arg_sort_last_dim(true)?;
    let expected_indexes: Vec<u32> = (0..SIZE).rev().map(|x| x as u32).collect();
    assert_eq!(indexes.to_vec1::<u32>()?, expected_indexes);

    let indexes = tensor.arg_sort_last_dim(false)?;
    let expected_indexes: Vec<u32> = (0..SIZE).map(|x| x as u32).collect();
    assert_eq!(indexes.to_vec1::<u32>()?, expected_indexes);
    Ok(())
}

fn unary_op(device: &Device) -> Result<()> {
    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.gelu()?, 4)?,
        [
            [-0.0036, 0.8412, 3.9999, -0.046, 0.3457],
            [2.6911, -0.0647, -0.1091, 1.7353, 2.7933]
        ]
    );
    let t_f16 = tensor.to_dtype(DType::F16)?.gelu()?.to_dtype(DType::F32)?;
    let max_diff = (tensor.gelu()? - t_f16)?.flatten_all()?.max(0)?;
    assert!(max_diff.to_vec0::<f32>()? < 5e-3);
    assert_eq!(
        test_utils::to_vec2_round(&tensor.gelu_erf()?, 4)?,
        [
            [-0.004, 0.8413, 3.9999, -0.046, 0.3457],
            [2.6906, -0.0647, -0.1091, 1.7353, 2.7928]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.erf()?, 4)?,
        [
            [-1.0, 0.8427, 1.0, -0.1125, 0.5205],
            [0.9999, -0.9891, -0.3079, 0.9891, 0.9999]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.silu()?, 4)?,
        [
            [-0.1423, 0.7311, 3.9281, -0.0475, 0.3112],
            [2.53, -0.2553, -0.1205, 1.5447, 2.6395]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.ceil()?, 4)?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -1.0, -0.0, 2.0, 3.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.floor()?, 4)?,
        [[-3.0, 1.0, 4.0, -1.0, 0.0], [2.0, -2.0, -1.0, 1.0, 2.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.round()?, 4)?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -2.0, -0.0, 2.0, 3.0]]
    );
    let tensor = Tensor::new(&[2997.9246, 314.15926f32], device)?;
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(2)?, 4)?,
        [2997.92, 314.16]
    );
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(-2)?, 4)?,
        [3000.0, 300.]
    );
    let tensor = Tensor::new(
        &[-1.01f32, -0.9, -0.1, 0.0, -0.0, 0.1, 0.9, 1.0, 1.1],
        device,
    )?;
    assert_eq!(
        tensor.sign()?.to_vec1::<f32>()?,
        [-1., -1., -1., 0., 0., 1., 1., 1., 1.]
    );
    let tensor = Tensor::new(&[-1.0f32, 0., -2., 3.], device)?;
    let y = tensor.elu(2.)?;
    assert_eq!(
        test_utils::to_vec1_round(&y, 4)?,
        [-1.2642, 0.0000, -1.7293, 3.0000]
    );
    // This test failed on metal prior to the following PR:
    // https://github.com/huggingface/candle/pull/2490
    let y = tensor.reshape((2, 2))?.t()?.elu(2.)?.flatten_all()?;
    assert_eq!(
        test_utils::to_vec1_round(&y, 4)?,
        [-1.2642, -1.7293, 0.0000, 3.0000]
    );
    Ok(())
}

fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);

    let min = tensor1.minimum(&(&tensor2 * 0.5)?)?;
    let max = tensor1.maximum(&(&tensor2 * 0.5)?)?;
    assert_eq!(
        min.to_vec2::<f32>()?,
        [[2.5, 1.0, 2.5, 1.0, 2.5], [1.0, 0.5, 3.5, 4.0, 1.0]],
    );
    assert_eq!(
        max.to_vec2::<f32>()?,
        [[3.0, 2.5, 4.0, 2.5, 5.0], [2.0, 1.0, 7.0, 8.0, 2.0]]
    );

    let (b, s, h, pad) = (3usize, 4usize, 5usize, 2usize);
    let base =
        Tensor::arange(1f32, (b * (s + pad) * h + 1) as f32, device)?.reshape((b, s + pad, h))?;
    let ub = base.narrow(1, 0, s)?; // uniform blocks: strides [h * (s + pad), h, 1]
    let cont = ub.contiguous()?;
    let scale = Tensor::arange(1f32, (b * s * h + 1) as f32, device)?.reshape((b, s, h))?;
    assert_eq!(
        (&ub + &scale)?.to_vec3::<f32>()?,
        (&cont + &scale)?.to_vec3::<f32>()?,
    );
    assert_eq!(
        (&ub * &scale)?.to_vec3::<f32>()?,
        (&cont * &scale)?.to_vec3::<f32>()?,
    );
    // contiguous lhs, uniform blocks rhs.
    assert_eq!(
        (&scale + &ub)?.to_vec3::<f32>()?,
        (&scale + &cont)?.to_vec3::<f32>()?,
    );
    Ok(())
}

fn ternary_op(device: &Device) -> Result<()> {
    let data = &[[0u8, 1, 0, 1, 0], [1, 1, 1, 0, 0]];
    let ids = Tensor::new(data, device)?;
    let data = &[[0f32, 1., 2., 3., 4.], [5., 6., 7., 8., 9.]];
    let a = Tensor::new(data, device)?;
    let data = &[[10f32, 11., 12., 13., 14.], [15., 16., 17., 18., 19.]];
    let b = Tensor::new(data, device)?;
    let tensor = ids.where_cond(&a, &b)?;
    let dims = tensor.dims();
    assert_eq!(dims, [2, 5]);
    let result: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    assert_eq!(result, [10., 1., 12., 3., 14., 5., 6., 7., 18., 19.]);

    let fill = Tensor::new(7f32, device)?.broadcast_as((2, 5))?; // strides [0, 0]
    assert_eq!(
        ids.where_cond(&fill, &b)?.to_vec2::<f32>()?,
        [[10., 7., 12., 7., 14.], [7., 7., 7., 18., 19.]],
    );

    let mask = Tensor::new(&[[0u8, 1, 1, 0, 1]], device)?.broadcast_as((2, 5))?; // strides [0, 1]
    assert_eq!(
        mask.where_cond(&a, &b)?.to_vec2::<f32>()?,
        mask.contiguous()?.where_cond(&a, &b)?.to_vec2::<f32>()?,
    );

    let a_t = Tensor::arange(0f32, 10., device)?.reshape((5, 2))?.t()?;
    let ids_t = Tensor::new(&[[0u8, 1], [1, 1], [0, 0], [1, 0], [1, 1]], device)?.t()?;
    assert_eq!(
        ids_t.where_cond(&a_t, &b)?.to_vec2::<f32>()?,
        ids_t
            .contiguous()?
            .where_cond(&a_t.contiguous()?, &b)?
            .to_vec2::<f32>()?,
    );

    let narrowed = Tensor::arange(0f32, 14., device)?
        .reshape((2, 7))?
        .narrow(1, 1, 5)?; // strides [7, 1], offset 1
    assert_eq!(
        ids.where_cond(&fill, &narrowed)?.to_vec2::<f32>()?,
        ids.where_cond(&fill.contiguous()?, &narrowed.contiguous()?)?
            .to_vec2::<f32>()?,
    );

    let fill2 = Tensor::new(9f32, device)?.broadcast_as((2, 5))?;
    assert_eq!(
        ids.where_cond(&a, &fill)?.to_vec2::<f32>()?,
        ids.where_cond(&a, &fill.contiguous()?)?.to_vec2::<f32>()?,
    );
    assert_eq!(
        ids.where_cond(&fill, &fill2)?.to_vec2::<f32>()?,
        ids.where_cond(&fill.contiguous()?, &fill2.contiguous()?)?
            .to_vec2::<f32>()?,
    );

    let row_mask = Tensor::new(&[[1u8], [0u8]], device)?.broadcast_as((2, 5))?; // strides [1, 0]
    let row_ref = row_mask.contiguous()?;
    assert_eq!(
        row_mask.where_cond(&a, &b)?.to_vec2::<f32>()?,
        row_ref.where_cond(&a, &b)?.to_vec2::<f32>()?,
    );
    assert_eq!(
        row_mask.where_cond(&a, &fill)?.to_vec2::<f32>()?,
        row_ref
            .where_cond(&a, &fill.contiguous()?)?
            .to_vec2::<f32>()?,
    );
    assert_eq!(
        row_mask.where_cond(&fill, &b)?.to_vec2::<f32>()?,
        row_ref
            .where_cond(&fill.contiguous()?, &b)?
            .to_vec2::<f32>()?,
    );
    assert_eq!(
        row_mask.where_cond(&fill, &fill2)?.to_vec2::<f32>()?,
        row_ref
            .where_cond(&fill.contiguous()?, &fill2.contiguous()?)?
            .to_vec2::<f32>()?,
    );
    Ok(())
}

fn transpose(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?.t()?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2::<f32>()?,
        &[[3f32, 2.], [1., 1.], [4., 7.], [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t()?.to_vec2::<f32>()?, data);
    assert_eq!(tensor.contiguous()?.t()?.to_vec2::<f32>()?, data);
    assert_eq!(((tensor + 1.)?.t()? - 1.)?.to_vec2::<f32>()?, data);
    Ok(())
}

fn var(device: &Device) -> Result<()> {
    // Values taken from https://pytorch.org/docs/stable/generated/torch.var.html
    let data = &[
        [0.2035f32, 1.2959, 1.8101, -0.4644],
        [1.5027, -0.3270, 0.5905, 0.6538],
        [-1.5745, 1.3330, -0.5596, -0.6548],
        [0.1264, -0.5080, 1.6420, 0.1992],
    ];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.var_keepdim(1)?, 4)?,
        &[[1.0631], [0.559], [1.4893], [0.8258]]
    );
    Ok(())
}

fn sum(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.sum_keepdim(2)?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec3::<u32>()?,
        &[[[5, 2, 11], [9, 7, 17]]],
    );
    assert_eq!(tensor.sum_keepdim((0, 2, 1))?.to_vec3::<u32>()?, &[[[51]]],);
    assert_eq!(
        tensor.t()?.sum_keepdim(1)?.t()?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim((2, 1))?.to_vec3::<u32>()?,
        &[[[8 + 15]], [[10 + 18]]]
    );
    let data: Vec<u32> = (0..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0)?.to_vec1::<u32>()?, &[7998000]);
    let tensor = tensor.reshape((2000, 2))?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2::<u32>()?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[3998000, 4000000]]
    );

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2::<u32>()?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[3998000, 4000000]]
    );

    let t1 = tensor.reshape((200, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.sum_keepdim((0, 1, 2))?.to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim(2)?
                .sum_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim((1, 2))?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(1)?
                .sum_keepdim((0, 2))?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [398000, 398200, 398400, 398600],
                [398800, 399000, 399200, 399400],
                [399600, 399800, 400000, 400200],
                [400400, 400600, 400800, 401000],
                [401200, 401400, 401600, 401800]
            ]]
        );
    }

    let data = &[[1u8, 2, 3], [4, 5, 6]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(tensor.sum_keepdim(1)?.to_vec2::<u8>()?, &[[6], [15]]);
    let data = &[[1i64, 2, 3], [4, 5, 6]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(tensor.sum_keepdim(1)?.to_vec2::<i64>()?, &[[6], [15]]);

    let mut data = vec![16_777_217u32];
    data.extend([1u32; 32]);
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0)?.to_vec1::<u32>()?, &[16_777_249]);
    if !device.is_metal() {
        let mut data = vec![16_777_217f64];
        data.extend([1f64; 32]);
        let tensor = Tensor::new(data.as_slice(), device)?;
        assert_eq!(tensor.sum_keepdim(0)?.to_vec1::<f64>()?, &[16_777_249.]);
    }

    let empty = Tensor::new(&[[1f32, 2.]; 4], device)?.narrow(1, 0, 0)?;
    assert_eq!(empty.sum(1)?.to_vec1::<f32>()?, &[0.; 4]);
    Ok(())
}

fn min(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.min_keepdim(2)?.to_vec3::<u32>()?,
        &[[[1], [1]], [[1], [2]]]
    );
    assert_eq!(
        tensor.min_keepdim(0)?.to_vec3::<u32>()?,
        &[[[2, 1, 4], [1, 2, 8]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.min_keepdim(0)?.to_vec1::<u32>()?, &[200]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2::<u32>()?, &[[200, 201]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2::<u32>()?, &[[200, 201]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .min_keepdim(0)?
                .min_keepdim(2)?
                .min_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[200]]]
        );
        assert_eq!(
            tensor.min_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [200, 201, 202, 203],
                [204, 205, 206, 207],
                [208, 209, 210, 211],
                [212, 213, 214, 215],
                [216, 217, 218, 219]
            ]]
        );
    }
    Ok(())
}

fn max(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.max_keepdim(2)?.to_vec3::<u32>()?,
        &[[[4], [9]], [[7], [8]]]
    );
    assert_eq!(
        tensor.max_keepdim(0)?.to_vec3::<u32>()?,
        &[[[3, 1, 7], [8, 5, 9]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.max_keepdim(0)?.to_vec1::<u32>()?, &[3999]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2::<u32>()?, &[[3998, 3999]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2::<u32>()?, &[[3998, 3999]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .max_keepdim(0)?
                .max_keepdim(2)?
                .max_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[3999]]]
        );
        assert_eq!(
            tensor.max_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [3980, 3981, 3982, 3983],
                [3984, 3985, 3986, 3987],
                [3988, 3989, 3990, 3991],
                [3992, 3993, 3994, 3995],
                [3996, 3997, 3998, 3999]
            ]]
        );
    }
    Ok(())
}

fn argmin(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmin_keepdim(2)?.to_vec3::<u32>()?,
        &[[[1], [0]], [[1], [1]]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0)?.to_vec3::<u32>()?,
        &[[[1, 0, 0], [0, 1, 1]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec1::<u32>()?, &[0]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2::<u32>()?, &[[0, 0]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2::<u32>()?, &[[0, 0]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmin_keepdim(0)?
                .argmin_keepdim(2)?
                .argmin_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmin_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]]
        );
    }
    Ok(())
}

fn argmax(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmax_keepdim(2)?.to_vec3::<u32>()?,
        &[[[2], [2]], [[2], [0]]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0)?.to_vec3::<u32>()?,
        &[[[0, 0, 1], [1, 0, 0]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec1::<u32>()?, &[3799]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2::<u32>()?, &[[1899, 1899]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2::<u32>()?, &[[1899, 1899]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmax_keepdim(0)?
                .argmax_keepdim(2)?
                .argmax_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmax_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
            ]]
        );
    }
    Ok(())
}

fn narrow(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.narrow(2, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 4.0], [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0, 9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1)?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1)?.to_vec3::<f32>()?,
        &[[[2.0, 1.0, 7.0], [8.0, 2.0, 8.0]]],
    );
    // The following has been checked against PyTorch via:
    //   import torch
    //   t = torch.tensor([[[3., 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]])
    //   t.transpose(-1, -2).narrow(1, 1, 2)
    assert_eq!(
        tensor.t()?.narrow(1, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}

fn broadcast(device: &Device) -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.broadcast_left((3, 1))?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}

fn slice_set(device: &Device) -> Result<()> {
    let (b, h, max_t, d) = (2, 4, 7, 3);
    let cache = Tensor::zeros((b, h, max_t, d), DType::F32, device)?;
    let tensor = Tensor::randn(0f32, 1f32, (b, h, 4, d), device)?;
    cache.slice_set(&tensor, 2, 0)?;
    let cache_t = cache.narrow(2, 0, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    cache.slice_set(&tensor, 2, 1)?;
    let cache_t = cache.narrow(2, 1, 4)?;
    let diff = (cache_t - &tensor)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let ones = Tensor::ones((b, h, 1, d), DType::F32, device)?;
    cache.slice_set(&ones, 2, 6)?;
    let diff = cache.narrow(2, 5, 1)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let diff = (cache.narrow(2, 6, 1)? - 1.)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    // This used to create a deadlock rather than returning an actual error.
    assert!(cache.slice_set(&cache, 0, 0).is_err());
    Ok(())
}

fn cat(device: &Device) -> Result<()> {
    // 1D
    let t1 = Tensor::new(&[3f32, 1., 4.], device)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], device)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], device)?;
    assert_eq!(Tensor::cat(&[&t1], 0)?.to_vec1::<f32>()?, [3f32, 1., 4.],);
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2, &t3], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );

    // 2D
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, device)?;
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    // PyTorch equivalent:
    //     import torch
    //     t1 = torch.tensor([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]])
    //     t2 = torch.tensor([[5]*5, [2, 7, 1, 8, 2]])
    //     torch.cat([t1.t(), t2.t()], dim=1).t()
    assert_eq!(
        Tensor::cat(&[&t1.t()?, &t2.t()?], 1)?
            .t()?
            .to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 1)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );

    // 3D
    let t1 = Tensor::arange(0, 48i64, device)?.reshape((2, 6, 4))?;
    let t2 = Tensor::arange(100, 124i64, device)?.reshape((2, 3, 4))?;
    let t3 = Tensor::arange(10000, 10032i64, device)?.reshape((2, 4, 4))?;

    let t_cat = Tensor::cat(&[&t1, &t2, &t3], 1)?;

    let t1 = t1.t()?.contiguous()?.t()?;
    let t2 = t2.t()?.contiguous()?.t()?;
    let t3 = t3.t()?.contiguous()?.t()?;
    let t_cat2 = Tensor::cat(&[&t1, &t2, &t3], 1)?;

    let diff = t_cat.eq(&t_cat2)?.to_dtype(DType::F32)?.sum_all()?;
    assert_eq!(diff.to_vec0::<f32>()?, 104.0);
    assert_eq!(t_cat.i((0, 0, 0))?.to_vec0::<i64>()?, 0);
    assert_eq!(t_cat.i((0, 4, 0))?.to_vec0::<i64>()?, 16);
    assert_eq!(t_cat.i((0, 5, 0))?.to_vec0::<i64>()?, 20);
    assert_eq!(t_cat.i((1, 5, 0))?.to_vec0::<i64>()?, 44);
    assert_eq!(t_cat.i((0, 6, 0))?.to_vec0::<i64>()?, 100);
    assert_eq!(t_cat.i((1, 6, 0))?.to_vec0::<i64>()?, 112);
    assert_eq!(t_cat.i((0, 6, 1))?.to_vec0::<i64>()?, 101);
    assert_eq!(t_cat.i((0, 7, 1))?.to_vec0::<i64>()?, 105);
    assert_eq!(t_cat.i((0, 12, 1))?.to_vec0::<i64>()?, 10013);
    assert_eq!(t_cat.i((1, 12, 3))?.to_vec0::<i64>()?, 10031);

    // compare contiguous to uniform blocks
    let (b, s, h, pad) = (3usize, 4, 5, 2);
    let base =
        Tensor::arange(0f32, (b * (s + pad) * h) as f32, device)?.reshape((b, s + pad, h))?;
    let tp1 = base.narrow(1, 0, s)?;
    let tp2 = base.narrow(1, pad, s)?;
    let cat_padded = Tensor::cat(&[&tp1, &tp2], 1)?;
    let cat_cont = Tensor::cat(&[&tp1.contiguous()?, &tp2.contiguous()?], 1)?;
    assert_eq!(
        cat_padded.to_vec3::<f32>()?,
        cat_cont.to_vec3::<f32>()?,
        "cat along dim=1 with padded outer stride should match contiguous result"
    );
    Ok(())
}

fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = t.embedding(&ids)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    let hs = t.index_select(&ids.to_dtype(DType::I64)?, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    let ids = Tensor::new(&[u32::MAX, 2u32, u32::MAX], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 0.0], [4.0, 5.0], [0.0, 0.0]]);
    Ok(())
}

#[test]
fn index_select_fail() -> Result<()> {
    // Check that an error is properly reported on out of bounds.
    let ids = Tensor::new(&[4u32, 2u32, 1u32], &Device::Cpu)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &Device::Cpu)?;
    let hs = t.index_select(&ids, 0);
    assert!(hs.is_err());
    Ok(())
}

// The test below triggers an unwinding panic as there is a panic within the
// #[cfg(feature = "cuda")]
// #[test]
// #[should_panic]
// fn index_select_fail_gpu() {
//     // Check that a panic happens for out of bounds in cuda
//     if let Ok(device) = Device::new_cuda(0) {
//         if let Ok(ids) = Tensor::new(&[4u32, 2u32, 1u32], &device) {
//             if let Ok(t) = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &device) {
//                 let _ = t.index_select(&ids, 0);
//             }
//         }
//     }
// }

fn cmp(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], device)?;
    assert_eq!(t1.eq(&t2)?.to_vec2::<u8>()?, &[[0, 0], [0, 1], [1, 0]]);
    assert_eq!(t1.ne(&t2)?.to_vec2::<u8>()?, &[[1, 1], [1, 0], [0, 1]]);
    assert_eq!(t1.le(&t2)?.to_vec2::<u8>()?, &[[1, 0], [1, 1], [1, 1]]);
    assert_eq!(t1.lt(&t2)?.to_vec2::<u8>()?, &[[1, 0], [1, 0], [0, 1]]);
    assert_eq!(t1.gt(&t2)?.to_vec2::<u8>()?, &[[0, 1], [0, 0], [0, 0]]);
    assert_eq!(t1.ge(&t2)?.to_vec2::<u8>()?, &[[0, 1], [0, 1], [1, 0]]);
    Ok(())
}

fn index_select(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    for dtype in [DType::U8, DType::U32, DType::I64] {
        let ids = ids.to_dtype(dtype)?;
        let hs = t.index_select(&ids, 1)?;
        assert_eq!(
            hs.to_vec2::<f32>()?,
            &[
                [0.0, 2.0, 1.0],
                [3.0, 5.0, 4.0],
                [6.0, 8.0, 7.0],
                [9.0, 11.0, 10.0]
            ]
        );
        let hs = t.index_select(&ids, 0)?;
        assert_eq!(
            hs.to_vec2::<f32>()?,
            &[[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0, 4.0, 5.0]]
        );
        // Prior to https://github.com/huggingface/candle/pull/1022
        // There would be a bug where the last values in the result tensor would be set to 0.
        let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 2u32, 1u32], device)?;
        let hs = t.index_select(&ids, 0)?;
        assert_eq!(
            hs.to_vec2::<f32>()?,
            &[
                [0.0, 1.0, 2.0],
                [6.0, 7.0, 8.0],
                [3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0],
                [6.0, 7.0, 8.0],
                [3.0, 4.0, 5.0],
            ]
        );

        // Test when selecting dim > 0 with ids size different from elem count of
        // target dim in source/input.
        let ids = Tensor::new(&[1u32, 0u32, 1u32], device)?;
        let t = Tensor::arange(1f32, 5f32, device)?.reshape((2, 2))?;
        assert_eq!(t.to_vec2::<f32>()?, &[[1.0, 2.0], [3.0, 4.0]]);
        let hs = t.index_select(&ids, 1)?;
        assert_eq!(hs.to_vec2::<f32>()?, &[[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]);
    }

    Ok(())
}

fn index_add(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[1.0, 4.0], [4.0, 10.0], [7.0, 16.0], [10.0, 22.0]],
    );
    let init = Tensor::zeros((4, 2), DType::F32, device)?;
    let ids = Tensor::new(&[1u32, 0u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[3.0, 0.0], [9.0, 3.0], [15.0, 6.0], [21.0, 9.0]],
    );

    let init = Tensor::zeros((6, 3), DType::F32, device)?;
    let ids = Tensor::new(&[5u32, 0u32, 1u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [12.0, 14.0, 16.0],
            [6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0]
        ]
    );
    Ok(())
}

fn slice_scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let src = Tensor::arange(100f32, 106f32, device)?.reshape((2, 3))?;
    assert_eq!(
        t.slice_scatter0(&src, 0)?.to_vec2::<f32>()?,
        &[
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 1)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 2)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
        ]
    );

    // The source is a view whose storage continues past it, which is what
    // `narrow` yields. Only the two rows the view describes may be written:
    // sizing the copy from the storage instead lets it run to the end of the
    // source's buffer and overwrite the rows after the scatter point.
    let big = Tensor::arange(100f32, 112f32, device)?.reshape((4, 3))?;
    let src_view = big.narrow(0, 0, 2)?;
    assert_eq!(
        t.slice_scatter0(&src_view, 1)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [9.0, 10.0, 11.0]
        ]
    );
    // Same, with the view starting partway into its storage.
    let src_view = big.narrow(0, 2, 1)?;
    assert_eq!(
        t.slice_scatter0(&src_view, 2)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [106.0, 107.0, 108.0],
            [9.0, 10.0, 11.0]
        ]
    );
    Ok(())
}

fn scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [1.0, 2.0, 3.0, 1.0, 1.0],
            [6.0, 1.0, 1.0, 4.0, 5.0],
            [1.0, 9.0, 1.0, 14.0, 1.0],
            [11.0, 1.0, 10.0, 1.0, 12.0]
        ]
    );

    let hs = init.scatter(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0, 1.0, 1.0],
            [5.0, 1.0, 1.0, 3.0, 4.0],
            [1.0, 8.0, 1.0, 7.0, 1.0],
            [10.0, 1.0, 9.0, 1.0, 11.0]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [1.0, 11.0, 6.0],
            [1.0, 2.0, 9.0],
            [10.0, 1.0, 3.0],
            [10.0, 8.0, 1.0],
            [1.0, 5.0, 12.0],
            [1.0, 1.0, 1.0]
        ]
    );
    let hs = init.scatter(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0.0, 10.0, 5.0],
            [1.0, 1.0, 8.0],
            [9.0, 1.0, 2.0],
            [6.0, 7.0, 1.0],
            [1.0, 4.0, 11.0],
            [1.0, 1.0, 1.0]
        ]
    );

    let hs = {
        let ids = Tensor::new(
            &[
                [0u32, u32::MAX, 2],
                [3, 4, u32::MAX],
                [3, 3, 1],
                [u32::MAX, u32::MAX, 4],
            ],
            device,
        )?;
        init.scatter(&ids, &t, 0)?
    };
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 8.0],
            [1.0, 1.0, 2.0],
            [6.0, 7.0, 1.0],
            [1.0, 4.0, 11.0],
            [1.0, 1.0, 1.0]
        ]
    );

    init.scatter_set(&ids, &t, 0)?;
    assert_eq!(
        init.to_vec2::<f32>()?,
        &[
            [0.0, 10.0, 5.0],
            [1.0, 1.0, 8.0],
            [9.0, 1.0, 2.0],
            [6.0, 7.0, 1.0],
            [1.0, 4.0, 11.0],
            [1.0, 1.0, 1.0]
        ]
    );

    Ok(())
}

fn gather(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[0u32], [2u32], [1u32], [0u32]], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let hs = t.gather(&ids, 1)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0], [5.0], [7.0], [9.0]]);
    let ids = Tensor::new(
        &[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[0.0, 0.0], [5.0, 3.0], [7.0, 7.0], [9.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 2u32, 0u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 7.0, 2.0]]);
    let ids = Tensor::new(&[[0u32, 2u32, 0u32], [0u32, 1u32, 1u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 7.0, 2.0], [0.0, 4.0, 5.0]]);

    let hs = {
        let ids = Tensor::new(
            &[
                [0u32, 0u32],
                [2u32, u32::MAX],
                [u32::MAX, 1u32],
                [0u32, 2u32],
            ],
            device,
        )?;
        t.gather(&ids, 1)?
    };
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[0.0, 0.0], [5.0, 0.0], [0.0, 7.0], [9.0, 11.0]]
    );

    // Random data

    // Dim: 0
    let t = Tensor::new(
        &[
            [
                [108_f32, -47., 16., -56., -83., -130., 210.],
                [253., 95., 151., 228., -210., -123., -127.],
                [-9., -217., 2., -78., 163., 245., -204.],
                [-246., 79., -238., 88., -226., -184., 171.],
                [8., -48., -153., 234., -34., 166., -153.],
                [124., 0., -10., -61., -242., -15., -238.],
            ],
            [
                [12., -64., -199., 244., -240., 156., -128.],
                [173., -57., 4., -198., 233., -110., 238.],
                [95., 82., 0., 240., 53., -211., 209.],
                [-122., 167., -212., 227., -144., 61., 118.],
                [-63., -146., 200., 244., 168., -167., 116.],
                [-125., -147., 110., -253., -178., -250., -18.],
            ],
            [
                [57., 86., -50., 56., 92., 205., -78.],
                [-137., -156., -18., 248., -61., -239., 14.],
                [-248., -30., -50., -70., -251., 250., -83.],
                [-221., 67., 72., 59., -24., -154., 232.],
                [-144., -23., -74., 5., 93., 171., 205.],
                [46., -77., -38., -226., 246., 161., -17.],
            ],
            [
                [-153., -231., -236., 161., 126., 2., -22.],
                [-229., -41., 209., 164., 234., 160., 57.],
                [223., 254., -186., -162., -46., -160., -102.],
                [65., 30., 213., -253., 59., 224., -154.],
                [-82., -203., -177., 17., 31., -256., -246.],
                [176., -135., -65., 54., -56., 210., 76.],
            ],
            [
                [-10., -245., 168., 124., -14., -33., -178.],
                [25., -43., -39., 132., -89., 169., 179.],
                [187., -215., 32., -133., 87., -7., -168.],
                [-224., -215., -5., -230., -58., -162., 128.],
                [158., -137., -122., -100., -202., -83., 136.],
                [30., -185., -144., 250., 209., -40., 127.],
            ],
            [
                [-196., 108., -245., 122., 146., -228., 62.],
                [-1., -66., 160., 137., 13., -172., -21.],
                [244., 199., -164., 28., 119., -175., 198.],
                [-62., 253., -162., 195., -95., -230., -211.],
                [123., -72., -26., -107., -139., 64., 245.],
                [11., -126., -182., 108., -12., 184., -127.],
            ],
            [
                [-159., 126., 176., 161., 73., -111., -138.],
                [-187., 214., -217., -33., -223., -201., -212.],
                [-61., -120., -166., -172., -95., 53., 196.],
                [-33., 86., 134., -152., 154., -53., 74.],
                [186., -28., -154., -174., 141., -109., 217.],
                [82., 35., 252., 145., 181., 74., -87.],
            ],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [
                [6_u32, 6, 4, 3, 4, 4, 6],
                [3, 3, 2, 4, 4, 4, 6],
                [3, 3, 0, 2, 4, 6, 4],
                [2, 5, 1, 2, 6, 6, 1],
                [2, 1, 6, 5, 3, 2, 3],
                [6, 1, 0, 1, 0, 2, 6],
            ],
            [
                [4, 6, 4, 3, 3, 3, 2],
                [4, 3, 2, 4, 4, 4, 6],
                [2, 3, 0, 2, 4, 6, 4],
                [6, 5, 1, 2, 6, 6, 1],
                [4, 1, 6, 5, 3, 2, 3],
                [1, 1, 0, 1, 0, 2, 6],
            ],
            [
                [3, 6, 4, 3, 3, 3, 2],
                [2, 3, 2, 4, 4, 4, 6],
                [4, 3, 0, 2, 4, 6, 4],
                [0, 5, 1, 2, 6, 6, 1],
                [6, 1, 6, 5, 3, 2, 3],
                [4, 1, 0, 1, 0, 2, 6],
            ],
            [
                [0, 6, 4, 3, 3, 3, 2],
                [5, 3, 2, 4, 4, 4, 6],
                [0, 3, 0, 2, 4, 6, 4],
                [3, 5, 1, 2, 6, 6, 1],
                [0, 1, 6, 5, 3, 2, 3],
                [3, 1, 0, 1, 0, 2, 6],
            ],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 0)?;
    assert_eq!(
        hs.to_vec3::<f32>()?,
        &[
            [
                [-159_f32, 126., 168., 161., -14., -33., -138.],
                [-229., -41., -18., 132., -89., 169., -212.],
                [223., 254., 2., -70., 87., 53., -168.],
                [-221., 253., -212., 59., 154., -53., 118.],
                [-144., -146., -154., -107., 31., 171., -246.],
                [82., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [-10., 126., 168., 161., 126., 2., -78.],
                [25., -41., -18., 132., -89., 169., -212.],
                [-248., 254., 2., -70., 87., 53., -168.],
                [-33., 253., -212., 59., 154., -53., 118.],
                [158., -146., -154., -107., 31., 171., -246.],
                [-125., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [-153., 126., 168., 161., 126., 2., -78.],
                [-137., -41., -18., 132., -89., 169., -212.],
                [187., 254., 2., -70., 87., 53., -168.],
                [-246., 253., -212., 59., 154., -53., 118.],
                [186., -146., -154., -107., 31., 171., -246.],
                [30., -147., -10., -253., -242., 161., -87.]
            ],
            [
                [108., 126., 168., 161., 126., 2., -78.],
                [-1., -41., -18., 132., -89., 169., -212.],
                [-9., 254., 2., -70., 87., 53., -168.],
                [65., 253., -212., 59., 154., -53., 118.],
                [8., -146., -154., -107., 31., 171., -246.],
                [176., -147., -10., -253., -242., 161., -87.]
            ]
        ]
    );

    // Dim: 1
    let t = Tensor::new(
        &[
            [
                [-117_f32, -175., 69., -163.],
                [200., 242., -21., -67.],
                [179., 150., -126., -75.],
                [-118., 38., -138., -13.],
                [-221., 136., -185., 180.],
                [58., 182., -204., -149.],
            ],
            [
                [3., -148., -58., -154.],
                [-43., 45., -108., 4.],
                [-69., -249., -71., -21.],
                [80., 110., -152., -235.],
                [-88., 7., 92., -250.],
                [-186., 207., -242., 98.],
            ],
            [
                [238., 19., 64., -242.],
                [-150., -97., 218., 58.],
                [111., -233., 204., -212.],
                [-242., -232., 83., 42.],
                [153., 62., -251., 219.],
                [-117., 36., -119., 10.],
            ],
            [
                [215., 159., -169., -27.],
                [-83., 101., -88., 169.],
                [-205., 93., 225., -64.],
                [-162., 240., 214., 23.],
                [-112., 6., 21., 245.],
                [-38., 113., 93., 215.],
            ],
            [
                [91., -188., -148., 101.],
                [74., 203., -35., 55.],
                [-116., -130., -153., -96.],
                [58., 22., -45., -194.],
                [-221., -134., 73., 159.],
                [-203., -254., 31., 235.],
            ],
            [
                [105., -53., 61., 186.],
                [-195., 234., 75., -1.],
                [51., 139., 160., -108.],
                [-173., -167., 161., 19.],
                [83., -246., 156., -222.],
                [109., 39., -149., 137.],
            ],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [[4_u32, 4, 4, 2]],
            [[0, 4, 4, 3]],
            [[1, 5, 3, 4]],
            [[0, 3, 3, 2]],
            [[1, 1, 5, 2]],
            [[1, 4, 5, 4]],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec3::<f32>()?,
        &[
            [[-221., 136., -185., -75.]],
            [[3., 7., 92., -235.]],
            [[-150., 36., 83., 219.]],
            [[215., 240., 214., -64.]],
            [[74., 203., 31., -96.]],
            [[-195., -246., -149., -222.]]
        ]
    );

    // Dim: 2
    let t = Tensor::new(
        &[
            [[-162_f32, 202.], [-126., -39.], [35., -65.], [1., 80.]],
            [[37., 248.], [-191., 89.], [117., -40.], [-217., 220.]],
        ],
        device,
    )?;

    let ids = Tensor::new(&[[[1_u32], [0], [1], [1]], [[0], [1], [0], [1]]], device)?;

    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3::<f32>()?,
        &[
            [[202.], [-126.], [-65.], [80.]],
            [[37.], [89.], [117.], [220.]]
        ]
    );

    let t = Tensor::new(
        &[
            [[-21_f32, -197.], [194., 122.]],
            [[255., -106.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[-130., 238.], [-217., -92.]],
        ],
        device,
    )?;

    let ids = Tensor::new(
        &[
            [[0_u32, 1], [1, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
        ],
        device,
    )?;

    let hs = t.gather(&ids, 2)?;
    assert_eq!(
        hs.to_vec3::<f32>()?,
        &[
            [[-21., -197.], [122., 194.]],
            [[-106., 255.], [-191., 250.]],
            [[33., -117.], [43., 10.]],
            [[238., -130.], [-92., -217.]]
        ]
    );

    Ok(())
}

fn broadcasting(device: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 24f32, device)?.reshape((4, 2, 3))?;
    let t2 = Tensor::new(&[100f32, 200f32], device)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[100.0, 101.0, 102.0], [203.0, 204.0, 205.0]],
            [[106.0, 107.0, 108.0], [209.0, 210.0, 211.0]],
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[100.0, 203.0], [101.0, 204.0], [102.0, 205.0]],
            [[106.0, 209.0], [107.0, 210.0], [108.0, 211.0]],
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-100.0, -99.0, -98.0], [-197.0, -196.0, -195.0]],
            [[-94.0, -93.0, -92.0], [-191.0, -190.0, -189.0]],
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-100.0, -197.0], [-99.0, -196.0], [-98.0, -195.0]],
            [[-94.0, -191.0], [-93.0, -190.0], [-92.0, -189.0]],
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    // Test a narrowed version as this uses a layout start_offset.
    let t1 = t1.i(2..)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    let t3 = Tensor::new(1f32, device)?.broadcast_div(&t2)?;
    let s = t1.broadcast_mul(&t2.reshape((2, 1))?)?;
    let s_div = t1.broadcast_div(&t3.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[1200.0, 1300.0, 1400.0], [3000.0, 3200.0, 3400.0]],
            [[1800.0, 1900.0, 2000.0], [4200.0, 4400.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3::<f32>()?, s_div.to_vec3::<f32>()?,);
    let s = t1.t()?.broadcast_mul(&t2)?;
    let s_div = t1.t()?.broadcast_div(&t3)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[1200.0, 3000.0], [1300.0, 3200.0], [1400.0, 3400.0]],
            [[1800.0, 4200.0], [1900.0, 4400.0], [2000.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3::<f32>()?, s_div.to_vec3::<f32>()?,);
    Ok(())
}

fn randn(device: &Device) -> Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    let tensor2 = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_ne!(tensor.to_vec2::<f32>()?, tensor2.to_vec2::<f32>()?);
    let tensor = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    let tensor2 = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_ne!(tensor.to_vec2::<f32>()?, tensor2.to_vec2::<f32>()?);
    // We do not expect deterministic elements at any index.
    // There once was a bug that had a deterministic zero element in evenly sized tensors.
    const N: usize = 2;
    let v = (0..100)
        .map(|_| Tensor::randn(0f32, 1f32, N, device).and_then(|t| t.to_vec1::<f32>()))
        .collect::<Result<Vec<_>>>()?;
    assert!(
        (0..N).all(|i| v.windows(2).any(|pair| pair[0][i] != pair[1][i])),
        "There are deterministic values in the randn tensors"
    );
    let v = (0..100)
        .map(|_| Tensor::rand(0f32, 1f32, N, device).and_then(|t| t.to_vec1::<f32>()))
        .collect::<Result<Vec<_>>>()?;
    assert!(
        (0..N).all(|i| v.windows(2).any(|pair| pair[0][i] != pair[1][i])),
        "There are deterministic values in the rand tensors"
    );
    Ok(())
}

fn repeat_with_zero_factor(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[[1u32, 2], [3, 4]], device)?;

    let repeated_rows = tensor.repeat((0, 2))?;
    assert_eq!(repeated_rows.dims(), &[0, 4]);
    assert_eq!(repeated_rows.elem_count(), 0);

    let repeated_columns = tensor.repeat((2, 0))?;
    assert_eq!(repeated_columns.dims(), &[4, 0]);
    assert_eq!(repeated_columns.elem_count(), 0);

    Ok(())
}

fn meshgrid_with_empty_axis(device: &Device) -> Result<()> {
    let empty_axis = Tensor::from_vec(Vec::<f32>::new(), 0, device)?;
    let coordinates = Tensor::new(&[10f32, 20., 30.], device)?;

    let grids = Tensor::meshgrid(&[&empty_axis, &coordinates], false)?;
    assert_eq!(grids.len(), 2);
    for grid in grids {
        assert_eq!(grid.dims(), &[0, 3]);
        assert_eq!(grid.elem_count(), 0);
    }

    Ok(())
}

fn zero_dim(device: &Device) -> Result<()> {
    let t = Tensor::zeros((4, 0, 1), DType::F32, device)?;
    assert_eq!(t.dims3()?, (4, 0, 1));
    let t2 = Tensor::zeros((4, 3, 1), DType::F32, device)?;
    let t_cat = Tensor::cat(&[&t, &t2], 1)?;
    assert_eq!(t_cat.dims3()?, (4, 3, 1));
    let t_cat = Tensor::cat(&[&t, &t], 1)?;
    assert_eq!(t_cat.dims3()?, (4, 0, 1));
    let t_unary = t.sqrt()?;
    assert_eq!(t_unary.dims3()?, (4, 0, 1));
    let t_plus = (&t + 1.)?;
    assert_eq!(t_plus.dims3()?, (4, 0, 1));
    let t_mm = t2.matmul(&t.t()?)?;
    assert_eq!(t_mm.dims3()?, (4, 3, 0));
    let t_mm = t.matmul(&t2.t()?)?;
    assert_eq!(t_mm.dims3()?, (4, 0, 3));
    let t_mm = t.t()?.matmul(&t)?;
    assert_eq!(t_mm.dims3()?, (4, 1, 1));
    Ok(())
}

test_device!(zeros, zeros_cpu, zeros_gpu, zeros_metal);
test_device!(ones, ones_cpu, ones_gpu, ones_metal);
test_device!(full, full_cpu, full_gpu, full_metal);
test_device!(const_set, cs_cpu, cs_gpu, cs_metal);
test_device!(arange, arange_cpu, arange_gpu, arange_metal);
test_device!(add_mul, add_mul_cpu, add_mul_gpu, add_mul_metal);
test_device!(tensor_2d, tensor_2d_cpu, tensor_2d_gpu, tensor_2d_metal);
test_device!(narrow, narrow_cpu, narrow_gpu, narrow_metal);
test_device!(broadcast, broadcast_cpu, broadcast_gpu, broadcast_metal);
test_device!(slice_set, ss_cpu, ss_gpu, ss_metal);
test_device!(cat, cat_cpu, cat_gpu, cat_metal);
test_device!(sum, sum_cpu, sum_gpu, sum_metal);
test_device!(min, min_cpu, min_gpu, min_metal);
test_device!(max, max_cpu, max_gpu, max_metal);
test_device!(argmax, argmax_cpu, argmax_gpu, argmax_metal);
test_device!(argmin, argmin_cpu, argmin_gpu, argmin_metal);
test_device!(transpose, transpose_cpu, transpose_gpu, transpose_metal);
test_device!(unary_op, unary_op_cpu, unary_op_gpu, unary_op_metal);
test_device!(binary_op, binary_op_cpu, binary_op_gpu, binary_op_metal);
test_device!(ternary_op, ternary_op_cpu, ternary_op_gpu, ternary_op_metal);
test_device!(embeddings, embeddings_cpu, embeddings_gpu, embeddings_metal);
test_device!(cmp, cmp_cpu, cmp_gpu, cmp_metal);
test_device!(
    broadcasting,
    broadcasting_cpu,
    broadcasting_gpu,
    broadcasting_metal
);
test_device!(
    index_select,
    index_select_cpu,
    index_select_gpu,
    index_select_metal
);
test_device!(index_add, index_add_cpu, index_add_gpu, index_add_metal);
test_device!(gather, gather_cpu, gather_gpu, gather_metal);
test_device!(scatter, scatter_cpu, scatter_gpu, scatter_metal);
test_device!(
    slice_scatter,
    slice_scatter_cpu,
    slice_scatter_gpu,
    slice_scatter_metal
);
test_device!(randn, randn_cpu, randn_gpu, randn_metal);
test_device!(clamp, clamp_cpu, clamp_gpu, clamp_metal);
test_device!(asort, asort_cpu, asort_gpu, asort_metal);
test_device!(asort_big, asort_big_cpu, asort_big_gpu, asort_big_metal);
test_device!(var, var_cpu, var_gpu, var_metal);
test_device!(
    repeat_with_zero_factor,
    repeat_with_zero_factor_cpu,
    repeat_with_zero_factor_gpu,
    repeat_with_zero_factor_metal
);
test_device!(
    meshgrid_with_empty_axis,
    meshgrid_with_empty_axis_cpu,
    meshgrid_with_empty_axis_gpu,
    meshgrid_with_empty_axis_metal
);
test_device!(zero_dim, zero_dim_cpu, zero_dim_gpu, zero_dim_metal);

fn tensor_send_sync(device: &Device) -> Result<()> {
    let tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], device)?;

    for _ in 0..10 {
        let tensor = tensor.clone();
        std::thread::spawn(move || {
            let new = tensor.add(&tensor).unwrap();
            let result: Vec<f32> = new.to_vec1().unwrap();
            assert_eq!(result, vec![2.0f32, 4.0, 6.0]);
        });
    }
    let result: Vec<f32> = tensor.to_vec1().unwrap();
    assert_eq!(result, vec![1.0f32, 2.0, 3.0]);

    let tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], device)?;
    tensor.device().synchronize().unwrap();

    let new = std::thread::spawn(move || {
        let new = tensor.add(&tensor).unwrap();
        new.device().synchronize().unwrap();
        new
    })
    .join()
    .unwrap();
    let result: Vec<f32> = new.to_vec1().unwrap();
    assert_eq!(result, vec![2.0f32, 4.0, 6.0]);

    Ok(())
}
test_device!(
    tensor_send_sync,
    tensor_send_sync_cpu,
    tensor_send_sync_gpu,
    tensor_send_sync_metal
);

// There was originally a bug on the CPU implementation for randn
// https://github.com/huggingface/candle/issues/381
#[test]
fn randn_hasneg() -> Result<()> {
    let t = Tensor::randn(0f32, 1f32, 200, &Device::Cpu)?.to_vec1::<f32>()?;
    if t.iter().all(|&v| v >= 0.) {
        candle_core::bail!("all values in tensors are non-negative")
    }
    Ok(())
}

#[test]
fn pad_with_same() -> Result<()> {
    let t = Tensor::arange(1f32, 5f32, &Device::Cpu)?.reshape((2, 2))?;
    let t0 = t.pad_with_same(0, 1, 2)?;
    assert_eq!(
        t0.to_vec2::<f32>()?,
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    );
    let t1 = t.pad_with_same(1, 1, 2)?;
    assert_eq!(
        t1.to_vec2::<f32>()?,
        [[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 4.0]]
    );
    Ok(())
}

#[test]
fn i64_abs() -> Result<()> {
    let t = Tensor::new(&[-42i64, 1337], &Device::Cpu)?;
    let t = t.abs()?;
    assert_eq!(t.to_vec1::<i64>()?, [42, 1337]);
    Ok(())
}

#[test]
fn tril_triu_eye() -> Result<()> {
    let t = Tensor::tril2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
    );
    let t = Tensor::triu2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    let t = Tensor::eye(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    Ok(())
}

fn cumsum(device: &Device) -> Result<()> {
    let t = &[3f32, 1., 4., 1., 5.];
    let t = Tensor::new(t, device)?;
    assert_eq!(t.cumsum(0)?.to_vec1::<f32>()?, [3., 4., 8., 9., 14.]);
    let t = t.unsqueeze(1)?;
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [[3.0], [4.0], [8.0], [9.0], [14.0]]
    );
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [[3.0], [1.0], [4.0], [1.0], [5.0]]
    );
    let t = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let t = Tensor::new(t, device)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [[3.0, 4.0, 8.0, 9.0, 14.0], [2.0, 3.0, 10.0, 18.0, 20.0]],
    );
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [[3.0, 1.0, 4.0, 1.0, 5.0], [5.0, 2.0, 11.0, 9.0, 7.0]]
    );
    let t = Tensor::arange(0f32, 2060f32, device)?.reshape((2, 1030))?;
    let c = t.cumsum(1)?;
    assert_eq!(c.i((0, 0))?.to_scalar::<f32>()?, 0.0);
    assert_eq!(c.i((0, 1029))?.to_scalar::<f32>()?, 529935.0);
    assert_eq!(c.i((1, 0))?.to_scalar::<f32>()?, 1030.0);

    let t = Tensor::zeros((2, 0, 4), DType::F32, device)?;
    assert_eq!(t.cumsum(1)?.dims(), &[2, 0, 4]);
    Ok(())
}

test_device!(cumsum, cumsum_cpu, cumsum_gpu, cumsum_metal);

fn cumsum_singleton_axis(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 24., device)?
        .reshape((4, 3, 2))?
        .narrow(1, 1, 1)?;
    let c = t.cumsum(1)?;
    assert_eq!(c.id(), t.id());
    assert_eq!(c.dims(), &[4, 1, 2]);
    assert_eq!(c.stride(), t.stride());
    assert_eq!(
        c.flatten_all()?.to_vec1::<f32>()?,
        [2., 3., 8., 9., 14., 15., 20., 21.]
    );

    let t = Tensor::new(&[-0f32, f32::INFINITY, f32::NAN], device)?.unsqueeze(1)?;
    let c = t.cumsum(D::Minus1)?;
    assert_eq!(c.id(), t.id());
    let values = c.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(values[0].to_bits(), (-0f32).to_bits());
    assert_eq!(values[1], f32::INFINITY);
    assert!(values[2].is_nan());
    assert!(t.cumsum(2).is_err());
    Ok(())
}

test_device!(
    cumsum_singleton_axis,
    cumsum_singleton_axis_cpu,
    cumsum_singleton_axis_cuda,
    cumsum_singleton_axis_metal
);

#[test]
fn cumsum_cpu_low_precision_accumulation() -> Result<()> {
    for (dtype, len) in [
        (DType::BF16, 1024),
        (DType::F16, 4096),
        (DType::F8E4M3, 128),
    ] {
        let input = Tensor::ones((2, len), dtype, &Device::Cpu)?.t()?;
        let actual = input.cumsum(0)?;
        let expected = Tensor::arange(1f32, (len + 1) as f32, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(actual.dims(), &[len, 2]);
        for (row, expected) in actual
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .iter()
            .zip(expected)
        {
            assert_eq!(row, &[expected, expected], "dtype {dtype:?}");
        }
    }
    Ok(())
}

#[test]
fn cumsum_cpu_non_contiguous() -> Result<()> {
    let device = &Device::Cpu;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((3, 4))?;
    let t = t.t()?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [
            [0.0, 4.0, 12.0],
            [1.0, 6.0, 15.0],
            [2.0, 8.0, 18.0],
            [3.0, 10.0, 21.0]
        ]
    );

    let t = Tensor::arange(0f32, 20f32, device)?.reshape((4, 5))?;
    let t = t.narrow(1, 1, 3)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [
            [1.0, 3.0, 6.0],
            [6.0, 13.0, 21.0],
            [11.0, 23.0, 36.0],
            [16.0, 33.0, 51.0]
        ]
    );
    cumsum_additional_layouts(device)
}

fn cumsum_additional_layouts(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 20., device)?.narrow(0, 5, 10)?;
    assert!(t.is_contiguous());
    assert_eq!(
        t.cumsum(0)?.to_vec1::<f32>()?,
        [5., 11., 18., 26., 35., 45., 56., 68., 81., 95.]
    );

    let t = Tensor::new(&[1f32, 2., 3.], device)?.broadcast_as((2, 3))?;
    assert_eq!(t.cumsum(0)?.to_vec2::<f32>()?, [[1., 2., 3.], [2., 4., 6.]]);

    let t = Tensor::arange(0f32, 30., device)?
        .reshape((1, 2, 3, 5))?
        .narrow(3, 1, 3)?;
    let c = t.cumsum(2)?;
    assert_eq!(c.dims(), &[1, 2, 3, 3]);
    assert_eq!(
        c.reshape((2, 3, 3))?.to_vec3::<f32>()?,
        [
            [[1., 2., 3.], [7., 9., 11.], [18., 21., 24.]],
            [[16., 17., 18.], [37., 39., 41.], [63., 66., 69.]]
        ]
    );
    Ok(())
}

#[test]
fn cumsum_cpu_integer_dtype() -> Result<()> {
    let device = &Device::Cpu;
    let t = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], device)?;
    assert_eq!(t.cumsum(1)?.to_vec2::<u32>()?, [[1u32, 3, 6], [4, 9, 15]]);
    let t = Tensor::new(&[-3i64, 1, 4, -1], device)?;
    assert_eq!(t.cumsum(0)?.to_vec1::<i64>()?, [-3, -2, 2, 1]);
    Ok(())
}

fn cumsum_integer_wrapping<T>(device: &Device, pattern: &[T]) -> Result<()>
where
    T: candle_core::WithDType + num_traits::WrappingAdd + std::fmt::Debug,
{
    for rows in [1, 2] {
        for len in [
            2, 3, 31, 32, 33, 1024, 1025, 4095, 4096, 4097, 4128, 4129, 8193, 12289, 12290,
        ] {
            let mut values = vec![T::zero()];
            let mut expected = Vec::with_capacity(rows * len);
            for row in 0..rows {
                let mut sum = T::zero();
                for col in 0..len {
                    let value = if len >= 12289 {
                        // Sparse tiles make the tile totals themselves overflow:
                        // e.g. MAX, 1, -1 for i64, exercising the second pass too.
                        if col % 4096 == 0 {
                            pattern[(col / 4096 + 3 * row) % pattern.len()]
                        } else {
                            T::zero()
                        }
                    } else {
                        pattern[(col + 3 * row) % pattern.len()]
                    };
                    values.push(value);
                    sum = sum.wrapping_add(&value);
                    expected.push(sum);
                }
            }
            // Preserve exact integer bits, including i64 values beyond 2^53.
            let input = Tensor::from_vec(values, 1 + rows * len, device)?
                .narrow(0, 1, rows * len)?
                .reshape((rows, len))?;
            for output in [input.cumsum(1)?, input.t()?.contiguous()?.cumsum(0)?.t()?] {
                assert_eq!(output.dtype(), T::DTYPE);
                assert_eq!(output.dims(), &[rows, len]);
                assert_eq!(
                    output.flatten_all()?.to_vec1::<T>()?,
                    expected,
                    "dtype {:?}, rows {rows}, len {len}",
                    T::DTYPE
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cumsum_integer_wrapping_cpu() -> Result<()> {
    let device = &Device::Cpu;
    cumsum_integer_wrapping(device, &[u8::MAX, 1, u8::MAX, 2, u8::MAX, u8::MAX, 3])?;
    cumsum_integer_wrapping(device, &[u32::MAX, 1, u32::MAX, 2, u32::MAX, u32::MAX, 3])?;
    cumsum_integer_wrapping(
        device,
        &[i16::MAX, 1, -1, -i16::MAX, i16::MIN, -1, 1, i16::MAX],
    )?;
    cumsum_integer_wrapping(
        device,
        &[i32::MAX, 1, -1, -i32::MAX, i32::MIN, -1, 1, i32::MAX],
    )?;
    cumsum_integer_wrapping(
        device,
        &[i64::MAX, 1, -1, -i64::MAX, i64::MIN, -1, 1, i64::MAX],
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_integer_wrapping_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    cumsum_integer_wrapping(&device, &[u32::MAX, 1, u32::MAX, 2, u32::MAX, u32::MAX, 3])?;
    cumsum_integer_wrapping(
        &device,
        &[i64::MAX, 1, -1, -i64::MAX, i64::MIN, -1, 1, i64::MAX],
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_adapts_layouts() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let t = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 6.0, 8.0, 10.0],
            [12.0, 15.0, 18.0, 21.0]
        ]
    );

    let t = t.t()?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [
            [0.0, 4.0, 12.0],
            [1.0, 6.0, 15.0],
            [2.0, 8.0, 18.0],
            [3.0, 10.0, 21.0]
        ]
    );
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [
            [0.0, 4.0, 8.0],
            [1.0, 9.0, 17.0],
            [3.0, 15.0, 27.0],
            [6.0, 22.0, 38.0]
        ]
    );

    let t = Tensor::arange(0f32, 20f32, &device)?.reshape((4, 5))?;
    let t = t.narrow(1, 1, 3)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [
            [1.0, 3.0, 6.0],
            [6.0, 13.0, 21.0],
            [11.0, 23.0, 36.0],
            [16.0, 33.0, 51.0]
        ]
    );
    cumsum_additional_layouts(&device)
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_supported_dtypes() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let t = Tensor::new(&[[1f64, 2., 3.], [4., 5., 6.]], &device)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f64>()?,
        [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
    );

    let t = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &device)?;
    assert_eq!(t.cumsum(1)?.to_vec2::<u32>()?, [[1u32, 3, 6], [4, 9, 15]]);

    let t = Tensor::new(&[[-3i64, 1, 4], [2, -5, 7]], &device)?;
    assert_eq!(t.cumsum(1)?.to_vec2::<i64>()?, [[-3, -2, 2], [2, -3, 4]]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_high_rank_axes() -> Result<()> {
    fn check<T: candle_core::WithDType + std::fmt::Debug>(
        values: Vec<T>,
        shape: &[usize],
        axis: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        let inner: usize = shape[axis + 1..].iter().product();
        let mut expected = values.clone();
        // Independent serial reference in logical row-major order. Integer
        // inputs and prefixes stay integer, including values beyond 2^53.
        for i in 0..expected.len() {
            if (i / inner) % shape[axis] != 0 {
                let previous = expected[i - inner];
                expected[i] += previous;
            }
        }
        let expected = Tensor::from_vec(expected, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_dtype(T::DTYPE)?
            .flatten_all()?
            .to_vec1::<T>()?;
        let input = Tensor::from_vec(values, shape, device)?.to_dtype(dtype)?;
        let output = input.cumsum(axis)?;
        assert_eq!(output.dims(), shape);
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output.to_dtype(T::DTYPE)?.flatten_all()?.to_vec1::<T>()?,
            expected,
            "dtype {dtype:?}, shape {shape:?}, axis {axis}"
        );
        Ok(())
    }

    let device = Device::new_cuda(0)?;
    for dims in [&[3, 5, 7][..], &[2, 3, 5, 7][..]] {
        for axis in 0..dims.len() - 1 {
            for multi_block in [false, true] {
                let mut shape = dims.to_vec();
                if multi_block {
                    shape[axis] = 4097;
                }
                let count = shape.iter().product();
                let inner: usize = shape[axis + 1..].iter().product();
                for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
                    // Dyadic inputs and prefixes are exact in f32, so only the
                    // final low-precision conversion rounds the reference.
                    let values = (0..count)
                        .map(|i| ((i * 13 + 7) % 17) as f64 / 16. - 0.5)
                        .collect();
                    check(values, &shape, axis, dtype, &device)?;
                }
                let values = (0..count)
                    .map(|i| {
                        let first = (i / inner) % shape[axis] == 0;
                        // Prefixes exceed f32's exact integer range, without
                        // overflowing u32 even on the long axis.
                        u32::from(first) * (1 << 24) + (i % 17) as u32
                    })
                    .collect();
                check(values, &shape, axis, DType::U32, &device)?;
                let values = (0..count)
                    .map(|i| {
                        let first = (i / inner) % shape[axis] == 0;
                        // Include negative values and prefixes beyond f64's
                        // exact integer range; never convert these to floats.
                        i64::from(first) * (1 << 54) + (i % 17) as i64 - 8
                    })
                    .collect();
                check(values, &shape, axis, DType::I64, &device)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_unsupported_dtypes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let inputs = [
        Tensor::new(&[1u8, 2, 3], &device)?,
        Tensor::new(&[1i16, 2, 3], &device)?,
        Tensor::new(&[1i32, 2, 3], &device)?,
        Tensor::new(
            &[
                F8E4M3::from_f32(1.),
                F8E4M3::from_f32(2.),
                F8E4M3::from_f32(3.),
            ],
            &device,
        )?,
    ];
    for base in inputs {
        for input in [
            base.clone(),
            base.unsqueeze(0)?.broadcast_as((2, 3))?,
            base.unsqueeze(1)?.broadcast_as((3, 2))?,
        ] {
            for axis in 0..input.rank() {
                let error = input.cumsum(axis).unwrap_err();
                assert!(
                    error.to_string().starts_with(&format!(
                        "unsupported dtype {:?} for op cumsum",
                        input.dtype()
                    )),
                    "{error}"
                );
            }
        }
        // No scan is needed for these shapes, regardless of dtype.
        let empty = base.narrow(0, 0, 0)?;
        assert_eq!(empty.cumsum(0)?.id(), empty.id());
        let singleton = base.narrow(0, 1, 1)?.broadcast_as((2, 1))?;
        assert_eq!(singleton.cumsum(1)?.id(), singleton.id());
        assert!(singleton.cumsum(2).is_err());
        // The CUDA-only restriction must not change CPU dtype support.
        assert_eq!(
            base.to_device(&Device::Cpu)?
                .cumsum(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
            [1., 3., 6.]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_half_precision_basic() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        let input = Tensor::ones((2, 33), dtype, &device)?;
        let output = input.cumsum(1)?;
        assert_eq!(output.dtype(), dtype);
        assert_eq!(
            output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            vec![(1..=33).map(|i| i as f32).collect::<Vec<_>>(); 2]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_cross_stream_handoff() -> Result<()> {
    use candle_core::scalar::Scalar;
    let producer = Device::new_cuda_with_stream(0)?;
    let consumer = Device::new_cuda_with_stream(0)?;
    let producer_stream = producer.as_cuda_device()?.cuda_stream();
    let consumer_stream = consumer.as_cuda_device()?.cuda_stream();
    let mut backlog = producer_stream
        .alloc_zeros::<u8>(32 * 1024 * 1024)
        .map_err(candle_core::Error::wrap)?;
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U32,
        DType::I64,
    ] {
        // Enough work to overlap the consumer copy with an untracked producer.
        // Both paths are required: low-precision multi-block already tracked
        // writes correctly, while all single-block outputs need tracking too.
        for (rows, cols) in [(1024, 4096), (1024, 32768), (1, 32768), (3, 4129)] {
            let input = Tensor::ones((rows, cols), dtype, &producer)?;
            drop(input.cumsum(1)?);
            producer.synchronize()?;
            for iteration in 0..4 {
                let value = (iteration + 2) as f64 / if dtype.is_int() { 1. } else { 1024. };
                let scalar = match dtype {
                    DType::F16 => Scalar::F16(half::f16::from_f64(value)),
                    DType::BF16 => Scalar::BF16(half::bf16::from_f64(value)),
                    DType::F32 => Scalar::F32(value as f32),
                    DType::F64 => Scalar::F64(value),
                    DType::U32 => Scalar::U32(value as u32),
                    DType::I64 => Scalar::I64(value as i64),
                    _ => unreachable!(),
                };
                input.const_set(scalar)?;
                // Only prepare the input, never the output.
                producer.synchronize()?;
                // Keep the producer busy using stream-ordered memory operations,
                // without adding a test kernel or relying on host sleeps.
                for _ in 0..32 {
                    producer_stream
                        .memset_zeros(&mut backlog)
                        .map_err(candle_core::Error::wrap)?;
                }
                let before_scan = producer_stream
                    .record_event(None)
                    .map_err(candle_core::Error::wrap)?;
                let transferred = {
                    let output = input.cumsum(1)?;
                    output.to_device(&consumer)?
                }; // Drop the producer output before the consumer finishes reading it.
                let copied = consumer_stream
                    .record_event(None)
                    .map_err(candle_core::Error::wrap)?;
                copied.synchronize().map_err(candle_core::Error::wrap)?;
                assert!(
                    before_scan.is_complete(),
                    "copy did not wait for its producer"
                );
                // No producer synchronization after cumsum. Consume the copied
                // result on the second stream, including windows in later tiles.
                for start in [0, cols / 2, cols - 33] {
                    let values: Vec<f64> = (start..start + 33)
                        .map(|col| (col + 1) as f64 * value)
                        .collect();
                    let values = Tensor::from_vec(values, 33, &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F64)?
                        .to_vec1::<f64>()?;
                    let mut sum = 0.;
                    let expected: Vec<f64> = values
                        .into_iter()
                        .map(|v| {
                            sum += v;
                            sum
                        })
                        .collect();
                    for row in [0, rows / 2, rows - 1] {
                        let actual = transferred
                            .narrow(0, row, 1)?
                            .narrow(1, start, 33)?
                            .to_dtype(DType::F64)?
                            .cumsum(1)?
                            .flatten_all()?
                            .to_vec1::<f64>()?;
                        assert_eq!(
                            actual, expected,
                            "dtype {dtype:?}, cols {cols}, iteration {iteration}, row {row}, start {start}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_low_precision_async_lifetimes() -> Result<()> {
    // Queue work on two independent streams. Temporary layout copies are
    // dropped immediately and similarly sized allocations encourage reuse.
    let devices = [
        Device::new_cuda_with_stream(0)?,
        Device::new_cuda_with_stream(0)?,
    ];
    for dtype in [DType::F16, DType::BF16] {
        let mut pending = Vec::new();
        for iteration in 0..64 {
            let device = &devices[iteration % 2];
            let len = [4097, 4128, 4129, 8193][iteration % 4];
            let rows = 1 + iteration % 3;
            let value = ((iteration % 7) + 1) as f64 / 64.;
            let output = {
                let input = Tensor::ones((rows, len + 2), dtype, device)?.affine(value, 0.)?;
                let sliced = input.narrow(1, 1, len)?;
                if iteration % 2 == 0 {
                    sliced.cumsum(1)?
                } else {
                    sliced.t()?.contiguous()?.cumsum(0)?.t()?
                }
            };
            // No synchronize/readback until the entire batch is queued.
            for _ in 0..3 {
                std::hint::black_box(
                    Tensor::ones((rows, len + 2), dtype, device)?.affine(-3., 0.)?,
                );
            }
            if iteration % 5 == 0 {
                drop(output); // Also exercise output/scratch retirement.
            } else {
                pending.push((output, rows, len, value));
            }
        }
        for device in &devices {
            device.synchronize()?;
        }
        for (output, rows, len, value) in pending {
            let expected = Tensor::arange(1f64, (len + 1) as f64, &Device::Cpu)?
                .affine(value, 0.)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            assert_eq!(
                output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                vec![expected; rows]
            );
        }
    }
    Ok(())
}

// These dyadic inputs and their f64 prefixes are exact in f32, so final
// low-precision rounding can be checked without a floating-point tolerance.
#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_low_precision_prefixes_and_layouts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut lengths = vec![
        2, 3, 17, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 8191, 8192,
        8193, 12289,
    ];
    lengths.extend(4096..=4129);
    for dtype in [DType::F16, DType::BF16] {
        for &len in &lengths {
            let mut values = vec![0f64];
            let mut expected = Vec::with_capacity(3 * len);
            for row in 0..3 {
                let mut prefix = 0.;
                for col in 0..len {
                    let mut bits = (col as u32) ^ (row as u32 + 1).wrapping_mul(0x9e3779b9);
                    bits ^= bits >> 16;
                    bits = bits.wrapping_mul(0x7feb352d);
                    let value = ((bits % 17) as f64 - 8.) / 16.;
                    values.push(value);
                    prefix += value;
                    expected.push(prefix);
                }
            }
            let expected = Tensor::from_vec(expected, (3, len), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            // Offset one also exercises input alignment and odd row strides.
            let input = Tensor::from_vec(values, 1 + 3 * len, &device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 3 * len)?
                .reshape((3, len))?;
            let actual = input.cumsum(1)?;
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                expected,
                "dtype {dtype:?}, len {len}"
            );
            // A separate GPU reference exercises all three f32 scan stages.
            let reference = input.to_dtype(DType::F32)?.cumsum(1)?.to_dtype(dtype)?;
            assert_eq!(
                actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                reference.to_dtype(DType::F32)?.to_vec2::<f32>()?
            );
            if [33, 4096, 4097, 4128, 4129, 8193].contains(&len) {
                for t in [input.t()?, input.t()?.contiguous()?] {
                    assert_eq!(
                        t.cumsum(0)?.t()?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                        expected
                    );
                }
                let padded = Tensor::cat(&[&input, &input], 1)?;
                let sliced = padded.narrow(1, 0, len)?;
                assert!(!sliced.is_contiguous());
                assert_eq!(
                    sliced.cumsum(1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    expected
                );
                let broadcast = input.unsqueeze(0)?.broadcast_as((2, 3, len))?;
                assert_eq!(
                    broadcast
                        .cumsum(2)?
                        .to_dtype(DType::F32)?
                        .to_vec3::<f32>()?,
                    vec![expected.clone(), expected.clone()]
                );
            }
        }
        // More than 1024 tile totals: bounded prefixes avoid f16 overflow.
        let len = 4096 * 1024 + 33;
        let input = Tensor::ones((1, len), dtype, &device)?.affine(1. / 4096., 0.)?;
        let actual = input
            .cumsum(1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let expected = Tensor::arange(1f32, (len + 1) as f32, &Device::Cpu)?
            .affine(1. / 4096., 0.)?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected, "long axis, dtype {dtype:?}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_low_precision_rounding() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        let first = if dtype == DType::F16 { 2048f32 } else { 256f32 };
        let input = Tensor::new(&[first, 1., 1.], &device)?.to_dtype(dtype)?;
        assert_eq!(
            input.cumsum(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            [first, first, first + 2.]
        );
        for len in [1024, 4096, 4097, 8193] {
            let input = Tensor::ones(len, dtype, &device)?;
            let expected = Tensor::arange(1f32, (len + 1) as f32, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            assert_eq!(
                input.cumsum(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                expected
            );
        }
    }
    // An overflowing output must not contaminate the running f32 state.
    let input = Tensor::new(&[65504f32, 65504., -65504.], &device)?.to_dtype(DType::F16)?;
    assert_eq!(
        input.cumsum(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        [65504., f32::INFINITY, 65504.]
    );
    // Tile-local 131008 exceeds f16's range, but adding the previous tile's
    // -65504 offset makes the final output finite. Do not round the scratch.
    let mut values = vec![0f32; 4099];
    values[0] = -65504.;
    values[4096..].copy_from_slice(&[65504., 65504., -65504.]);
    let actual = Tensor::from_vec(values, 4099, &device)?
        .to_dtype(DType::F16)?
        .cumsum(0)?
        .narrow(0, 4096, 3)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(actual, [0., 65504., 0.]);
    // Finite double-rounding counterexamples in both formats.
    for (dtype, magnitude) in [(DType::F16, 2048f32), (DType::BF16, 256f32)] {
        let mut values = vec![0f32; 4098];
        values[0] = -magnitude;
        values[4096..].copy_from_slice(&[magnitude, 1.]);
        let actual = Tensor::from_vec(values, 4098, &device)?
            .to_dtype(dtype)?
            .cumsum(0)?
            .narrow(0, 4096, 2)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(actual, [0., 1.], "dtype {dtype:?}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_low_precision_special_values() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        for len in [3, 33, 4096, 4097, 4128, 4129, 8193] {
            let mut values = vec![0.125f32; 6 * len];
            values[len - 1] = f32::NAN;
            values[2 * len - 2] = f32::INFINITY;
            values[3 * len - 2] = f32::NEG_INFINITY;
            values[3 * len] = f32::INFINITY;
            values[4 * len - 1] = f32::NEG_INFINITY;
            values[4 * len] = -0.;
            // Smallest positive subnormal in each format.
            let tiny = if dtype == DType::F16 {
                2f32.powi(-24)
            } else {
                2f32.powi(-133)
            };
            values[5 * len..].fill(tiny);
            let input = Tensor::from_vec(values, (6, len), &device)?.to_dtype(dtype)?;
            let actual = input
                .cumsum(1)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let source = input.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            for (row, source) in source.iter().enumerate() {
                let mut prefix = 0f64;
                let expected: Vec<f64> = source
                    .iter()
                    .map(|&x| {
                        prefix += x as f64;
                        prefix
                    })
                    .collect();
                let expected = Tensor::from_vec(expected, len, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                for (col, expected) in expected.into_iter().enumerate() {
                    let actual = actual[row * len + col];
                    assert!(
                        if expected.is_nan() {
                            actual.is_nan()
                        } else {
                            actual == expected
                        },
                        "dtype {dtype:?}, len {len}, row {row}, col {col}: {actual} != {expected}"
                    );
                }
            }
        }
        let empty = Tensor::zeros((2, 0), dtype, &device)?.cumsum(1)?;
        assert_eq!(empty.dims(), &[2, 0]);
        let singleton = Tensor::new(&[-0f32], &device)?.to_dtype(dtype)?.cumsum(0)?;
        assert_eq!(
            singleton
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .to_scalar::<f32>()?
                .to_bits(),
            (-0f32).to_bits()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_multi_block_large_last_dim() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let t = Tensor::ones((2, 4097), DType::F32, &device)?;
    let c = t.cumsum(1)?;
    assert_eq!(c.i((0, 0))?.to_scalar::<f32>()?, 1.0);
    assert_eq!(c.i((0, 4095))?.to_scalar::<f32>()?, 4096.0);
    assert_eq!(c.i((0, 4096))?.to_scalar::<f32>()?, 4097.0);
    assert_eq!(c.i((1, 4096))?.to_scalar::<f32>()?, 4097.0);

    let t = Tensor::ones((2, 4097), DType::F64, &device)?;
    let c = t.cumsum(1)?;
    assert_eq!(c.i((0, 4096))?.to_scalar::<f64>()?, 4097.0);
    assert_eq!(c.i((1, 4096))?.to_scalar::<f64>()?, 4097.0);

    let t = Tensor::arange(0u32, 8194, &device)?.reshape((2, 4097))?;
    let c = t.cumsum(1)?;
    assert_eq!(c.i((0, 4096))?.to_scalar::<u32>()?, 8390656);
    assert_eq!(c.i((1, 4096))?.to_scalar::<u32>()?, 25176065);

    let t = Tensor::ones((2, 4097), DType::I64, &device)?;
    let c = t.cumsum(1)?;
    assert_eq!(c.i((0, 4096))?.to_scalar::<i64>()?, 4097);
    assert_eq!(c.i((1, 4096))?.to_scalar::<i64>()?, 4097);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_more_than_1024_tiles() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for len in [
        4096 * 1024 - 1,
        4096 * 1024,
        4096 * 1024 + 1,
        4096 * 2048 + 17,
    ] {
        for dtype in [DType::F32, DType::F64, DType::U32, DType::I64] {
            let input = Tensor::ones((2, len), dtype, &device)?;
            let output = input.cumsum(1)?;
            assert_eq!(output.dtype(), dtype);
            for row in 0..2 {
                for col in [0, 4095, 4096, 4096 * 1024 - 2, len - 1] {
                    let value = output
                        .i((row, col))?
                        .to_dtype(DType::F64)?
                        .to_scalar::<f64>()?;
                    assert_eq!(
                        value,
                        (col + 1) as f64,
                        "dtype {dtype:?}, length {len}, row {row}, col {col}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires at least 5 GiB of free CUDA memory"]
fn cumsum_cuda_flattened_tile_boundary() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let boundary = 4096 * 65535;
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U32,
        DType::I64,
    ] {
        for len in [
            boundary,
            boundary + 1,
            boundary + 32,
            boundary + 33,
            boundary + 4096,
            boundary + 4097,
        ] {
            let input = Tensor::ones((1, len), dtype, &device)?;
            // Keep low-precision results finite, with exact power-of-two inputs.
            let value = if dtype.is_int() { 1. } else { 1. / 8192. };
            let input = if dtype.is_int() {
                input
            } else {
                input.affine(value, 0.)?
            };
            let output = input.cumsum(1)?;
            assert_eq!(output.dims(), &[1, len]);
            assert_eq!(output.dtype(), dtype);
            for col in [0, 4095, 4096, boundary - 1, len - 1] {
                // Materialize only the scalar view. F64 needs an identity
                // affine operation to avoid downloading its full backing storage.
                let sample = output.i((0, col))?;
                let sample = if dtype == DType::F64 {
                    sample.affine(1., 0.)?
                } else {
                    sample
                };
                let actual = sample.to_dtype(DType::F64)?.to_scalar::<f64>()?;
                let expected = if dtype.is_int() {
                    (col + 1) as f64
                } else {
                    Tensor::new((col + 1) as f64 * value, &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F64)?
                        .to_scalar::<f64>()?
                };
                assert_eq!(actual, expected, "dtype {dtype:?}, len {len}, col {col}");
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_two_tile_dispatch_layouts() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    let device = Device::new_cuda(0)?;
    let sms = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)? as usize;
    for rows in [2, 3, (sms / 2).max(2), (sms / 2 + 1).max(2)] {
        for len in [4097, 4128, 4129, 8192] {
            for dtype in [
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
                DType::U32,
                DType::I64,
            ] {
                let mut values = vec![0f64];
                let mut expected = Vec::with_capacity(rows * len);
                for row in 0..rows {
                    let mut prefix = 0.;
                    for col in 0..len {
                        let value = ((row * 7 + col * 13) % 17) as f64
                            - if dtype == DType::U32 { 0. } else { 8. };
                        let value = value / if dtype.is_int() { 1. } else { 8. };
                        values.extend([17., value]);
                        prefix += value;
                        expected.push(prefix);
                    }
                }
                let input = Tensor::from_vec(values, 1 + rows * len * 2, &device)?
                    .to_dtype(dtype)?
                    .narrow(0, 1, rows * len * 2)?
                    .reshape((rows, len, 2))?
                    .narrow(2, 1, 1)?
                    .squeeze(2)?;
                let expected = Tensor::from_vec(expected, rows * len, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F64)?
                    .to_vec1::<f64>()?;
                for output in [input.cumsum(1)?, input.t()?.contiguous()?.cumsum(0)?.t()?] {
                    assert_eq!(output.dtype(), dtype);
                    assert_eq!(output.dims(), &[rows, len]);
                    assert_eq!(
                        output
                            .flatten_all()?
                            .to_dtype(DType::F64)?
                            .to_vec1::<f64>()?,
                        expected,
                        "dtype {dtype:?}, rows {rows}, len {len}, SMs {sms}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_single_row_layouts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U32,
        DType::I64,
    ] {
        for len in [4097, 4127, 4128, 4129, 8192, 8193, 12289] {
            let mut values = vec![0f64];
            let mut expected = Vec::with_capacity(len);
            let mut sum = 0.;
            for col in 0..len {
                let value = if dtype == DType::U32 {
                    (col % 9) as f64
                } else {
                    ((col % 9) as f64 - 4.) / if dtype.is_int() { 1. } else { 8. }
                };
                values.extend([17., value]);
                sum += value;
                expected.push(sum);
            }
            // A strided row with a nonzero storage offset exercises the
            // layout copy before dispatching to the single-row tile entries.
            let input = Tensor::from_vec(values, 1 + 2 * len, &device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 2 * len)?
                .reshape((len, 2))?
                .narrow(1, 1, 1)?
                .t()?;
            let expected = Tensor::from_vec(expected, len, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F64)?
                .to_vec1::<f64>()?;
            for output in [
                input.cumsum(1)?,
                input.contiguous()?.cumsum(1)?,
                input.t()?.unsqueeze(2)?.cumsum(0)?,
            ] {
                assert_eq!(output.dtype(), dtype);
                assert_eq!(
                    output
                        .flatten_all()?
                        .to_dtype(DType::F64)?
                        .to_vec1::<f64>()?,
                    expected,
                    "dtype {dtype:?}, len {len}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_tile_tails() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for tail in 0..=33 {
        let len = 4096 + tail;
        for dtype in [DType::F32, DType::F64, DType::U32, DType::I64] {
            let mut values = vec![0f64];
            let mut expected = Vec::with_capacity(3 * len);
            for row in 0..3 {
                let mut prefix = 0.;
                for col in 0..len {
                    let mut bits = (col as u32) ^ (row as u32 + 1).wrapping_mul(0x9e3779b9);
                    bits ^= bits >> 16;
                    bits = bits.wrapping_mul(0x7feb352d);
                    bits ^= bits >> 15;
                    let value = (bits % 17) as f64 - if dtype == DType::U32 { 0. } else { 8. };
                    values.push(value);
                    prefix += value;
                    expected.push(prefix);
                }
            }
            // A contiguous view with a nonzero storage offset also misaligns
            // subsequent rows when the row length is not a vector multiple.
            let input = Tensor::from_vec(values, 1 + 3 * len, &device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 3 * len)?
                .reshape((3, len))?;
            let output = input.cumsum(1)?;
            assert_eq!(output.dtype(), dtype);
            assert_eq!(
                output
                    .flatten_all()?
                    .to_dtype(DType::F64)?
                    .to_vec1::<f64>()?,
                expected,
                "dtype {dtype:?}, tail {tail}"
            );
            if tail == 32 {
                let output = input.t()?.contiguous()?.cumsum(0)?.t()?;
                assert_eq!(
                    output
                        .flatten_all()?
                        .to_dtype(DType::F64)?
                        .to_vec1::<f64>()?,
                    expected
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_tile_integer_values() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for len in [4097, 4128, 4129, 8193] {
        let mut unsigned = vec![0u32];
        let mut wide = vec![0i64];
        let mut unsigned_expected = Vec::new();
        let mut wide_expected = Vec::new();
        for row in 0..2 {
            let (mut unsigned_sum, mut wide_sum) = (0u32, 0i64);
            for col in 0..len {
                let u = u32::MAX - ((col + row) % 17) as u32;
                let i = (1i64 << 40) + (col % 13) as i64 - row as i64;
                unsigned.push(u);
                wide.push(i);
                unsigned_sum = unsigned_sum.wrapping_add(u);
                wide_sum += i;
                unsigned_expected.push(unsigned_sum);
                wide_expected.push(wide_sum);
            }
        }
        let input = Tensor::from_vec(unsigned, 1 + 2 * len, &device)?
            .narrow(0, 1, 2 * len)?
            .reshape((2, len))?;
        assert_eq!(
            input.cumsum(1)?.flatten_all()?.to_vec1::<u32>()?,
            unsigned_expected
        );
        let input = Tensor::from_vec(wide, 1 + 2 * len, &device)?
            .narrow(0, 1, 2 * len)?
            .reshape((2, len))?;
        assert_eq!(
            input.cumsum(1)?.flatten_all()?.to_vec1::<i64>()?,
            wide_expected
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_cuda_tile_nonfinite_values() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for len in [4097, 4113, 4128, 4129] {
        for dtype in [DType::F32, DType::F64] {
            let mut input = Vec::new();
            let mut expected = Vec::new();
            for row in 0..6 {
                let mut prefix = 0f64;
                for col in 0..len {
                    let value = match (row, col) {
                        (0, 4096) => f64::NAN,
                        (1, 4096) | (4, 0) | (5, 4096) => f64::INFINITY,
                        (2, 4096) | (4, 4096) | (5, 4112) => f64::NEG_INFINITY,
                        (3, _) => -0.,
                        _ => 0.125,
                    };
                    input.push(value);
                    prefix += value;
                    expected.push(prefix);
                }
            }
            let actual = Tensor::from_vec(input, (6, len), &device)?
                .to_dtype(dtype)?
                .cumsum(1)?
                .flatten_all()?
                .to_dtype(DType::F64)?
                .to_vec1::<f64>()?;
            for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                assert!(
                    actual == expected || (actual.is_nan() && expected.is_nan()),
                    "dtype {dtype:?}, length {len}, index {index}: {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}

fn cumsum_scan_boundaries(device: &Device) -> Result<()> {
    for len in [31, 32, 33, 1023, 1024, 1025, 4096, 4097, 8193, 131073] {
        let mut input = Vec::with_capacity(2 * len);
        let mut expected = Vec::with_capacity(2 * len);
        for row in 0..2 {
            let mut prefix = 0f32;
            for col in 0..len {
                let value = if row == 0 {
                    f32::from(col % 7 == 0 || col % 4096 == 0)
                } else {
                    (col % 11) as f32 - 5.
                };
                input.push(value);
                prefix += value;
                expected.push(prefix);
            }
        }
        let t = Tensor::from_vec(input, (2, len), device)?;
        assert_eq!(
            t.cumsum(1)?.flatten_all()?.to_vec1::<f32>()?,
            expected,
            "row length {len}"
        );
    }

    let input: Vec<f32> = (0..32769).map(|i| (i % 17 + 1) as f32).collect();
    let total: f64 = input.iter().map(|&p| f64::from(p)).sum();
    let probabilities: Vec<f32> = input
        .iter()
        .map(|&p| (f64::from(p) / total) as f32)
        .collect();
    let output = Tensor::new(probabilities.as_slice(), device)?
        .cumsum(0)?
        .to_vec1::<f32>()?;
    let mut expected = 0f64;
    let mut previous = 0f32;
    for (actual, probability) in output.into_iter().zip(probabilities) {
        expected += f64::from(probability);
        assert!(actual >= previous);
        assert!((f64::from(actual) - expected).abs() < 1e-4);
        previous = actual;
    }
    Ok(())
}

#[test]
fn cumsum_scan_boundaries_cpu() -> Result<()> {
    cumsum_scan_boundaries(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn cumsum_scan_boundaries_cuda() -> Result<()> {
    cumsum_scan_boundaries(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires at least 17 GiB of free CUDA memory"]
fn cumsum_cuda_large_tensor_indexing() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (rows, cols) = (65537, 32768);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input = Tensor::ones((rows, cols), dtype, &device)?;
        let output = input.cumsum(1)?;
        // A different dtype materializes just the scalar view. Reading an
        // f32 view directly would download the entire multi-GiB storage.
        for row in [0, rows - 1] {
            for col in [0, 4095, 4096, cols - 1] {
                let expected = Tensor::new((col + 1) as f32, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F64)?
                    .to_scalar::<f64>()?;
                assert_eq!(
                    output
                        .i((row, col))?
                        .to_dtype(DType::F64)?
                        .to_scalar::<f64>()?,
                    expected,
                    "dtype {dtype:?}, row {row}, col {col}"
                );
            }
        }
    }
    Ok(())
}

/// A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
/// Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
fn assert_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
    let a_vec: Vec<f64> = a.to_vec1()?;
    let b_vec: Vec<f64> = b.to_vec1()?;

    assert_eq!(a_vec.len(), b_vec.len());
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        assert!((a - b).abs() < epsilon);
    }
    Ok(())
}

#[test]
fn log_sum_exp() -> Result<()> {
    let input = Tensor::new(
        &[
            [[1f64, 2., 3.], [4., 5., 6.]],
            [[-1000.0, -999.0, -1001.0], [1000.0, 999.0, 1001.0]],
        ],
        &Device::Cpu,
    )?;

    let output = input.log_sum_exp(D::Minus1)?;
    // The expectations obtained from pytorch.
    let expected = Tensor::new(&[[3.4076, 6.4076], [-998.5924, 1001.4076]], &Device::Cpu)?;
    assert_eq!(output.dims(), expected.dims());
    assert_close(&output.flatten_all()?, &expected.flatten_all()?, 0.00001)?;

    assert_eq!(
        input.log_sum_exp((0, 1))?.to_vec1::<f64>()?,
        [1000.0, 999.0, 1001.0]
    );
    assert_eq!(
        input.log_sum_exp(())?.to_vec3::<f64>()?,
        input.to_vec3::<f64>()?
    );

    Ok(())
}

#[test]
fn pow() -> Result<()> {
    let lhs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let rhs = (&lhs - 2.)?;
    let res = lhs.pow(&rhs)?;
    assert_eq!(
        test_utils::to_vec2_round(&res, 3)?,
        [[1.0, 1.0, 3.0], [16.0, 125.0, 1296.0]]
    );
    Ok(())
}

#[test]
fn test_flip_1d() -> Result<()> {
    // 1D: [0, 1, 2, 3, 4]
    let t = Tensor::arange(0.0, 5.0, &Device::Cpu)?.reshape((5,))?;
    let flipped = t.flip(&[0])?;
    // Expected: [4, 3, 2, 1, 0]
    let expected = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0, 0.0], (5,), &Device::Cpu)?;
    candle_core::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}

#[test]
fn test_flip_2d() -> Result<()> {
    // 2D:
    // [[0, 1, 2],
    //  [3, 4, 5]]
    let t = Tensor::arange(0.0, 6.0, &Device::Cpu)?.reshape((2, 3))?;
    let flipped = t.flip(&[0, 1])?;
    // Expected:
    // [[5, 4, 3],
    //  [2, 1, 0]]
    let expected = Tensor::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0], (2, 3), &Device::Cpu)?;
    candle_core::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}

#[test]
fn test_flip_3d_channels() -> Result<()> {
    // 3D:
    // [[[0,1,2],
    //   [3,4,5]],
    //
    //  [[6,7,8],
    //   [9,10,11]]]
    let t = Tensor::arange(0.0, 12.0, &Device::Cpu)?.reshape((2, 2, 3))?;
    let flipped = t.flip(&[2])?;
    // Expected:
    // [[[2,1,0],
    //   [5,4,3]],
    //
    //  [[8,7,6],
    //   [11,10,9]]]
    let expected = Tensor::from_vec(
        vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 8.0, 7.0, 6.0, 11.0, 10.0, 9.0],
        (2, 2, 3),
        &Device::Cpu,
    )?;
    candle_core::test_utils::assert_tensor_eq(&flipped, &expected)?;
    Ok(())
}

#[test]
fn tensor_from_data_validates_concrete_shape() -> Result<()> {
    let values = [1f32, 2., 3., 4.];

    assert!(Tensor::from_slice(&values, (2, 2, 2), &Device::Cpu).is_err());
    assert!(Tensor::from_slice(&values, (2, 1), &Device::Cpu).is_err());
    assert!(Tensor::from_vec(values.to_vec(), (2, 2, 2), &Device::Cpu).is_err());
    assert!(Tensor::from_vec(values.to_vec(), (2, 1), &Device::Cpu).is_err());

    let tensor = Tensor::from_slice(&values, (2, 2), &Device::Cpu)?;
    assert_eq!(tensor.dims(), &[2, 2]);

    let tensor = Tensor::from_vec(values.to_vec(), ((), 2), &Device::Cpu)?;
    assert_eq!(tensor.dims(), &[2, 2]);

    let empty: &[f32] = &[];
    let tensor = Tensor::from_slice(empty, (0, 3), &Device::Cpu)?;
    assert_eq!(tensor.dims(), &[0, 3]);

    Ok(())
}

#[test]
fn tensor_new() -> Result<()> {
    let t1 = Tensor::new(vec![1f32, 2.0, 3.0], &Device::Cpu)?;
    assert_eq!(t1.to_vec1::<f32>()?, [1.0, 2.0, 3.0]);
    let t2 = Tensor::new(vec![vec![1f32, 2., 3.], vec![4., 5., 6.]], &Device::Cpu)?;
    assert_eq!(t2.to_vec2::<f32>()?, [[1., 2., 3.], [4., 5., 6.]]);
    let t3 = Tensor::new(
        vec![
            vec![vec![1f32, 2., 3.], vec![4., 5., 6.]],
            vec![vec![3f32, 1., 4.], vec![1., 5., 9.]],
        ],
        &Device::Cpu,
    )?;
    assert_eq!(
        t3.to_vec3::<f32>()?,
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]
        ]
    );
    Ok(())
}

#[test]
fn tensor_norm() -> Result<()> {
    let t = Tensor::new(&[[3., 4.], [0., 0.]], &Device::Cpu)?;
    let norm = t.norm()?;
    assert_eq!(norm.to_scalar::<f64>()?, 5.);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn transfers_cuda_to_device() -> Result<()> {
    use rand::seq::SliceRandom;

    let devices = cudarc::driver::safe::CudaContext::device_count()
        .map_err(candle_core::cuda::CudaError::from)?;
    if devices < 2 {
        return Ok(());
    }
    let first = Device::new_cuda(0)?;

    let mut data: Vec<u32> = (0..262144).collect();
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let t1 = Tensor::from_vec(data, (512, 512), &first)?;
    let second = Device::new_cuda(1)?;
    let t2 = t1.to_device(&second)?;

    assert_ne!(
        t1.device().as_cuda_device()?.id(),
        t2.device().as_cuda_device()?.id()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn allocates_twice_when_transferring_to_same_device() -> Result<()> {
    use std::ops::Deref;

    use candle_core::{Storage, StorageRef};
    use rand::seq::SliceRandom;

    let first = Device::new_cuda(0)?;
    let second = Device::new_cuda(0)?;

    let mut data: Vec<u32> = (0..262144).collect();
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let t1 = Tensor::from_vec(data, (512, 512), &first)?;
    let t2 = t1.to_device(&second)?;

    let (storage1, _) = t1.storage_and_layout();
    let (storage2, _) = t2.storage_and_layout();
    let extract = |s: StorageRef| match &s.deref() {
        Storage::Cuda(c) => {
            use cudarc::driver::DevicePtr;
            let slice = c.as_cuda_slice::<u32>().unwrap();
            let ptr = slice.device_ptr(slice.stream()).0;
            ptr
        }
        _ => unimplemented!(),
    };
    let id1 = extract(storage1);
    let id2 = extract(storage2);
    assert_ne!(id1, id2);
    Ok(())
}
