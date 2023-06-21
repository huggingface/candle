use candle::{DType, Device, Result, Tensor};

#[test]
fn zeros() -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, Device::Cpu)?;
    let (dim1, dim2) = tensor.shape().r2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}

#[test]
fn add_mul() -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], Device::Cpu)?;
    let dim1 = tensor.shape().r1()?;
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

#[test]
fn tensor_2d() -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, Device::Cpu)?;
    let dims = tensor.shape().r2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content, data);
    Ok(())
}

#[test]
fn binary_op() -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, Device::Cpu)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, Device::Cpu)?;
    let tensor = (&tensor + (&tensor * &tensor)? / (&tensor + &tensor2))?;
    let dims = tensor.shape().r2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);
    Ok(())
}
