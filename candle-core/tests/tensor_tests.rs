mod test_utils;
use candle::{DType, Device, IndexOp, Result, Tensor};
use test_utils::to_vec3_round;

fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
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

fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor + (&tensor * &tensor)? / (&tensor + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);
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

fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = tensor.log()?.softmax(0)?;
    let t1 = tensor.log()?.softmax(1)?;
    let t2 = tensor.log()?.softmax(2)?;
    assert_eq!(
        to_vec3_round(t0, 4)?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.3636], [0.1111, 0.7143, 0.5294]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]
        ]
    );
    assert_eq!(
        to_vec3_round(t1, 4)?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.1667, 0.3077], [0.25, 0.8333, 0.6923]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]
        ]
    );
    assert_eq!(
        to_vec3_round(t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
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
    Ok(())
}

fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = Tensor::embedding(&ids, &t)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    Ok(())
}

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
    Ok(())
}

fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3::<f32>()?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
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

test_device!(zeros, zeros_cpu, zeros_gpu);
test_device!(add_mul, add_mul_cpu, add_mul_gpu);
test_device!(tensor_2d, tensor_2d_cpu, tensor_2d_gpu);
test_device!(narrow, narrow_cpu, narrow_gpu);
test_device!(broadcast, broadcast_cpu, broadcast_gpu);
test_device!(cat, cat_cpu, cat_gpu);
test_device!(sum, sum_cpu, sum_gpu);
test_device!(transpose, transpose_cpu, transpose_gpu);
test_device!(binary_op, binary_op_cpu, binary_op_gpu);
test_device!(softmax, softmax_cpu, softmax_gpu);
test_device!(embeddings, embeddings_cpu, embeddings_gpu);
test_device!(cmp, cmp_cpu, cmp_gpu);
test_device!(matmul, matmul_cpu, matmul_gpu);
test_device!(broadcasting, broadcasting_cpu, broadcasting_gpu);
test_device!(index_select, index_select_cpu, index_select_gpu);
