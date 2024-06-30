use candle_core::{test_device, DType, Device, Tensor, D};
use anyhow::Result;

fn convert(device: &Device) -> Result<()> {


    let vf32 = Tensor::arange(0f32, 4f32, device)?;

    let vf32_u32 : Vec<u32> = vf32.to_dtype(candle_core::DType::U32)?.to_vec1()?;
    assert_eq!(vf32_u32, [0u32, 1u32, 2u32, 3u32]);

    let vu32 = Tensor::new(vf32_u32, device)?;
    let vu32_f32 : Vec<f32> = vu32.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    assert_eq!(vu32_f32, [0f32, 1f32, 2f32, 3f32]);


    let vu32_u8 : Vec<u8> = vu32.to_dtype(candle_core::DType::U8)?.to_vec1()?;
    assert_eq!(vu32_u8, [0, 1, 2, 3]);

    let vf32_u8 : Vec<u8> = vf32.to_dtype(candle_core::DType::U8)?.to_vec1()?;
    assert_eq!(vf32_u8, [0, 1, 2, 3]);

    let vu8 = vu32.to_dtype(candle_core::DType::U8)?;
    let vu8_f32 : Vec<f32> = vu8.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    assert_eq!(vu8_f32, [0f32, 1f32, 2f32, 3f32]);

    Ok(())
}

fn alloc(device: &Device) -> Result<()> {
    let t = 5.0f64;
    let ratio = (Tensor::ones(1, candle_core::DType::F32, &device)? * t)?;
    assert_eq!(ratio.to_vec1::<f32>()?, [5f32]);

    let ratio = (Tensor::ones(1, candle_core::DType::U32, &device)? * t)?;

    assert_eq!(ratio.to_vec1::<u32>()?, [5u32]);

    Ok(())
}

fn sum2(device: &Device) -> Result<()> {
    //[[1, 256, 256, 1], f32, wgpu:0]

    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.reshape((1, 2, 2, 3))?;
    assert_eq!(
        tensor.sum_keepdim(3)?.reshape((2, 2, 1))?.to_vec3::<f32>()?,
        &[[[8f32], [15f32]], [[10f32], [18f32]]]
    );
    assert_eq!(
        tensor.sum_keepdim(D::Minus1)?.reshape((2, 2, 1))?.to_vec3::<f32>()?,
        &[[[8f32], [15f32]], [[10f32], [18f32]]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.reshape((1, 2, 3))?.to_vec3::<f32>()?,
        &[[[5f32, 2f32, 11f32], [9f32, 7f32, 17f32]]],
    );
    Ok(())
}


fn sum3(device: &Device) -> Result<()> {
    //[[1, 256, 256, 1], f32, wgpu:0]

    //let rs : usize = 379;
    //let a : usize = 60;  
    //let b : usize = 2;

    let rs : usize = 384;
    
    //let a : usize = 256;  
    //let b : usize = 256;

    let a : usize = 256;  
    let b : usize = 256;

    let data_cpu = Tensor::ones((1,b,a,rs), DType::U32, &Device::Cpu)?;
    let data_cpu = data_cpu.reshape((1, b, a, rs))?;
    let data = data_cpu.to_device(device)?;

    let result1 = data.sum_keepdim(D::Minus1)?;
    let result2 = data_cpu.sum_keepdim(D::Minus1)?;

    let result1 = result1.reshape((a, b))?;
    let result2 = result2.reshape((a, b))?;
   
    let m = result1.to_vec2::<u32>()?;
    let m2 = result2.to_vec2::<u32>()?;

    if m != m2{
         panic!("m != m2")
    }
    //assert_eq!(m, m2);
    Ok(())
}



test_device!(convert, convert_cpu, convert_gpu, convert_metal, convert_webgpu);
test_device!(alloc, alloc_cpu, alloc_gpu, alloc_metal, alloc_webgpu);
test_device!(sum2, sum2_cpu, sum2_gpu, sum2_metal, sum2_webgpu);
test_device!(sum3, sum3_cpu, sum3_gpu, sum3_metal, sum3_webgpu);