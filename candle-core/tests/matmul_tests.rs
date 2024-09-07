use candle_core::{test_device, test_utils, DType, Device, IndexOp, Result, Tensor, D};

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

fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}

fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            let sum_diff2 = sum_diff2.to_vec0::<f32>()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2 < 1e-6)
        }
    }
    Ok(())
}

// https://github.com/huggingface/candle/issues/1948
fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), &[1, 32]);
    Ok(())
}

// https://github.com/huggingface/candle/issues/1992
fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], start_offset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal, matmul_wgpu);
test_device!(
    matmul_bf16,
    matmul_bf16_cpu,
    matmul_bf16_gpu,
    matmul_bf16_metal,
    matmul_bf16_wgpu
);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal,
    broadcast_matmul_wgpu
);
test_device!(squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal,squeeze_mm_wgpu);
test_device!(mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal, mm_layout_wgpu);



fn test_functions(device: &Device, fm : impl Fn(usize) -> usize){
    let sizes = vec![1usize, 8, 32, 128, 130, 256,258, 1024,1026, 2048, 2050].into_iter();
   
    for size in sizes{
        println!("size: {size}");
        test_matmul(device, 1, fm(size), size, size);
    }
}


fn test_matmul(device: &Device, b : usize, m : usize, n : usize, k : usize){
    let dtype = DType::F32;
   
    let lhs = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, k, n), dtype, device).unwrap();

    lhs.matmul(&rhs).unwrap();

}

#[cfg(feature="wgpu")]
#[test]
fn test_matmul_kernels_wgpu2()-> Result<()> {
    let device = &Device::new_wgpu_sync(0)?;

    println!("matmul_m_1");
    test_functions(device, |_| 1);

    println!("matmul_full");
    test_functions(device, |size| size);
    
    println!("matmul_(24x1544)");
    test_matmul(device, 1, 24, 6144, 1536);

    println!("matmul_(32x2320)");
    test_matmul(device, 1, 32, 5120, 2304);

    println!("matmul_2*(653x1536)");
    test_matmul(device, 2, 653, 1536, 1536);

    println!("matmul_48*(24x6144) ");
    test_matmul(device, 48, 24, 6144, 1536);

    println!("matmul_32*(32x2304 * 2304x5120)");
    test_matmul(device, 32, 32, 5120, 2304);

    Ok(())
}


#[cfg(feature="wgpu")]
#[test]
//test different wgpu matmul shaders
fn test_matmul_kernels_wgpu()-> Result<()> {
    use candle_core::wgpu::MatmulAlgorithm;
    let mut combinations = Vec::new();
    
    for a in &[false] { //prefatch
        for b in &[false] { //nopadding
            combinations.push((*a, *b));
        }
    }
    
    let algs = vec![
        MatmulAlgorithm::Matmul1_64B(false, false),
        MatmulAlgorithm::Matmul7,
        MatmulAlgorithm::Matmul1,
        MatmulAlgorithm::MatmulX,
        MatmulAlgorithm::Matmul16_16,
    ];

    let algs = algs.into_iter()
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul32_32(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul128_128(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul64_64(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul64_64_8_8(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul64_128(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul64_128_8_8(*a, *b)))
    // .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul16_64(*a, *b)))
    .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul1_64B(*a, *b)))
    .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul1_64(*a, *b)))
    .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul1_128(*a, *b)))
    //.chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul1_256(*a, *b)))
    .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul24_24(*a, *b)))
    .chain(combinations.iter().map(|(a, b)| MatmulAlgorithm::Matmul24_48(*a, *b)));

    let device = Device::new_wgpu_sync(0)?;

    if let Device::Wgpu(wgpu) = &device{
        for alg in algs{
            println!("Testing: {:?}", alg);
            (*wgpu.matmul_alg.lock().unwrap()) = alg;

            big_matmul_wgpu(&device, false, true)?; //transpose b
            big_matmul_wgpu(&device, false, false)?; 
           
            big_matmul_wgpu(&device, true, false)?; //transpoe a
            big_matmul_wgpu(&device, true, true)?; //transpose a and b


            matmul(&device)?;
            broadcast_matmul(&device)?;
            squeeze_mm(&device)?;
            mm_layout(&device)?;
        }
    }

    Ok(())
}


//compares wgpu matmul impl, with cpu impl
#[cfg(feature="wgpu")]
fn big_matmul_wgpu(device: &Device, tpa : bool, tpb : bool)-> Result<()> {
    let b = 1;
    let m = 1;
    let n = 1;
    let k = 128;

    let lhs;
    if tpa{
        lhs = Tensor::arange(0.0f32, (b*m*k) as f32,&Device::Cpu)?.reshape((b, k, m))?.transpose(D::Minus1, D::Minus2)?;
    }
    else{
        lhs = Tensor::arange(0.0f32, (b*m*k) as f32,&Device::Cpu)?.reshape((b, m, k))?;
    }

    let rhs;
    if tpb{
        rhs = Tensor::arange(0.0f32, (b*k*n) as f32,&Device::Cpu)?.reshape((b, n, k))?.transpose(D::Minus1, D::Minus2)?;
    }
    else{
        rhs = Tensor::arange(0.0f32, (b*k*n) as f32,&Device::Cpu)?.reshape((b, k, n))?;
    }

    let t1 = lhs.matmul(&rhs)?.reshape((b,m,n))?;

    let lhs = lhs.to_device(&device)?;
    let rhs = rhs.to_device(&device)?;
    let t2 = lhs.matmul(&rhs)?.reshape((b,m,n))?;

    let m = test_utils::to_vec3_round(&t1, 3)?;
    let m2 = test_utils::to_vec3_round(&t2, 3)?;

    assert_eq!(m, m2);
    Ok(())
}

#[cfg(feature="wgpu")]
#[test]
fn big_matmul_wgpu_tests()-> Result<()> {
    let device = Device::new_wgpu_sync(0)?;
   
    if let Device::Wgpu(wgpu) = &device{
        (*wgpu.matmul_alg.lock().unwrap()) = candle_core::wgpu::MatmulAlgorithm::Matmul32_32(false, false);
    }
   
    big_matmul_wgpu(&device, false, false)?;
    Ok(())
}
