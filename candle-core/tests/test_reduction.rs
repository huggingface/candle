use candle_core::{Device, Tensor};
use candle_core::Result;

#[test]
fn test_reduction() -> Result<()> {
    let dev = &pollster::block_on(Device::new_webgpu(0))?; 

    let inputa = Tensor::arange(0.0f32, 24.0, dev)?.reshape((2,3,4))?;
    let inputb = Tensor::arange(0.0f32, 24.0, &Device::Cpu)?.reshape((2,3,4))?;

    let r1a = inputa.sum(0)?; println!("r1a {r1a}");
    let r1b = inputb.sum(0)?; println!("r1b {r1b}");
    let r2a = inputa.sum(1)?; println!("r2a {r2a}");
    let r2b = inputb.sum(1)?; println!("r2b {r2b}");
    let r3a = inputa.sum(2)?; println!("r3a {r3a}");
    let r3b = inputb.sum(2)?; println!("r3b {r3b}");
    let r4a = inputa.sum((1,2))?;  println!("r4a {r4a}");
    let r4b = inputb.sum((1,2))?; println!("r4b {r4b}");
    let r5a = inputa.sum((0,2))?; println!("r5a {r5a}");
    let r5b = inputb.sum((0,2))?; println!("r5b {r5b}");
    
    assert_eq!(r1a.to_vec2::<f32>()?, r1b.to_vec2::<f32>()?);
    assert_eq!(r2a.to_vec2::<f32>()?, r2b.to_vec2::<f32>()?);
    assert_eq!(r3a.to_vec2::<f32>()?, r3b.to_vec2::<f32>()?);
    assert_eq!(r4a.to_vec1::<f32>()?, r4b.to_vec1::<f32>()?);
    assert_eq!(r5a.to_vec1::<f32>()?, r5b.to_vec1::<f32>()?);

    //test transpose:
    let inputa = inputa.transpose(0, 1)?;
    let inputb = inputb.transpose(0, 1)?;

    let r1a = inputa.sum(0)?; println!("r1a {r1a}");
    let r1b = inputb.sum(0)?; println!("r1b {r1b}");
    let r2a = inputa.sum(1)?; println!("r2a {r2a}");
    let r2b = inputb.sum(1)?; println!("r2b {r2b}");
    let r3a = inputa.sum(2)?; println!("r3a {r3a}");
    let r3b = inputb.sum(2)?; println!("r3b {r3b}");
    let r4a = inputa.sum((1,2))?;  println!("r4a {r4a}");
    let r4b = inputb.sum((1,2))?; println!("r4b {r4b}");
    let r5a = inputa.sum((0,2))?; println!("r5a {r5a}");
    let r5b = inputb.sum((0,2))?; println!("r5b {r5b}");
    
    assert_eq!(r1a.to_vec2::<f32>()?, r1b.to_vec2::<f32>()?);
    assert_eq!(r2a.to_vec2::<f32>()?, r2b.to_vec2::<f32>()?);
    assert_eq!(r3a.to_vec2::<f32>()?, r3b.to_vec2::<f32>()?);
    assert_eq!(r4a.to_vec1::<f32>()?, r4b.to_vec1::<f32>()?);
    assert_eq!(r5a.to_vec1::<f32>()?, r5b.to_vec1::<f32>()?);
    Ok(())
}


#[test]
fn test_conv() -> Result<()> {
    let dev = &pollster::block_on(Device::new_webgpu(0))?; 
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        dev,
    )?;
    let w = Tensor::new(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 5, 5))?;
    let w = w.reshape((2, 4, 3, 3))?;
    let w = w.transpose(0, 1)?;

    let tcpu = t.to_device(&Device::Cpu)?;
    let wcpu = w.to_device(&Device::Cpu)?;
    
    
    //let wcpu = wcpu.transpose(0, 1)?;

    let r1 = t.conv_transpose2d(&w, 0, 0, 1, 1)?;
    let rcpu1 = tcpu.conv_transpose2d(&wcpu, 0, 0, 1, 1)?;

    println!("r: {r1}");
    println!("rcpu: {rcpu1}");
    Ok(())
}