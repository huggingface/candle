use sgemm::{GenericMatmulSettings, StrideOptimization};

use crate::wgpu_backend::MatmulAlgorithm;

use super::*;
pub struct SGEMMParams{
    pub b : u32, 
    pub m : u32, 
    pub k : u32, 
    pub n : u32
}

impl SGEMMParams {
    pub fn new<T : ToU32>(b: T, m: T, k: T, n: T) -> Self {
        Self { b : b.to_u32(), m : m.to_u32(), k : k.to_u32(), n : n.to_u32()}
    }
}

mod sgemm{   

    use crate::{Layout, Shape};
    use super::*;

    #[cfg(feature="wgpu_debug")]
    fn get_debug_string(params : &SGEMMParams) -> String
    {
        let b = params.b;
        let m = params.m;
        let n = params.n;
        let k = params.k;
        let use_batch = b != 1;
        if use_batch{
            format!("Batched: {b}*({m}x{k} * {k}x{n})")
        }
        else{
            format!("({m}x{k} * {k}x{n})")
        }
    }
    
    pub fn queue_matmul_buffer1(
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        params : SGEMMParams,
        layout_input1: &Layout,
        layout_input2: &Layout,
        _dtype: crate::DType,
        pipeline : Pipelines,
        is_16bytes_aligned : bool
    ) -> crate::Result<()> {
        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();
    
        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);
    
        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        let input2_stride_b = *input2_stride.next().unwrap_or(&1);
    
        let const_vec = vec![(input1_stride_k == 1) as usize, (input1_stride_m == 1 )as usize, (input2_stride_n == 1) as usize, (input2_stride_k == 1) as usize, (params.b != 1) as usize];
    
        let mut meta = get_meta(&dev);
        meta.add(params.b);
        meta.add(params.m);
        meta.add(params.k);
        meta.add(params.n);
    
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1.start_offset()); //input1_offset
    
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2.start_offset()); //input2_offset
    
        meta.add(input1_stride_k);
        meta.add(input1_stride_m);
        meta.add(input2_stride_n);
        meta.add(input2_stride_k);

        let pipeline = meta.get_pipeline_const(pipeline, const_vec.clone());

        let input_alignment : BindgroupAlignment = _dtype.into();
        let bind_group = if input_alignment == BindgroupAlignment::Aligned4 && is_16bytes_aligned{
            create_bind_group_input2_with_alignment(
                buffer_dest,
                buffer_input1,
                buffer_input2,
                BindgroupAlignmentLayout::Bindgroup2(BindgroupAlignment::Aligned4, BindgroupAlignment::Aligned16, BindgroupAlignment::Aligned16)
            )
        }
        else{
            create_bind_group_input2_with_alignment(
                buffer_dest,
                buffer_input1,
                buffer_input2,
                BindgroupAlignmentLayout::Bindgroup2(BindgroupAlignment::Aligned4, BindgroupAlignment::Aligned4, BindgroupAlignment::Aligned4)
            )
        };

        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            (params.n  + 15) / 16,
            (params.m  + 15) / 16,
            params.b,
            params.k as usize * params.m as usize* params.n as usize ,
            #[cfg(feature="wgpu_debug")]
            Some(get_debug_string(&params))
        );
        return Ok(());
    }

    fn round_to_next_divisible(num: u32, n: u32) -> u32 {
        if n == 0 {
            panic!("Divisor cannot be zero");
        }
        (num + n - 1) / n * n
    }    
    
    pub(crate) enum StrideOptimization {
        None, //no stride preferred
        StrideK(bool), //if true stride must be 1, if false stride is preferred to 1
        StrideNM(bool), //if true stride must be 1, if false stride is preferred to 1
    }

    pub(crate)  struct GenericMatmulSettings{
        pub m_tile : u32,
        pub n_tile : u32,
        pub k_tile : u32,

        pub input1_stride : StrideOptimization,
        pub input2_stride : StrideOptimization,

        pub needs_padding : bool //wheter this shader input matrices must be padded if it is not divisible by tile size
    }
    
    impl GenericMatmulSettings {
        pub(crate) fn new(m_tile: u32, n_tile: u32, k_tile: u32, input1_stride: StrideOptimization, input2_stride: StrideOptimization) -> Self {
            Self { m_tile, n_tile, k_tile, input1_stride, input2_stride, needs_padding : true }
        }

        pub(crate) fn new_nopadding(m_tile: u32, n_tile: u32, k_tile: u32, input1_stride: StrideOptimization, input2_stride: StrideOptimization) -> Self {
            Self { m_tile, n_tile, k_tile, input1_stride, input2_stride, needs_padding : false }
        }
    }

    pub fn queue_matmul_generic( 
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        params : SGEMMParams,
        layout_input1: &Layout,
        layout_input2: &Layout,
        dtype: crate::DType,
        settings : GenericMatmulSettings,
        pipeline : Pipelines
    ) -> crate::Result<()> {

        let m_tile = settings.m_tile;
        let n_tile = settings.n_tile;
        let k_tile = settings.k_tile;

        const NON_PADDED : bool = false;

        let new_m;
        let new_n;
        let new_k;

        if NON_PADDED{
            new_m = params.m;
            new_n = params.n;
            new_k = params.k;
        }
        else{
            new_m = round_to_next_divisible(params.m, m_tile);
            new_n = round_to_next_divisible(params.n, n_tile);
            new_k = round_to_next_divisible(params.k, k_tile);
        }

        const USE_DIFFERENT_PADDED_OUTPUT : bool = true;

        let need_different_output_buffer = params.m != new_m || params.n != new_n;
    
        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();
    
        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
    
        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);

        assert!(k_tile % 4 == 0);

        let no_padding_needed_input1 =  ((params.m % m_tile == 0 && params.k % k_tile == 0) || NON_PADDED)
            && layout_input1.start_offset() % 4 == 0 //input will be loaded 16 bytes aligned
            && (input1_stride_k == 1 || input1_stride_m == 1) && 
            !(input1_stride_k != 1 && matches!(settings.input1_stride, StrideOptimization::StrideK(true) )) &&
            !(input1_stride_m != 1 && matches!(settings.input1_stride, StrideOptimization::StrideNM(true) ));

        let no_padding_needed_input2 = ((params.n % n_tile == 0 && params.k % k_tile == 0) || NON_PADDED) 
            && layout_input2.start_offset() % 4 == 0 //input will be loaded 16 bytes aligned
            && (input2_stride_n == 1 || input2_stride_k == 1) &&
            !(input2_stride_k != 1 && matches!(settings.input2_stride, StrideOptimization::StrideK(true) )) &&
            !(input2_stride_n != 1 && matches!(settings.input2_stride, StrideOptimization::StrideNM(true) ));


        let (buffer_input1_padded, layout_input1_padded) = 
            if no_padding_needed_input1{
                (buffer_input1, layout_input1.clone())
            }
            else{
                let mut cache = dev.cache.lock().unwrap();
                let buffer_input1_padded = cache.create_buffer_reference(params.b * (new_m * new_k) * dtype.size_in_bytes() as u32, false);

                let dest_layout;

                if let StrideOptimization::StrideNM(_) = settings.input1_stride{
                    dest_layout = crate::Layout::new(Shape::from((params.b as usize, new_m as usize, new_k as usize)), vec![(new_m * new_k) as usize, 1, new_m as usize], 0);
                }
                else if let StrideOptimization::StrideK(_) = settings.input1_stride{
                    dest_layout = crate::Layout::contiguous(&Shape::from((params.b as usize, new_m as usize, new_k as usize)));
                }
                else if input1_stride_m == 1 {
                    dest_layout = crate::Layout::new(Shape::from((params.b as usize, new_m as usize, new_k as usize)), vec![(new_m * new_k) as usize, 1, new_m as usize], 0);
                }
                else{
                    dest_layout = crate::Layout::contiguous(&Shape::from((params.b as usize, new_m as usize, new_k as usize)));
                }
             
                super::queue_copy3d_padded(dev, buffer_input1_padded.clone(), buffer_input1, dtype, layout_input1, (params.b, params.m, params.k), &dest_layout)?;
                //let res : Vec<f32> = block_on(read_data_from_gpu_async(dev, buffer_input1_padded.clone()));
                //println!("pad1: {:?}", res);
                (buffer_input1_padded, dest_layout)
            };
    
        let (buffer_input2_padded, layout_input2_padded) = 
            if no_padding_needed_input2{
                (buffer_input2, layout_input2.clone())
            }
            else{
                let mut cache = dev.cache.lock().unwrap();
                let buffer_input2_padded = cache.create_buffer_reference(params.b * (new_k * new_n) * dtype.size_in_bytes() as u32, false);
                
                let dest_layout;

                if let StrideOptimization::StrideNM(_) = settings.input2_stride{
                    dest_layout = crate::Layout::contiguous(Shape::from((params.b as usize, new_k as usize, new_n as usize)));
                }
                else if let StrideOptimization::StrideK(_) = settings.input2_stride{
                    dest_layout = crate::Layout::new(Shape::from((params.b as usize, new_k as usize, new_n as usize)), vec![(new_n * new_k) as usize, 1, new_k as usize], 0);
                }
                else if input2_stride_k == 1{
                    dest_layout = crate::Layout::new(Shape::from((params.b as usize, new_k as usize, new_n as usize)), vec![(new_n * new_k) as usize, 1, new_k as usize], 0);
                }
                else{
                    dest_layout = crate::Layout::contiguous(Shape::from((params.b as usize, new_k as usize, new_n as usize)));
                }
                super::queue_copy3d_padded(dev, buffer_input2_padded.clone(), buffer_input2, dtype, layout_input2, (params.b, params.k, params.n),&dest_layout)?;
                //let res : Vec<f32> = block_on(read_data_from_gpu_async(dev, buffer_input2_padded.clone()));
                //println!("pad2: {:?}", res);
                (buffer_input2_padded, dest_layout)
            };
        


        let buffer_dest_padded = if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT{
            let mut cache = dev.cache.lock().unwrap();
            cache.create_buffer_reference(params.b * (new_m * new_n) * dtype.size_in_bytes() as u32, false)
        }
        else{
            buffer_dest.clone()
        };
    
        let mut input1_stride = layout_input1_padded.stride().iter().rev();
        let mut input2_stride = layout_input2_padded.stride().iter().rev();
    
        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);
    
        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        let input2_stride_b = *input2_stride.next().unwrap_or(&1);
    
        let use_batch = params.b != 1;
        let const_vec = vec![(input1_stride_k == 1) as usize, (input1_stride_m == 1 )as usize, (input2_stride_n == 1) as usize, (input2_stride_k == 1) as usize, use_batch as usize];
    
        let mut meta = get_meta(&dev);
        meta.add(params.b);
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_m} else {params.m});
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_k} else {params.k});
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_n} else {params.n});
    
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1_padded.start_offset()); //input1_offset
    
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2_padded.start_offset()); //input2_offset

        meta.add(input1_stride_k);
        meta.add(input1_stride_m);
        meta.add(input2_stride_n);
        meta.add(input2_stride_k);

        if need_different_output_buffer && !USE_DIFFERENT_PADDED_OUTPUT{
            meta.add_const(candle_wgpu_kernels::Constants::Isoutputpadded, true);
        }

        let pipeline = meta.get_pipeline_const(pipeline, const_vec.clone());
        let input_alignment : BindgroupAlignment = dtype.into();
        if input_alignment != BindgroupAlignment::Aligned4{
            panic!("matmul can only be performed with f32 and i32");
        }

        let bind_group = create_bind_group_input2_with_alignment(
            buffer_dest_padded.clone(),
            buffer_input1_padded.clone(),
            buffer_input2_padded.clone(),
            BindgroupAlignmentLayout::Bindgroup2(BindgroupAlignment::Aligned4, BindgroupAlignment::Aligned16, BindgroupAlignment::Aligned16)
        );

        let lx;
        let ly;
        if NON_PADDED{
            lx = (new_n + n_tile - 1 ) / n_tile;
            ly = (new_m + m_tile - 1 ) / m_tile;
        }
        else{
            lx = (new_n ) / n_tile;
            ly = (new_m ) / m_tile;
        }

        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            lx,
            ly,
            params.b,
            params.k as usize * params.m as usize * params.n as usize,
            #[cfg(feature="wgpu_debug")]
            Some(get_debug_string(&params))
        );
    
        if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT{
            //let res : Vec<f32> = pollster::block_on(read_data_from_gpu_async(dev, buffer_dest_padded.clone()));
            //println!("res: {:?}", res);
            let dest_padding_layout = crate::Layout::contiguous(&Shape::from((params.b as usize, new_m as usize, new_n as usize)));
            let dest_layout = crate::Layout::contiguous(&Shape::from((params.b as usize, params.m as usize, params.n as usize)));

            super::queue_copy3d(dev, buffer_dest, buffer_dest_padded, dtype, &dest_padding_layout, (params.b, params.m, params.n), &dest_layout)?;
        }
    
        return Ok(());
    }
}



pub fn queue_matmul_buffer(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params : SGEMMParams,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType
) -> crate::Result<()> {
    let alg = dev.matmul_alg.lock().unwrap();
    queue_matmul_buffer_alg(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, dtype, alg.clone())
}

fn get_matmul_setting(alg : &MatmulAlgorithm) -> GenericMatmulSettings{
    return  match alg{
        MatmulAlgorithm::Matmul7 => GenericMatmulSettings::new(16, 16, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul1 => GenericMatmulSettings::new_nopadding(16, 16, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul1_4 => GenericMatmulSettings::new_nopadding(16, 16, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul16_16 => GenericMatmulSettings::new(16, 16, 16, StrideOptimization::None, StrideOptimization::None),
        
        MatmulAlgorithm::Matmul32_64 => GenericMatmulSettings::new(32, 64, 4, StrideOptimization::None, StrideOptimization::None),
        
        MatmulAlgorithm::Matmul32_32(_, _) =>GenericMatmulSettings::new(32, 32, 32, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul64_64(_, _) =>  GenericMatmulSettings::new(64, 64, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul64_64_8_8(_, _) => GenericMatmulSettings::new(64, 64, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul64_64_4_8(_, _) =>  GenericMatmulSettings::new(64, 64, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul16_64(_, _) => GenericMatmulSettings::new(16, 64, 16, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul1_64(_, _) =>  GenericMatmulSettings::new(1, 64, 64, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul1_64B(_, _) => GenericMatmulSettings::new(1, 64, 4, StrideOptimization::StrideK(true), StrideOptimization::StrideK(true)),
        MatmulAlgorithm::Matmul1_128(_, _) => GenericMatmulSettings::new(1, 128, 128, StrideOptimization::StrideNM(true), StrideOptimization::None),
        MatmulAlgorithm::Matmul24_24(_, _) => GenericMatmulSettings::new(24, 24, 24, StrideOptimization::None, StrideOptimization::None),
        MatmulAlgorithm::Matmul24_48(_, _) => GenericMatmulSettings::new(24, 48, 24, StrideOptimization::None, StrideOptimization::None),
        alg => {panic!("alg {alg:?} not supported")}
    };
}

pub fn queue_matmul_buffer_alg(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params : SGEMMParams,
    layout_input1: &Layout,
    layout_input2: &Layout,
    cdtype: crate::DType,
    alg : MatmulAlgorithm
) -> crate::Result<()> {
    let dtype = get_dtype(cdtype)?;
    match alg{
        MatmulAlgorithm::MatmulX => return queue_matmul_buffer_best(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, cdtype),
        MatmulAlgorithm::Matmul1_4 => return super::matmul::sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params,layout_input1, layout_input2, cdtype, Pipelines::Matmul(dtype, matmul::Functions::Matmul116), true),
        MatmulAlgorithm::Matmul7 => return super::matmul::sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params,  layout_input1, layout_input2, cdtype, Pipelines::Matmul(dtype, matmul::Functions::Matmul7), false),
        MatmulAlgorithm::Matmul1 => return super::matmul::sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, cdtype, Pipelines::Matmul(dtype, matmul::Functions::Matmul1), false),
         _ => {}
    }
    use candle_wgpu_kernels::{sgemm, matmul};
    
    let setting = get_matmul_setting(&alg);

    let pipeline = match alg{
        MatmulAlgorithm::Matmul7 => Pipelines::Matmul(dtype, matmul::Functions::Matmul7),
        MatmulAlgorithm::Matmul1 => Pipelines::Matmul(dtype, matmul::Functions::Matmul1),
        MatmulAlgorithm::Matmul1_4 => Pipelines::Matmul(dtype, matmul::Functions::Matmul116),
        MatmulAlgorithm::Matmul16_16 => Pipelines::Matmul16x16(dtype, sgemm::matmul16x16::Functions::Matmul),
        
        MatmulAlgorithm::Matmul32_64 => Pipelines::Matmul32x64(dtype, sgemm::matmul32x64::Functions::Matmul),
        
        MatmulAlgorithm::Matmul32_32(_, _) => Pipelines::Matmul32x32(dtype, sgemm::matmul32x32::Functions::Matmul),
        MatmulAlgorithm::Matmul64_64(_, _) =>  Pipelines::Matmul64x64(dtype, sgemm::matmul64x64::Functions::Matmul),
        MatmulAlgorithm::Matmul64_64_8_8(_, _) => Pipelines::Matmul64x648x8(dtype, sgemm::matmul64x64_8x8::Functions::Matmul),
        MatmulAlgorithm::Matmul64_64_4_8(_, _) => Pipelines::Matmul64x644x8(dtype, sgemm::matmul64x64_4x8::Functions::Matmul),
        MatmulAlgorithm::Matmul16_64(_, _) => Pipelines::Matmul16x64(dtype, sgemm::matmul16x64::Functions::Matmul),
        MatmulAlgorithm::Matmul1_64(_, _) =>  Pipelines::Matmul1x64(dtype, sgemm::matmul1x64::Functions::Matmul),
        MatmulAlgorithm::Matmul1_64B(_, _) => Pipelines::Matmul1x64b(dtype, sgemm::matmul1x64b::Functions::Matmul),
        MatmulAlgorithm::Matmul1_128(_, _) => Pipelines::Matmul1x128(dtype, sgemm::matmul1x128::Functions::Matmul),
        MatmulAlgorithm::Matmul24_24(_, _) => Pipelines::Matmul24x24(dtype, sgemm::matmul24x24::Functions::Matmul),
        MatmulAlgorithm::Matmul24_48(_, _) => Pipelines::Matmul24x48(dtype, sgemm::matmul24x48::Functions::Matmul),
        alg => {panic!("alg {alg:?} not supported")}
    };

    return super::matmul::sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, cdtype,setting,pipeline);
}


fn get_matmul_naive(
    k : usize,
    layout_input1: &Layout,
    layout_input2: &Layout,
) ->  crate::wgpu_backend::MatmulAlgorithm {
    if k % 4 == 0 && layout_input1.start_offset() % 4 == 0 && layout_input2.start_offset() % 4 == 0
    {

        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();
    
        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
    
        let _input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);

        if input1_stride_k == 1 && input2_stride_k == 1{
            return crate::wgpu_backend::MatmulAlgorithm::Matmul1_4;
        }
    }
    return crate::wgpu_backend::MatmulAlgorithm::Matmul1;
}

pub fn queue_matmul_buffer_best(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params : SGEMMParams, 
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let m = params.m as usize;
    let k = params.k as usize;
    let n = params.n as usize;

    // let get_matmul_alg = |size1 : usize, size2 : usize, size3 : usize|{
    //     //we might use kernel 32x64 if the dimensions are big enaugh:
    //     if m >= 64 * 11 && n >= 64 * 11 && k >= 16 && (m >= 64*size1 || n >= 64*size1) {
    //         return MatmulAlgorithm::Matmul32_64;
    //     }
    //     if m >= 48 && n >= 48 && k >= 16 && ( m >= 64*size2 || n >= 64*size2){
    //         return crate::wgpu_backend::device::MatmulAlgorithm::Matmul64_64(false, false);
    //     }
    //     if m >= 16 && n >= 16 && k >= 16 && (m >= 32*size3 || n >= 32*size3){
    //         return crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false);
    //     } 
    //     else{
    //         return crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;
    //     }
    // };

    let alg;


    if m <= 2 || n <= 2{ 

        if m <= 2{
            let mut input2_stride = layout_input2.stride().iter().rev();
            let input2_stride_n = *input2_stride.next().unwrap_or(&1);
            let input2_stride_k = *input2_stride.next().unwrap_or(&1);

            if k % 64 == 0 && n % 128 == 0 && input2_stride_k == 1{
                alg = MatmulAlgorithm::Matmul1_64B(false, false);
            }
            else if k % 64 == 0 && n % 64 == 0 && input2_stride_n == 1{
                alg = MatmulAlgorithm::Matmul1_64(false, false);
            }
            else{
                alg = get_matmul_naive(k, layout_input1, layout_input2);
            }
        }   
        else{
            alg = get_matmul_naive(k, layout_input1, layout_input2);
        }

        //use 1x128 shader for small m:
        // if k >= 128 && n >= 128{ //m <= 2:
            
        //     if input2_stride[1] == 1{
        //         alg = MatmulAlgorithm::Matmul1_64B(false, false);
        //     }
        //     else{
        //         alg = MatmulAlgorithm::Matmul1_64(false, false);
        //     }
        // }
        // else{
        //     alg = get_matmul_naive(k, layout_input1, layout_input2);
        // }
    }
    else{

        let shaders = [MatmulAlgorithm::Matmul32_64, 
            MatmulAlgorithm::Matmul32_32(false, false), 
            MatmulAlgorithm::Matmul24_48(false, false),  
            MatmulAlgorithm::Matmul24_24(false, false), 
            MatmulAlgorithm::Matmul16_16];

        let mut best_no_padding_25: Option<&MatmulAlgorithm> = None;
        let mut best_wgs_25: Option<&MatmulAlgorithm> = None;

        for a in shaders.iter(){
            let s = get_matmul_setting(a);
            let padding_score = !s.needs_padding || (m % s.m_tile as usize == 0 && k % s.k_tile as usize == 0 && n % s.n_tile as usize == 0);
            
            let new_m = (m as u32+ s.m_tile - 1) / s.m_tile;
            let new_n = (n as u32+ s.n_tile - 1) / s.n_tile;
            let wgs = new_m * new_n;

            if padding_score{
                if wgs > 64{
                    alg = a.clone();
                    return queue_matmul_buffer_alg(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, dtype, alg);
                }
                if wgs >= 25{
                    if best_no_padding_25.is_none() {
                        best_no_padding_25 = Some(a); // Store the first match
                    }
                }
            }
            else if wgs >= 25 {
                if best_wgs_25.is_none() {
                    best_wgs_25 = Some(a); // Store the first match
                }
            }
        }
        if let Some(entry) = best_no_padding_25 {
            alg = entry.clone();
        } else if let Some(entry) = best_wgs_25 {
            alg = entry.clone();
        } else {
            alg = get_matmul_naive(k, layout_input1, layout_input2);
        }


        // if m < 16 && n < 16{
        //     alg = get_matmul_naive(k, layout_input1, layout_input2);
        // }
        // else if m <= 32 && n <= 32{
        //     if m % 16 == 0 && n % 16 == 0 && k % 16 == 0{
        //         alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul16_16;
        //     } 
        //     else{
        //         alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;
        //     }
        // }
        // else{ //we may pad an input matrix
        //     //if a dimension is not divisible by 2, padding may be way faster for bigger matrices:
        //     let need_pad_a = m % 2 == 1 || k % 2 == 1;
        //     let need_pad_b = n % 2 == 1 || k % 2 == 1;

        //     let might_use_64x64 =  ((m % 64 == 0 && k % 64 == 0)|| need_pad_a) && ((n % 64 == 0 && k % 16 == 0) || need_pad_b);
        //     let might_use_32x32 =  ((m % 32 == 0 && k % 32 == 0)|| need_pad_a) && ((n % 32 == 0 && k % 16 == 0) || need_pad_b);

        //     //let might_use_64x64 =  ((m % 64 == 0 && k % 64 == 0)) && ((n % 64 == 0 && k % 16 == 0));
        //     //let might_use_32x32 =  ((m % 32 == 0 && k % 32 == 0)) && ((n % 32 == 0 && k % 16 == 0));

        //     //use 64x64 or 32x32: 
        //     if need_pad_a || need_pad_b{
        //         if might_use_32x32{
        //             if !might_use_64x64{
        //                 alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false);
        //             }
        //             else{
        //                 alg = get_matmul_alg(16, 4, 0);
        //             }
        //         }
        //         else{
        //             if might_use_64x64{
        //                 alg = get_matmul_alg(16, 2, 0);
        //             }
        //             else if might_use_32x32{
        //                 alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false);
        //             }
        //             else{ //only for big sizes for m, n padding is worth;
        //                 alg = get_matmul_alg(64, 32, 1000);
        //             }
        //         }
        //     }
        //     else{
        //         if might_use_64x64{
        //             alg = get_matmul_alg(16, 4, 0);
        //         }
        //         else if might_use_32x32{
        //             alg = MatmulAlgorithm::Matmul32_32(false, false);
        //         }
        //         else{
        //             //use 24x24 shader:
        //             if m % 24 == 0 && n % 24 == 0 && k % 24 == 0{
        //                 alg = MatmulAlgorithm::Matmul24_24(false, false);
        //             }
        //             else{
        //                 //only for big sizes for m, n padding is worth;

        //                 //padding input matrices may be faster, as the 64x64-shader is faster
        //                 //but this needs more memory. 
        //                 //alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;

        //                 //let single_pad_64 =  (m % 64 == 0 || n % 64 == 0) && k % 16 == 0;
        //                 //let single_pad_32 =  (m % 32 == 0 || n % 32 == 0) && k % 16 == 0;

        //                 let new_m = next_divisible_by_n(m, 32);
        //                 let new_n = next_divisible_by_n(n, 32);

        //                 if n >= 128 || m >= 128{
        //                     //padding needed for 32x32, will also make 64x64 usable:
        //                     if new_m % 64 == 0 && new_n % 64 == 0{
        //                         if dev.backend == wgpu::Backend::Vulkan{
        //                             //on native vulkan this was up two twice as fast, but in the browser this was 30 times slower
        //                             alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul64_64_8_8(false, false); 
        //                         }
        //                         else{
        //                             alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul64_64_4_8(false, false);
        //                         }
        //                     }
        //                     else{
        //                         alg = MatmulAlgorithm::Matmul32_32(false, false);
        //                     }
        //                 }
        //                 else{
        //                     alg = MatmulAlgorithm::Matmul7;
        //                 }
        //             }
        //         }
        //     }
        // }
    }
    queue_matmul_buffer_alg(dev, buffer_dest, buffer_input1, buffer_input2, params, layout_input1, layout_input2, dtype, alg)
}
