use crate::wgpu_backend::MatmulAlgorithm;

use super::*;
pub struct SGEMMParams{
    b : u32, 
    m : u32, 
    k : u32, 
    n : u32
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
        dest_offset : u32,
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
    
        let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k, (params.b != 1) as usize,  dest_offset as usize];
    
        let mut meta = get_meta(&dev);
        meta.add(params.b);
        meta.add(params.m);
        meta.add(params.k);
        meta.add(params.n);
    
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1.start_offset()); //input1_offset
    
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2.start_offset()); //input2_offset
    
        let pipeline = meta.get_pipeline_const(pipeline, const_vec.clone());

        let bind_group = if is_16bytes_aligned{
            create_bind_group_input2_16(
                buffer_dest,
                buffer_input1,
                buffer_input2,
            )
        }
        else{
            create_bind_group_input2(
                buffer_dest,
                buffer_input1,
                buffer_input2,
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
    
    pub fn queue_matmul_generic( 
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        params : SGEMMParams,
        dest_offset : u32,
        layout_input1: &Layout,
        layout_input2: &Layout,
        dtype: crate::DType,
    
        m_tile : u32,
        n_tile : u32,
        k_tile : u32,
        get_pipeline : impl Fn(candle_wgpu_kernels::DType) -> Pipelines,
        
        load_a : bool,
        load_b : bool,
        non_padded : bool    
    ) -> crate::Result<()> {

        let new_m;
        let new_n;
        let new_k;

        if non_padded{
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
    
        let input2_stride_n = *input2_stride.next().unwrap_or(&1);


        let no_padding_needed_input1 = ((params.m % m_tile == 0 && params.k % k_tile == 0) || non_padded)
            && layout_input1.start_offset() % 4 == 0 //input will be loaded 16 bytes aligned
            && (input1_stride_k == 1 || m_tile % 4 == 0); //if input1_stride_k is not 1, input will be loaded with 16byte allignmen in tile m, but then m_tile must be divisible by 4

        let no_padding_needed_input2 = ((params.n % n_tile == 0 && params.k % k_tile == 0) || non_padded) 
            && layout_input2.start_offset() % 4 == 0 //input will be loaded 16 bytes aligned
            && (input2_stride_n == 1 || k_tile % 4 == 0); //if input1_stride_n is not 1, input will be loaded with 16byte allignmen in tile k, but then k_tile must be divisible by 4


        let (buffer_input1_padded, layout_input1_padded) = 
            if no_padding_needed_input1{
                (buffer_input1, layout_input1.clone())
            }
            else{
                let mut cache = dev.cache.lock().unwrap();
                let buffer_input1_padded = cache.create_buffer_reference(params.b * (new_m * new_k) * 4, false);

                let dest_layout = crate::Layout::contiguous(&Shape::from((params.b as usize, new_m as usize, new_k as usize)));
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
                let buffer_input2_padded = cache.create_buffer_reference(params.b * (new_k * new_n) * 4, false);

                //let dest_layout = crate::Layout::contiguous(&Shape::from((b as usize, new_k as usize, new_n as usize))); 
                let dest_layout = crate::Layout::new(Shape::from((params.b as usize, new_k as usize, new_n as usize)), vec![(new_n * new_k) as usize, 1, new_k as usize], 0);
                super::queue_copy3d_padded(dev, buffer_input2_padded.clone(), buffer_input2, dtype, layout_input2, (params.b, params.k, params.n),&dest_layout)?;
                //let res : Vec<f32> = block_on(read_data_from_gpu_async(dev, buffer_input2_padded.clone()));
                //println!("pad2: {:?}", res);
                (buffer_input2_padded, dest_layout)
            };
        


        let buffer_dest_padded = if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT{
            let mut cache = dev.cache.lock().unwrap();
            cache.create_buffer_reference(params.b * (new_m * new_n) * 4, false)
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

        let padded_dest_offset = if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT{
            0
        }else{
            dest_offset
        };

        let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k, use_batch as usize, padded_dest_offset as usize];
    
        let mut meta = get_meta(&dev);
        meta.add(params.b);
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_m} else {params.m});
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_k} else {params.k});
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {new_n} else {params.n});
    
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1_padded.start_offset()); //input1_offset
    
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2_padded.start_offset()); //input2_offset
        
        if !load_a{
            meta.add_const(candle_wgpu_kernels::Constants::Preloada, false);
        }
        if !load_b{
            meta.add_const(candle_wgpu_kernels::Constants::Preloadb, false);
        }
        if need_different_output_buffer && !USE_DIFFERENT_PADDED_OUTPUT{
            meta.add_const(candle_wgpu_kernels::Constants::Isoutputpadded, true);
        }

        let pipeline = meta.get_pipeline_const(get_pipeline(get_dtype(dtype)?), const_vec.clone());
       
        let bind_group = create_bind_group_input2_16(
            buffer_dest_padded.clone(),
            buffer_input1_padded.clone(),
            buffer_input2_padded.clone(),
        );

        let lx;
        let ly;
        if non_padded{
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
            let dest_layout = crate::Layout::contiguous_with_offset(&Shape::from((params.b as usize, params.m as usize, params.n as usize)), dest_offset as usize);

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
    dest_offset : u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType
) -> crate::Result<()> {
    let alg = dev.matmul_alg.lock().unwrap();
    queue_matmul_buffer_alg(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, alg.clone())
}

pub fn queue_matmul_buffer_alg(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params : SGEMMParams,
    dest_offset : u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
    alg : MatmulAlgorithm
) -> crate::Result<()> {
    match alg{
        crate::wgpu_backend::MatmulAlgorithm::MatmulX => queue_matmul_buffer_best(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype),
       
        crate::wgpu_backend::MatmulAlgorithm::Matmul1_4 => sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, Pipelines::Matmul(get_dtype(dtype)?, candle_wgpu_kernels::matmul::Functions::Matmul116), true),
        crate::wgpu_backend::MatmulAlgorithm::Matmul7 => sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, Pipelines::Matmul(get_dtype(dtype)?, candle_wgpu_kernels::matmul::Functions::Matmul7), false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul1 => sgemm::queue_matmul_buffer1(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, Pipelines::Matmul(get_dtype(dtype)?, candle_wgpu_kernels::matmul::Functions::Matmul1), false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul16_16 => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 16, 16,16, |dtype| Pipelines::Matmul(dtype, candle_wgpu_kernels::matmul::Functions::Matmul5),  true, true, false),
        
        crate::wgpu_backend::MatmulAlgorithm::Matmul32_32(false, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 32, 32,32, |dtype| Pipelines::Matmul32x32(dtype, candle_wgpu_kernels::sgemm::matmul32x32::Functions::Matmul), load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_64(false, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 64,16, |dtype| Pipelines::Matmul64x64(dtype, candle_wgpu_kernels::sgemm::matmul64x64::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul128_128(false, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 128, 128,16, |dtype| Pipelines::Matmul128x128(dtype, candle_wgpu_kernels::sgemm::matmul128x128::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_64_8_8(false, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 64,16, |dtype| Pipelines::Matmul64x648x8(dtype, candle_wgpu_kernels::sgemm::matmul64x64_8x8::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_128(false, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 128,16, |dtype| Pipelines::Matmul64x1284x8(dtype, candle_wgpu_kernels::sgemm::matmul64x128_4x8::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_128_8_8(false, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 128,16, |dtype| Pipelines::Matmul64x1288x8(dtype, candle_wgpu_kernels::sgemm::matmul64x128_8x8::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul16_64(false, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 16, 64,16, |dtype| Pipelines::Matmul16x64(dtype, candle_wgpu_kernels::sgemm::matmul16x64::Functions::Matmul), load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul1_128 (false, false, load_a)=> sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 1, 128,128, |dtype| Pipelines::Matmul1x128(dtype, candle_wgpu_kernels::sgemm::matmul1x128::Functions::Matmul), load_a, false, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul1_256 (false, false, load_a)=> sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 1, 256,256, |dtype| Pipelines::Matmul1x256(dtype, candle_wgpu_kernels::sgemm::matmul1x256::Functions::Matmul), load_a, false, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul24_24(false, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 24, 24,24, |dtype| Pipelines::Matmul24x24(dtype, candle_wgpu_kernels::sgemm::matmul24x24::Functions::Matmul), load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul24_48(false, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 24, 48,24, |dtype| Pipelines::Matmul24x48(dtype, candle_wgpu_kernels::sgemm::matmul24x48::Functions::Matmul), load_a, load_b, false),
        

        crate::wgpu_backend::MatmulAlgorithm::Matmul32_32(true, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 32, 32,32, |dtype| Pipelines::Matmul32x32Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul32x32_prefetch::Functions::Matmul),load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_64(true, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 64,16, |dtype| Pipelines::Matmul64x64Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul64x64_prefetch::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_128(true, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 128,16, |dtype| Pipelines::Matmul64x1284x8Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul64x128_4x8_prefetch::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul128_128(true, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 128, 128,16, |dtype| Pipelines::Matmul128x128Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul128x128_prefetch::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_64_8_8(true, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 64,16, |dtype| Pipelines::Matmul64x648x8Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul64x64_8x8_prefetch::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul64_128_8_8(true, false) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 64, 128,16, |dtype| Pipelines::Matmul64x1288x8Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul64x128_8x8_prefetch::Functions::Matmul), true, true, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul16_64(true, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 16, 64,16, |dtype| Pipelines::Matmul16x64Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul16x64_prefetch::Functions::Matmul), load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul1_128(true, false, load_a) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 1, 128,128, |dtype| Pipelines::Matmul1x128Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul1x128_prefetch::Functions::Matmul), load_a, false, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul1_256(true, false, load_a) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 1, 256,256, |dtype| Pipelines::Matmul1x256Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul1x256_prefetch::Functions::Matmul), load_a, false, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul24_24(true, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 24, 24,24, |dtype| Pipelines::Matmul24x24Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul24x24_prefetch::Functions::Matmul), load_a, load_b, false),
        crate::wgpu_backend::MatmulAlgorithm::Matmul24_48(true, false, load_a, load_b) => sgemm::queue_matmul_generic(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, 24, 48,24, |dtype| Pipelines::Matmul24x48Prefetch(dtype, candle_wgpu_kernels::sgemm::matmul24x48_prefetch::Functions::Matmul), load_a, load_b, false),
         _ => {panic!()}
    }
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
    dest_offset : u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let m = params.m as usize;
    let k = params.k as usize;
    let n = params.n as usize;

    let get_matmul_alg = |size1 : usize, size2 : usize, size3 : usize|{
        //we might use kernel 64x64 if the dimensions are big enaugh:
        if m >= 64 * 11 && n >= 64 * 11 && k >= 16 && (m >= 64*size1 || n >= 64*size1) {
            return crate::wgpu_backend::device::MatmulAlgorithm::Matmul64_64_8_8(false, false);
        }
        if m >= 48 && n >= 48 && k >= 16 && ( m >= 64*size2 || n >= 64*size2){
            return crate::wgpu_backend::device::MatmulAlgorithm::Matmul64_64(false, false);
        }
        if m >= 16 && n >= 16 && k >= 16 && (m >= 32*size3 || n >= 32*size3){
            return crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false,true, true);
        } 
        else{
            return crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;
        }
    };

    let alg;


    if m <= 2 || n <= 2{ 
        alg = get_matmul_naive(k, layout_input1, layout_input2);
    }
    else{
        if m < 16 && n < 16{
            alg = get_matmul_naive(k, layout_input1, layout_input2);
        }
        else if m <= 32 && n <= 32{
            if m % 16 == 0 && n % 16 == 0 && k % 16 == 0{
                alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul16_16;
            } 
            else{
                alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;
            }
        }
        else{ //we may pad an input matrix
            //if a dimension is not divisible by 2, padding may be way faster for bigger matrices:
            let need_pad_a = m % 2 == 1 || k % 2 == 1;
            let need_pad_b = n % 2 == 1 || k % 2 == 1;

            let might_use_64x64 =  ((m % 64 == 0 && k % 64 == 0)|| need_pad_a) && ((n % 64 == 0 && k % 16 == 0) || need_pad_b);
            let might_use_32x32 =  ((m % 32 == 0 && k % 32 == 0)|| need_pad_a) && ((n % 32 == 0 && k % 16 == 0) || need_pad_b);

            //let might_use_64x64 =  ((m % 64 == 0 && k % 64 == 0)) && ((n % 64 == 0 && k % 16 == 0));
            //let might_use_32x32 =  ((m % 32 == 0 && k % 32 == 0)) && ((n % 32 == 0 && k % 16 == 0));

            //use 64x64 or 32x32: 
            if need_pad_a || need_pad_b{
                if might_use_32x32{
                    if !might_use_64x64{
                        alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false,true, true);
                    }
                    else{
                        alg = get_matmul_alg(16, 4, 0);
                    }
                }
                else{
                    if might_use_64x64{
                        alg = get_matmul_alg(16, 2, 0);
                    }
                    else if might_use_32x32{
                        alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul32_32(false, false,true, true);
                    }
                    else{ //only for big sizes for m, n padding is worth;
                        alg = get_matmul_alg(64, 32, 1000);
                    }
                }
            }
            else{
                if might_use_64x64{
                    alg = get_matmul_alg(16, 4, 0);
                }
                else if might_use_32x32{
                    alg = MatmulAlgorithm::Matmul32_32(false, false,true, true);
                }
                else{
                    //use 24x24 shader:
                    if m % 24 == 0 && n % 24 == 0 && k % 24 == 0{
                        alg = MatmulAlgorithm::Matmul24_24(false, false, true, true);
                    }
                    else{
                        //only for big sizes for m, n padding is worth;

                        //padding input matrices may be faster, as the 64x64-shader is faster
                        //but this needs more memory. 
                        //alg = crate::wgpu_backend::device::MatmulAlgorithm::Matmul7;

                        //let single_pad_64 =  (m % 64 == 0 || n % 64 == 0) && k % 16 == 0;
                        //let single_pad_32 =  (m % 32 == 0 || n % 32 == 0) && k % 16 == 0;

                        let new_m = next_divisible_by_n(m, 32);
                        let new_n = next_divisible_by_n(n, 32);

                        if n >= 128 || m >= 128{
                            //padding needed for 32x32, will also make 64x64 usable:
                            if new_m % 64 == 0 && new_n % 64 == 0{
                                alg = MatmulAlgorithm::Matmul64_64_8_8(false, false);
                            }
                            else{
                                alg = MatmulAlgorithm::Matmul32_32(false, false, true, true);
                            }
                        }
                        else{
                            alg = MatmulAlgorithm::Matmul7;
                        }

                    

                        // //only one input needs to be padded for 32x32
                        // if single_pad_32{
                        //     if single_pad_64{
                        //         alg = get_matmul_alg(64, 8, 4);
                        //     }
                        //     else{
                        //         alg = get_matmul_alg(64, 16, 8);
                        //     }
                        // }
                        // else{ //both inputs needs to be padded:
                        //     alg = get_matmul_alg(64, 16, 8);
                        // }
                    }
                }
            }

            // //we might use kernel 32x32
            // else if ((m >= 32 && k >= 16 )|| need_pad_a) && ((n >= 32 && k >= 16 ) || need_pad_b) {
                
            // } 
        }
    }
    queue_matmul_buffer_alg(dev, buffer_dest, buffer_input1, buffer_input2, params, dest_offset, layout_input1, layout_input2, dtype, alg)
}
