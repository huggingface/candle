use sgemm::{GenericMatmulSettings, StrideOptimization};

use crate::wgpu_backend::MatmulAlgorithm;

use super::*;
pub struct SGEMMParams {
    pub b: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl SGEMMParams {
    pub fn new<T: ToU32>(b: T, m: T, k: T, n: T) -> Self {
        Self {
            b: b.to_u32(),
            m: m.to_u32(),
            k: k.to_u32(),
            n: n.to_u32(),
        }
    }
}

mod transpose {
    use super::*;
    use candle_wgpu_kernels::Pipelines;

    pub fn queue_transpose3d_generic(
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input: BufferReferenceId,
        dtype: crate::DType,
        batch: u32,
        width: u32,
        height: u32,
        start_offset: usize,
        batch_stride : usize,
        _debug_info : Option<String>
    ) -> crate::Result<()> {
        let pipeline;
        let tile_w;
        let tile_h;
        if width % 32 == 0 && height % 32 == 0 {
            pipeline = Pipelines::Tranpose3232(
                get_dtype(dtype)?,
                candle_wgpu_kernels::sgemm::tranpose32_32::Functions::TransposeBatched,
            );
            tile_w = 32;
            tile_h = 32;
        } else if width % 24 == 0 && height % 24 == 0 {
            pipeline = Pipelines::Tranpose2424(
                get_dtype(dtype)?,
                candle_wgpu_kernels::sgemm::tranpose24_24::Functions::TransposeBatched,
            );
            tile_w = 24;
            tile_h = 24;
        } else if width % 16 == 0 && height % 16 == 0 {
            pipeline = Pipelines::Tranpose1616(
                get_dtype(dtype)?,
                candle_wgpu_kernels::sgemm::tranpose16_16::Functions::TransposeBatched,
            );
            tile_w = 16;
            tile_h = 16;
        } else {
            return queue_transpose3d(
                dev,
                buffer_dest,
                buffer_input,
                dtype,
                batch,
                width,
                height,
                start_offset,
                batch_stride
            );
        }

        let const_vec = vec![batch > 1, start_offset == 0];

        let mut meta = get_meta(dev);

        meta.add(width);
        meta.add(height);
        meta.add(start_offset);
        meta.add(batch_stride);

        let pipeline = meta.get_pipeline_const(pipeline, const_vec);

        let bind_group = create_bind_group_input1(buffer_dest, buffer_input, dtype.into());
        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            (width + tile_w - 1) / tile_w,
            (height + tile_h - 1) / tile_h,
            batch,
            (width * height * batch) as usize,
            #[cfg(feature="wgpu_debug")]
            _debug_info
        );
        Ok(())
    }
}

mod sgemm {

    use super::*;
    use crate::{Layout, Shape};

    //#[cfg(feature = "wgpu_debug")]
    fn get_debug_string(params: &SGEMMParams) -> String {
        let b = params.b;
        let m = params.m;
        let n = params.n;
        let k = params.k;
        let use_batch = b != 1;
        if use_batch {
            format!("Batched: {b}*({m}x{k} * {k}x{n})")
        } else {
            format!("({m}x{k} * {k}x{n})")
        }
    }

    pub fn queue_matmul_buffer1(
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        params: SGEMMParams,
        layout_input1: &Layout,
        layout_input2: &Layout,
        _dtype: crate::DType,
        pipeline: Pipelines,
        is_16bytes_aligned: bool,
    ) -> crate::Result<()> {
        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        let input2_stride_b = *input2_stride.next().unwrap_or(&1);

        let const_vec = vec![
            (input1_stride_k == 1) as usize,
            (input1_stride_m == 1) as usize,
            (input2_stride_n == 1) as usize,
            (input2_stride_k == 1) as usize,
            (params.b != 1) as usize,
        ];

        let mut meta = get_meta(dev);
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

        let input_alignment: BindgroupAlignment = _dtype.into();
        let bind_group = if input_alignment == BindgroupAlignment::Aligned4 && is_16bytes_aligned {
            create_bind_group_input2_with_alignment(
                buffer_dest,
                buffer_input1,
                buffer_input2,
                BindgroupAlignmentLayout::Bindgroup2(
                    BindgroupAlignment::Aligned4,
                    BindgroupAlignment::Aligned16,
                    BindgroupAlignment::Aligned16,
                ),
            )
        } else {
            create_bind_group_input2_with_alignment(
                buffer_dest,
                buffer_input1,
                buffer_input2,
                BindgroupAlignmentLayout::Bindgroup2(
                    BindgroupAlignment::Aligned4,
                    BindgroupAlignment::Aligned4,
                    BindgroupAlignment::Aligned4,
                ),
            )
        };

        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            (params.n + 15) / 16,
            (params.m + 15) / 16,
            params.b,
            params.k as usize * params.m as usize * params.n as usize,
            #[cfg(feature = "wgpu_debug")]
            Some(get_debug_string(&params)),
        );
        Ok(())
    }

    fn round_to_next_divisible(num: u32, n: u32) -> u32 {
        if n == 0 {
            panic!("Divisor cannot be zero");
        }
        (num + n - 1) / n * n
    }

    #[derive(Debug)]
    pub(crate) enum StrideOptimization {
        None,           //no stride preferred
        StrideK(bool),  //if true stride must be 1, if false stride is preferred to 1
        StrideNM(bool), //if true stride must be 1, if false stride is preferred to 1
    }

    pub(crate) struct GenericMatmulSettings {
        pub m_tile: u32,
        pub n_tile: u32,
        pub k_tile: u32,

        pub input1_stride: StrideOptimization,
        pub input2_stride: StrideOptimization,

        pub needs_padding: bool, //wheter this shader input matrices must be padded if it is not divisible by tile size
        pub alignment: bool,
    }

    impl GenericMatmulSettings {
        pub(crate) fn new(
            m_tile: u32,
            n_tile: u32,
            k_tile: u32,
            input1_stride: StrideOptimization,
            input2_stride: StrideOptimization,
        ) -> Self {
            Self {
                m_tile,
                n_tile,
                k_tile,
                input1_stride,
                input2_stride,
                needs_padding: true,
                alignment: true,
            }
        }

        pub(crate) fn new_nopadding(
            m_tile: u32,
            n_tile: u32,
            k_tile: u32,
            input1_stride: StrideOptimization,
            input2_stride: StrideOptimization,
        ) -> Self {
            Self {
                m_tile,
                n_tile,
                k_tile,
                input1_stride,
                input2_stride,
                needs_padding: false,
                alignment: true,
            }
        }

        pub(crate) fn need_padding_input1(&self, k_stride: usize, m_stride: usize) -> bool {
            (m_stride != 1 && k_stride != 1)
                || match &self.input1_stride {
                    StrideOptimization::StrideK(true) => k_stride != 1,
                    StrideOptimization::StrideNM(true) => m_stride != 1,
                    _ => false,
                }
        }

        pub(crate) fn need_padding_input2(&self, n_stride: usize, k_stride: usize) -> bool {
            (n_stride != 1 && k_stride != 1)
                || match &self.input2_stride {
                    StrideOptimization::StrideK(true) => k_stride != 1,
                    StrideOptimization::StrideNM(true) => n_stride != 1,
                    _ => false,
                }
        }
    }

    pub fn queue_matmul_generic(
        dev: &WgpuDevice,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        params: SGEMMParams,
        layout_input1: &Layout,
        layout_input2: &Layout,
        dtype: crate::DType,
        settings: GenericMatmulSettings,
        pipeline: Pipelines,
    ) -> crate::Result<()> {
        let m_tile = settings.m_tile;
        let n_tile = settings.n_tile;
        let k_tile = settings.k_tile;

        const NON_PADDED: bool = false;

        let new_m;
        let new_n;
        let new_k;

        if NON_PADDED {
            new_m = params.m;
            new_n = params.n;
            new_k = params.k;
        } else {
            new_m = round_to_next_divisible(params.m, m_tile);
            new_n = round_to_next_divisible(params.n, n_tile);
            new_k = round_to_next_divisible(params.k, k_tile);
        }

        const USE_DIFFERENT_PADDED_OUTPUT: bool = true;

        let need_different_output_buffer = params.m != new_m || params.n != new_n;

        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
     
        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);

        assert!(k_tile % 4 == 0);

        let no_padding_needed_input1_tile = ((params.m % m_tile == 0 && params.k % k_tile == 0) || NON_PADDED)
            && layout_input1.start_offset() % 4 == 0 //input will be loaded 16 bytes aligned
            && !(input1_stride_m != 1 && input1_stride_k != 1);

        let no_padding_needed_input1_stride =
            !settings.need_padding_input1(input1_stride_k, input1_stride_m);
        let no_padding_needed_input1 =
            no_padding_needed_input1_tile && no_padding_needed_input1_stride;

        let no_padding_needed_input2_tile = ((params.n % n_tile == 0 && params.k % k_tile == 0)
            || NON_PADDED)
            && layout_input2.start_offset() % 4 == 0
            && !(input2_stride_n != 1 && input2_stride_k != 1);

        let no_padding_needed_input2_stride =
            !settings.need_padding_input2(input2_stride_n, input2_stride_k);
        let no_padding_needed_input2 =
            no_padding_needed_input2_tile && no_padding_needed_input2_stride;

        let (buffer_input1_padded, layout_input1_padded) = if no_padding_needed_input1 {
            (buffer_input1, layout_input1.clone())
        } else {
            let mut cache = dev.cache.lock().unwrap();
            let buffer_input1_padded;
            let mut dest_layout;
            //we need to realy pad the input:
            let can_transpose = (input1_stride_k==1) != (input1_stride_m ==1); //either stride k or m (but not both) must be one for the transpose shader to work.
            
            let should_transpose_while_padding =  !no_padding_needed_input1_stride && !can_transpose;
            
            if !no_padding_needed_input1_tile || should_transpose_while_padding {
                buffer_input1_padded = cache.create_buffer_reference(
                    params.b * (new_m * new_k) * dtype.size_in_bytes() as u32,
                    false,
                );

                let is_contiguous = if should_transpose_while_padding ||  ((input1_stride_k==1) && (input1_stride_m ==1)){
                    !matches!(settings.input1_stride, StrideOptimization::StrideNM(_))
                } else {
                    input1_stride_k == 1
                };

                if is_contiguous {
                    dest_layout = crate::Layout::contiguous(Shape::from((
                        params.b as usize,
                        new_m as usize,
                        new_k as usize,
                    )));
                }
                else{
                    dest_layout = crate::Layout::new(
                        Shape::from((params.b as usize, new_m as usize, new_k as usize)),
                        vec![(new_m * new_k) as usize, 1, new_m as usize],
                        0,
                    );
                }
                super::queue_copy3d_padded(
                    dev,
                    buffer_input1_padded,
                    buffer_input1,
                    dtype,
                    layout_input1,
                    (params.b, params.m, params.k),
                    &dest_layout,
                    Some(format!("{}: input1", get_debug_string(&params)))
                )?;
            } else {
                buffer_input1_padded = buffer_input1;
                dest_layout = layout_input1.clone();
            }

            //we need to transpose the input matrix
            if !no_padding_needed_input1_stride && can_transpose {
                let buffer_input1_tranposed = cache.create_buffer_reference(
                    params.b * (new_m * new_k) * dtype.size_in_bytes() as u32,
                    false,
                );
                let width;
                let height;
                let start_offset =  dest_layout.start_offset();
                let batch_stride = *dest_layout.stride().iter().rev().nth(2).unwrap_or(&1);
                if let StrideOptimization::StrideNM(_) = settings.input1_stride {
                    dest_layout = crate::Layout::new(
                        Shape::from((params.b as usize, new_m as usize, new_k as usize)),
                        vec![(new_m * new_k) as usize, 1, new_m as usize],
                        0,
                    );
                    width = new_k;
                    height = new_m;
                } else {
                    dest_layout = crate::Layout::contiguous_with_offset(
                        Shape::from((params.b as usize, new_m as usize, new_k as usize)),
                        0,
                    );
                    width = new_m;
                    height = new_k;
                }
                transpose::queue_transpose3d_generic(
                    dev,
                    buffer_input1_tranposed,
                    buffer_input1_padded,
                    dtype,
                    params.b,
                    width,
                    height,
                    start_offset,
                    batch_stride,
                    Some(format!("{}: input1", get_debug_string(&params)))
                )?;

                (buffer_input1_tranposed, dest_layout)
            } else {
                (buffer_input1_padded, dest_layout)
            }
        };

        let (buffer_input2_padded, layout_input2_padded) = if no_padding_needed_input2 {
            (buffer_input2, layout_input2.clone())
        } else {
            let mut cache = dev.cache.lock().unwrap();

            let mut dest_layout;
            let buffer_input2_padded;

            let can_transpose =  //false;
                (input2_stride_k==1) != (input2_stride_n ==1); //either stride k or n (but not both) must be one for the transpose shader to work.

            let should_transpose_while_padding = !no_padding_needed_input2_stride && !can_transpose;
            
            if !no_padding_needed_input2_tile || should_transpose_while_padding
            {
                buffer_input2_padded = cache.create_buffer_reference(
                    params.b * (new_k * new_n) * dtype.size_in_bytes() as u32,
                    false,
                );

                let is_contiguous = if should_transpose_while_padding {
                    matches!(settings.input2_stride, StrideOptimization::StrideNM(_))
                } else {
                    input2_stride_k != 1
                };

                if is_contiguous {
                    dest_layout = crate::Layout::contiguous(Shape::from((
                        params.b as usize,
                        new_k as usize,
                        new_n as usize,
                    )));
                }
                else{
                    dest_layout = crate::Layout::new(
                        Shape::from((params.b as usize, new_k as usize, new_n as usize)),
                        vec![(new_n * new_k) as usize, 1, new_k as usize],
                        0,
                    );
                }
                super::queue_copy3d_padded(
                    dev,
                    buffer_input2_padded,
                    buffer_input2,
                    dtype,
                    layout_input2,
                    (params.b, params.k, params.n),
                    &dest_layout,
                    Some(format!("{}: input2", get_debug_string(&params))),
                )?;
            } else {
                buffer_input2_padded = buffer_input2;
                dest_layout = layout_input2.clone();
            }

            if !no_padding_needed_input2_stride && can_transpose {
                let buffer_input2_tranposed = cache.create_buffer_reference(
                    params.b * (new_k * new_n) * dtype.size_in_bytes() as u32,
                    false,
                );

                let width;
                let height;
                let start_offset = dest_layout.start_offset();
                let batch_stride = *dest_layout.stride().iter().rev().nth(2).unwrap_or(&1);
                if let StrideOptimization::StrideNM(_) = settings.input2_stride {
                    dest_layout = crate::Layout::contiguous_with_offset(
                        Shape::from((params.b as usize, new_k as usize, new_n as usize)),
                        0,
                    );
                    width = new_k;
                    height = new_n;
                } else {
                    dest_layout = crate::Layout::new(
                        Shape::from((params.b as usize, new_k as usize, new_n as usize)),
                        vec![(new_n * new_k) as usize, 1, new_k as usize],
                        0
                    );
                    width = new_n;
                    height = new_k;
                }

                transpose::queue_transpose3d_generic(
                    dev,
                    buffer_input2_tranposed,
                    buffer_input2_padded,
                    dtype,
                    params.b,
                    width,
                    height,
                    start_offset,
                    batch_stride,
                    Some(format!("{}: input2", get_debug_string(&params)))
                )?;
                (buffer_input2_tranposed, dest_layout)
            } else {
                (buffer_input2_padded, dest_layout)
            }
        };

        let buffer_dest_padded = if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT {
            let mut cache = dev.cache.lock().unwrap();
            cache.create_buffer_reference(
                params.b * (new_m * new_n) * dtype.size_in_bytes() as u32,
                false,
            )
        } else {
            buffer_dest
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
        let const_vec = vec![
            (input1_stride_k == 1) as usize,
            (input1_stride_m == 1) as usize,
            (input2_stride_n == 1) as usize,
            (input2_stride_k == 1) as usize,
            use_batch as usize,
        ];

        let mut meta = get_meta(dev);
        meta.add(params.b);
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {
            new_m
        } else {
            params.m
        });
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {
            new_k
        } else {
            params.k
        });
        meta.add(if USE_DIFFERENT_PADDED_OUTPUT {
            new_n
        } else {
            params.n
        });

        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1_padded.start_offset()); //input1_offset

        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2_padded.start_offset()); //input2_offset

        meta.add(input1_stride_k);
        meta.add(input1_stride_m);
        meta.add(input2_stride_n);
        meta.add(input2_stride_k);

        if need_different_output_buffer && !USE_DIFFERENT_PADDED_OUTPUT {
            meta.add_const(candle_wgpu_kernels::Constants::Isoutputpadded, true);
        }

        let pipeline = meta.get_pipeline_const(pipeline, const_vec.clone());
        let input_alignment: BindgroupAlignment = dtype.into();
        if input_alignment != BindgroupAlignment::Aligned4 {
            panic!("matmul can only be performed with f32 and i32");
        }

        let bind_group = create_bind_group_input2_with_alignment(
            buffer_dest_padded,
            buffer_input1_padded,
            buffer_input2_padded,
            BindgroupAlignmentLayout::Bindgroup2(
                BindgroupAlignment::Aligned4,
                BindgroupAlignment::Aligned16,
                BindgroupAlignment::Aligned16,
            ),
        );

        let lx;
        let ly;
        if NON_PADDED {
            lx = (new_n + n_tile - 1) / n_tile;
            ly = (new_m + m_tile - 1) / m_tile;
        } else {
            lx = (new_n) / n_tile;
            ly = (new_m) / m_tile;
        }

        enqueue_workgroups_extra(
            meta,
            pipeline,
            bind_group,
            lx,
            ly,
            params.b,
            params.k as usize * params.m as usize * params.n as usize,
            #[cfg(feature = "wgpu_debug")]
            Some(get_debug_string(&params)),
        );

        if need_different_output_buffer && USE_DIFFERENT_PADDED_OUTPUT {
            //let res : Vec<f32> = pollster::block_on(read_data_from_gpu_async(dev, buffer_dest_padded.clone()));
            //println!("res: {:?}", res);
            let dest_padding_layout = crate::Layout::contiguous(Shape::from((
                params.b as usize,
                new_m as usize,
                new_n as usize,
            )));
            let dest_layout = crate::Layout::contiguous(Shape::from((
                params.b as usize,
                params.m as usize,
                params.n as usize,
            )));

            super::queue_copy3d(
                dev,
                buffer_dest,
                buffer_dest_padded,
                dtype,
                &dest_padding_layout,
                (params.b, params.m, params.n),
                &dest_layout,
            )?;
        }

        Ok(())
    }
}

pub fn queue_matmul_buffer(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params: SGEMMParams,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let alg = dev.matmul_alg.lock().unwrap();
    queue_matmul_buffer_alg(
        dev,
        buffer_dest,
        buffer_input1,
        buffer_input2,
        params,
        layout_input1,
        layout_input2,
        dtype,
        alg.clone(),
    )
}

fn get_matmul_setting(alg: &MatmulAlgorithm) -> GenericMatmulSettings {
    match alg {
        MatmulAlgorithm::Matmul7 => GenericMatmulSettings::new(
            16,
            16,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul1 => GenericMatmulSettings::new_nopadding(
            16,
            16,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul1_4 => GenericMatmulSettings::new_nopadding(
            16,
            16,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul16_16 => GenericMatmulSettings::new(
            16,
            16,
            4,
            StrideOptimization::None,
            StrideOptimization::StrideNM(true),
        ),

        MatmulAlgorithm::Matmul32_64 => GenericMatmulSettings::new(
            32,
            64,
            4,
            StrideOptimization::None,
            StrideOptimization::StrideNM(true),
        ), //this shader was way slower when input2, stride n != 1

        MatmulAlgorithm::Matmul32_64B => GenericMatmulSettings::new(
            32,
            64,
            8,
            StrideOptimization::None,
            StrideOptimization::StrideK(true),
        ), 


        MatmulAlgorithm::Matmul32_32 => GenericMatmulSettings::new(
            32,
            32,
            8,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul64_64 => GenericMatmulSettings::new(
            64,
            64,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul64_64_8_8 => GenericMatmulSettings::new(
            64,
            64,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul64_64_4_8 => GenericMatmulSettings::new(
            64,
            64,
            16,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul1_64 => GenericMatmulSettings::new(
            1,
            64,
            64,
            StrideOptimization::None,
            StrideOptimization::None,
        ),
        MatmulAlgorithm::Matmul1_64B => GenericMatmulSettings::new(
            1,
            64,
            128,
            StrideOptimization::StrideK(true),
            StrideOptimization::StrideK(true),
        ),
        MatmulAlgorithm::Matmul24_24 => GenericMatmulSettings::new(
            24,
            24,
            32,
            StrideOptimization::None,
            StrideOptimization::StrideK(true),
        ),
        MatmulAlgorithm::Matmul24_48 => GenericMatmulSettings::new(
            24,
            48,
            32,
            StrideOptimization::None,
            StrideOptimization::StrideK(true),
        ),
        MatmulAlgorithm::Matmul24_24B => GenericMatmulSettings::new(
            24,
            24,
            8,
            StrideOptimization::None,
            StrideOptimization::StrideNM(true),
        ),
        MatmulAlgorithm::Matmul24_48B => GenericMatmulSettings::new(
            24,
            48,
            8,
            StrideOptimization::None,
            StrideOptimization::StrideNM(true),
        ),
        alg => {
            panic!("alg {alg:?} not supported")
        }
    }
}

pub fn queue_matmul_buffer_alg(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params: SGEMMParams,
    layout_input1: &Layout,
    layout_input2: &Layout,
    cdtype: crate::DType,
    alg: MatmulAlgorithm,
) -> crate::Result<()> {
    let dtype = get_dtype(cdtype)?;
    match alg {
        MatmulAlgorithm::MatmulX => {
            return queue_matmul_buffer_best(
                dev,
                buffer_dest,
                buffer_input1,
                buffer_input2,
                params,
                layout_input1,
                layout_input2,
                cdtype,
            )
        }
        MatmulAlgorithm::Matmul1_4 => {
            return super::matmul::sgemm::queue_matmul_buffer1(
                dev,
                buffer_dest,
                buffer_input1,
                buffer_input2,
                params,
                layout_input1,
                layout_input2,
                cdtype,
                Pipelines::Matmul(dtype, matmul::Functions::Matmul116),
                true,
            )
        }
        MatmulAlgorithm::Matmul7 => {
            return super::matmul::sgemm::queue_matmul_buffer1(
                dev,
                buffer_dest,
                buffer_input1,
                buffer_input2,
                params,
                layout_input1,
                layout_input2,
                cdtype,
                Pipelines::Matmul(dtype, matmul::Functions::Matmul7),
                false,
            )
        }
        MatmulAlgorithm::Matmul1 => {
            return super::matmul::sgemm::queue_matmul_buffer1(
                dev,
                buffer_dest,
                buffer_input1,
                buffer_input2,
                params,
                layout_input1,
                layout_input2,
                cdtype,
                Pipelines::Matmul(dtype, matmul::Functions::Matmul1),
                false,
            )
        }
        _ => {}
    }
    use candle_wgpu_kernels::{matmul, sgemm};

    let setting = get_matmul_setting(&alg);

    let pipeline = match alg {
        MatmulAlgorithm::Matmul7 => Pipelines::Matmul(dtype, matmul::Functions::Matmul7),
        MatmulAlgorithm::Matmul1 => Pipelines::Matmul(dtype, matmul::Functions::Matmul1),
        MatmulAlgorithm::Matmul1_4 => Pipelines::Matmul(dtype, matmul::Functions::Matmul116),
        MatmulAlgorithm::Matmul16_16 => {
            Pipelines::Matmul16x16(dtype, sgemm::matmul16x16::Functions::Matmul)
        }

        MatmulAlgorithm::Matmul32_64 => {
            Pipelines::Matmul32x64(dtype, sgemm::matmul32x64::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul32_64B => {
            Pipelines::Matmul32x64b(dtype, sgemm::matmul32x64b::Functions::Matmul)
        }

        MatmulAlgorithm::Matmul32_32 => {
            Pipelines::Matmul32x32(dtype, sgemm::matmul32x32::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul64_64 => {
            Pipelines::Matmul64x64(dtype, sgemm::matmul64x64::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul64_64_8_8 => {
            Pipelines::Matmul64x648x8(dtype, sgemm::matmul64x64_8x8::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul64_64_4_8 => {
            Pipelines::Matmul64x644x8(dtype, sgemm::matmul64x64_4x8::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul1_64 => {
            Pipelines::Matmul1x64(dtype, sgemm::matmul1x64::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul1_64B => {
            Pipelines::Matmul1x64b(dtype, sgemm::matmul1x64b::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul24_24 => {
            Pipelines::Matmul24x24(dtype, sgemm::matmul24x24::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul24_48 => {
            Pipelines::Matmul24x48(dtype, sgemm::matmul24x48::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul24_24B => {
            Pipelines::Matmul24x24b(dtype, sgemm::matmul24x24b::Functions::Matmul)
        }
        MatmulAlgorithm::Matmul24_48B => {
            Pipelines::Matmul24x48b(dtype, sgemm::matmul24x48b::Functions::Matmul)
        }
        alg => {
            panic!("alg {alg:?} not supported")
        }
    };

    super::matmul::sgemm::queue_matmul_generic(
        dev,
        buffer_dest,
        buffer_input1,
        buffer_input2,
        params,
        layout_input1,
        layout_input2,
        cdtype,
        setting,
        pipeline,
    )
}

fn get_matmul_naive(
    k: usize,
    layout_input1: &Layout,
    layout_input2: &Layout,
) -> crate::wgpu_backend::MatmulAlgorithm {
    if k % 4 == 0 && layout_input1.start_offset() % 4 == 0 && layout_input2.start_offset() % 4 == 0
    {
        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);

        let _input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);

        if input1_stride_k == 1 && input2_stride_k == 1 {
            return crate::wgpu_backend::MatmulAlgorithm::Matmul1_4;
        }
    }
    crate::wgpu_backend::MatmulAlgorithm::Matmul1
}

pub fn queue_matmul_buffer_best(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    params: SGEMMParams,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let b = params.b as usize;
    let m = params.m as usize;
    let k = params.k as usize;
    let n = params.n as usize;

    let mut input1_stride = layout_input1.stride().iter().rev();
    let input1_stride_k = *input1_stride.next().unwrap_or(&1);
    let input1_stride_m = *input1_stride.next().unwrap_or(&1);

    let mut input2_stride = layout_input2.stride().iter().rev();
    let input2_stride_n = *input2_stride.next().unwrap_or(&1);
    let input2_stride_k = *input2_stride.next().unwrap_or(&1);

    let alg;
    if m <= 2 || n <= 2 {
        if m <= 2 {
            if k % 64 == 0 && n % 128 == 0 && input2_stride_k == 1 {
                alg = MatmulAlgorithm::Matmul1_64B;
            } else if k % 64 == 0 && n % 64 == 0 && input2_stride_n == 1 {
                alg = MatmulAlgorithm::Matmul1_64;
            } else {
                alg = get_matmul_naive(k, layout_input1, layout_input2);
            }
        } else {
            alg = get_matmul_naive(k, layout_input1, layout_input2);
        }
    } else {
        let shaders = [
            MatmulAlgorithm::Matmul32_64,
            MatmulAlgorithm::Matmul32_64B,
            MatmulAlgorithm::Matmul32_32,
            //MatmulAlgorithm::Matmul64_64_4_8,
            MatmulAlgorithm::Matmul24_48,
            MatmulAlgorithm::Matmul24_48B,
            MatmulAlgorithm::Matmul24_24,
            MatmulAlgorithm::Matmul24_24B,
            MatmulAlgorithm::Matmul16_16,
        ];

        let mut best_no_padding_25: Option<&MatmulAlgorithm> = None;
        let mut best_no_padding_tiled_25: Option<&MatmulAlgorithm> = None;
        let mut best_wgs_25: Option<&MatmulAlgorithm> = None;
        let mut best_no_padding_tiled_wgs: u32 = 0;

        for a in shaders.iter() {
            let s = get_matmul_setting(a);

            let no_padding_tiled = !s.needs_padding
                || (m % s.m_tile as usize == 0
                    && k % s.k_tile as usize == 0
                    && n % s.n_tile as usize == 0);
            let no_padding_stride = !s.needs_padding
                || (!s.need_padding_input1(input1_stride_k, input1_stride_m)
                    && !s.need_padding_input2(input2_stride_n, input2_stride_k)
                    && (!s.alignment
                        || (layout_input1.start_offset() % 4 == 0
                            && layout_input2.start_offset() % 4 == 0)));

            let no_padding_needed = no_padding_tiled && no_padding_stride;

            let lm = (m as u32 + s.m_tile - 1) / s.m_tile;
            let ln = (n as u32 + s.n_tile - 1) / s.n_tile;
            let new_k = (((k as u32 + s.k_tile - 1) / s.k_tile) * s.k_tile) as usize;
            let new_m = (lm * s.m_tile) as usize;
            let new_n = (ln * s.n_tile) as usize;
            let wgs = lm * ln;

            if no_padding_needed {
                if wgs > 64 && (best_no_padding_tiled_wgs == 0 || wgs * 8 < best_no_padding_tiled_wgs) {
                    //make sure, that we dont select 16x16, if we could use 32x64 but have to transpose matrix b
                    alg = a.clone();
                    return queue_matmul_buffer_alg(
                        dev,
                        buffer_dest,
                        buffer_input1,
                        buffer_input2,
                        params,
                        layout_input1,
                        layout_input2,
                        dtype,
                        alg,
                    );
                }
                if wgs >= 25 && best_no_padding_25.is_none() {
                    best_no_padding_25 = Some(a); // Store the first match
                }
            } else {
                let new_input1_size = b * new_k * new_m * dtype.size_in_bytes();
                let new_input2_size = b * new_k * new_n * dtype.size_in_bytes();
                let new_output_size = b * new_m * new_n * dtype.size_in_bytes();

                if new_input1_size > dev.device_limits.max_storage_buffer_binding_size as usize
                    || new_input2_size > dev.device_limits.max_storage_buffer_binding_size as usize
                    || new_output_size > dev.device_limits.max_storage_buffer_binding_size as usize
                {
                    continue;
                }
                if wgs >= 25 {
                    if no_padding_tiled && best_no_padding_tiled_25.is_none() {
                        best_no_padding_tiled_25 = Some(a); // Store the first match
                        best_no_padding_tiled_wgs = wgs;
                    }
                    if best_wgs_25.is_none() {
                        best_wgs_25 = Some(a); // Store the first match
                    }
                }
            }
        }
        if let Some(entry) = best_no_padding_25 {
            alg = entry.clone();
        } else if let Some(entry) = best_no_padding_tiled_25 {
            //we need to pad the input because of stride ristrictions, but we do not increase input buffers
            alg = entry.clone();
        } else if let Some(entry) = best_wgs_25 {
            alg = entry.clone();
        } else {
            alg = get_matmul_naive(k, layout_input1, layout_input2);
        }
    }
    queue_matmul_buffer_alg(
        dev,
        buffer_dest,
        buffer_input1,
        buffer_input2,
        params,
        layout_input1,
        layout_input2,
        dtype,
        alg,
    )
}
