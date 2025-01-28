use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, DType, Kernels, MetalKernelError, Source};
use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLResourceOptions, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn call_arg_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    nrows: usize,
    ncols: usize,
    ncols_pad: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), crate::MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Sort, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&src, dst, ncols as i64, ncols_pad as i64));

    let thread_group_count = MTLSize {
        width: 1,
        height: nrows as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: ncols_pad as u64,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(src.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);
    encoder.set_threadgroup_memory_length(0, (ncols_pad * 4).max(16) as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

fn mlx_dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I64 => "int64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
    }
}

#[allow(clippy::too_many_arguments)]
pub fn multi_block_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    bn: usize,
    tn: usize,
    nblocks: usize,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dtype_str = mlx_dtype_str(dtype);
    // Do allocations
    let el_count = nrows * ncols;
    let bytes_len = (el_count * dtype.size_in_bytes()) as u64;
    let mut dev_vals_0 = device.new_buffer(bytes_len, MTLResourceOptions::StorageModePrivate);
    let mut dev_vals_1 = device.new_buffer(bytes_len, MTLResourceOptions::StorageModePrivate);
    let mut dev_idxs_0 =
        device.new_buffer(el_count as u64 * 4, MTLResourceOptions::StorageModePrivate);
    let mut dev_idxs_1 =
        device.new_buffer(el_count as u64 * 4, MTLResourceOptions::StorageModePrivate);
    let mut block_partitions = device.new_buffer(
        (nrows * (nblocks + 1)) as u64 * 4,
        MTLResourceOptions::StorageModePrivate,
    );
    // Prepare command encoder
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    // Do blockwise sort
    {
        let name = format!("sort_mbsort_{dtype_str}_uint32_bn{bn}_tn{tn}");
        let pipeline = kernels.load_pipeline(device, Source::MlxSort, name)?;
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(
            encoder,
            (
                &src,
                &mut dev_vals_0,
                &mut dev_idxs_0,
                /* size_sorted_axis */ ncols as i32,
                /* stride_sorted_axis */ 1i32,
                /* nc_dim */ 1i32,
                /* nc_shape */ nrows as i32,
                /* nc_str */ ncols as i32
            )
        );
        let thread_group_count = MTLSize {
            width: nblocks as u64,
            height: nrows as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: bn as u64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }
    // Do merges
    let mut ping = false;
    let mut merge_tiles = 2;
    let n_thr_per_group = usize::min(nblocks + 1, 1024);
    let partition_name = format!("partition_mbsort_{dtype_str}_uint32_bn{bn}_tn{tn}");
    let merge_name = format!("merge_mbsort_float32_uint32_bn{bn}_tn{tn}");
    while merge_tiles / 2 < nblocks {
        let (dev_vals_in, dev_vals_out) = if ping {
            (&mut dev_vals_1, &mut dev_vals_0)
        } else {
            (&mut dev_vals_0, &mut dev_vals_1)
        };
        let (dev_idxs_in, dev_idxs_out) = if ping {
            (&mut dev_idxs_1, &mut dev_idxs_0)
        } else {
            (&mut dev_idxs_0, &mut dev_idxs_1)
        };
        ping = !ping;
        // Do partition
        {
            let pipeline =
                kernels.load_pipeline(device, Source::MlxSort, partition_name.clone())?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    &mut block_partitions,
                    &mut *dev_vals_in,
                    &mut *dev_idxs_in,
                    /* size_sorted_axis */ ncols as i32,
                    /* merge_tiles */ merge_tiles as i32,
                    /* n_blocks */ nblocks as i32
                )
            );
            let thread_group_count = MTLSize {
                width: 1,
                height: nrows as u64,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: n_thr_per_group as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        }
        // Do merge
        {
            let pipeline = kernels.load_pipeline(device, Source::MlxSort, merge_name.clone())?;
            encoder.set_compute_pipeline_state(&pipeline);
            set_params!(
                encoder,
                (
                    &block_partitions,
                    &*dev_vals_in,
                    &*dev_idxs_in,
                    &*dev_vals_out,
                    &*dev_idxs_out,
                    /* size_sorted_axis */ ncols as i32,
                    /* merge_tiles */ merge_tiles as i32,
                    /* n_blocks */ nblocks as i32
                )
            );
            let thread_group_count = MTLSize {
                width: nblocks as u64,
                height: nrows as u64,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: bn as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        }
        merge_tiles *= 2;
    }
    let dev_idxs_out = if ping {
        &mut dev_idxs_1
    } else {
        &mut dev_idxs_0
    };
    // Copy output with appropriate strides
    let copy_kernel = match dtype {
        DType::U8 => crate::copy2d::U8,
        DType::U32 => crate::copy2d::U32,
        DType::I64 => crate::copy2d::I64,
        DType::BF16 => crate::copy2d::BFLOAT,
        DType::F16 => crate::copy2d::HALF,
        DType::F32 => crate::copy2d::FLOAT,
    };
    crate::call_copy2d(
        device,
        encoder,
        kernels,
        copy_kernel,
        dev_idxs_out,
        dst,
        /* d1 */ nrows,
        /* d2 */ ncols,
        /* src_s */ ncols,
        /* dst_s */ ncols,
        /* src_o_in_bytes */ 0,
        /*dst_o_in_bytes */ 0,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn block_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    bn: usize,
    tn: usize,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dtype_str = mlx_dtype_str(dtype);
    let name = format!("carg_block_sort_{dtype_str}_uint32_bn{bn}_tn{tn}");
    let pipeline = kernels.load_pipeline(device, Source::MlxSort, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            &src,
            dst,
            ncols as i32,
            1i32,
            1i32,
            ncols as i32,
            ncols as i32
        )
    );
    let thread_group_count = MTLSize {
        width: 1,
        height: nrows as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: bn as u64,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(src.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_arg_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let tn = 8;
    let bn = match ncols.div_ceil(tn) {
        257.. if dtype.size_in_bytes() <= 4 => 512,
        129.. => 256,
        0..129 => 128,
    };
    let n_per_block = bn * tn;
    let n_blocks = ncols.div_ceil(n_per_block);
    if n_blocks > 1 {
        multi_block_sort(
            device, ep, kernels, dtype, bn, tn, n_blocks, nrows, ncols, src, dst,
        )?
    } else {
        block_sort(device, ep, kernels, dtype, bn, tn, nrows, ncols, src, dst)?
    }
    Ok(())
}
