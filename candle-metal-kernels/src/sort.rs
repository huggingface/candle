use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Kernels, MetalKernelError, Source};
use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLSize};

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

#[allow(clippy::too_many_arguments)]
pub fn block_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: String,
    bn: usize,
    nrows: usize,
    ncols: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
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
    nrows: usize,
    ncols: usize,
    size_of_dtype: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let tn = 8;
    let bn = match ncols.div_ceil(tn) {
        257.. if size_of_dtype <= 4 => 512,
        129.. => 256,
        0..129 => 128,
    };
    let n_per_block = bn * tn;
    let n_blocks = ncols.div_ceil(n_per_block);
    if n_blocks > 1 {
        todo!()
    } else {
        let name = format!("carg_block_sort_float32_uint32_bn{bn}_tn{tn}");
        block_sort(device, ep, kernels, name, bn, nrows, ncols, src, dst)?
    }
    Ok(())
}
