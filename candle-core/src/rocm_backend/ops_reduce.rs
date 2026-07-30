//! Launchers for the reduction and index-select kernels in `reduce.cu` /
//! `indexing.cu`.

use super::{
    kernels, launch_config, launch_kernel, try_kernel_name, Map1Any, RocmDevice,
    SendSyncDeviceMemory, S,
};
use crate::op::ReduceOp;
use crate::{Layout, Result, WithDType};

pub(super) struct FastReduce<'a>(pub &'a [usize], pub ReduceOp);

impl Map1Any for FastReduce<'_> {
    fn f<T: Copy + Send + Sync + 'static, W: Fn(SendSyncDeviceMemory<T>) -> S>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S> {
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();

        // Source dims and strides with the reduced dims moved to the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(layout.stride()[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(layout.stride()[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        // The reduction loop needs the shared array fully initialized, which
        // requires the thread count to be a power of two.
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();

        let (name, check_empty, return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false, false),
            ReduceOp::Min => ("fast_min", true, false),
            ReduceOp::Max => ("fast_max", true, false),
            ReduceOp::ArgMin => ("fast_argmin", true, true),
            ReduceOp::ArgMax => ("fast_argmax", true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }

        let func_name = try_kernel_name::<T>(name)?;

        let ds_data: Vec<usize> = [dims.as_slice(), stride.as_slice()].concat();
        let ds = dev.clone_htod(&ds_data)?;

        // `fast_*` maps one block to one output element, so the grid is sized by
        // the output rather than through `launch_config`.
        let grid = rocm_rs::hip::Dim3::from(dst_el as u32);
        let block = rocm_rs::hip::Dim3::from(block_dim as u32);

        let launch = |out_ptr: *mut std::ffi::c_void| -> Result<()> {
            unsafe {
                let src_ptr = src.ptr_at(layout.start_offset());
                let ds_ptr = ds.as_ptr() as *const usize;
                launch_kernel(
                    dev,
                    &kernels::REDUCE,
                    &func_name,
                    grid,
                    block,
                    &mut [
                        &src_el as *const usize as *mut std::ffi::c_void,
                        &el_to_sum_per_block as *const usize as *mut std::ffi::c_void,
                        &src_dims.len() as *const usize as *mut std::ffi::c_void,
                        (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                        (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    ],
                )
            }
        };

        if return_index {
            // `fast_argmin`/`fast_argmax` write `uint32_t` whatever the input
            // dtype is, so the output buffer must be u32-sized and u32-tagged.
            let output = dev.alloc::<u32>(dst_el)?;
            launch(output.as_ptr())?;
            Ok(S::U32(output))
        } else {
            let output = dev.alloc::<T>(dst_el)?;
            launch(output.as_ptr())?;
            Ok(wrap(output))
        }
    }
}

pub(super) fn index_select_typed<T: Copy + Send + Sync + WithDType + 'static>(
    ids_prefix: &str,
    ids_ptr: *mut std::ffi::c_void,
    ds: &SendSyncDeviceMemory<usize>,
    src_ptr: *mut std::ffi::c_void,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    dst_el: usize,
    device: &RocmDevice,
) -> Result<SendSyncDeviceMemory<T>> {
    let func_name = try_kernel_name::<T>(ids_prefix)?;
    let output = device.alloc::<T>(dst_el)?;
    let num_dims = ds.count() / 2;
    let (grid, block) = launch_config(dst_el);

    unsafe {
        let out_ptr = output.as_ptr();
        let ds_ptr = ds.as_ptr() as *const usize;

        launch_kernel(
            device,
            &kernels::INDEXING,
            &func_name,
            grid,
            block,
            &mut [
                &dst_el as *const usize as *mut std::ffi::c_void,
                &num_dims as *const usize as *mut std::ffi::c_void,
                (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                &left_size as *const usize as *mut std::ffi::c_void,
                &src_dim_size as *const usize as *mut std::ffi::c_void,
                &ids_dim_size as *const usize as *mut std::ffi::c_void,
                &right_size as *const usize as *mut std::ffi::c_void,
            ],
        )?;
    }

    Ok(output)
}
