//! Launchers for the indexing kernels in `candle-kernels/src/indexing.cu`.
//!
//! Argument order comes from the macros in that file — `IS_OP`, `GATHER_OP`,
//! `IA_OP`, `S_OP`, `SA_OP` — and from nowhere else. They are *not* uniform:
//! `IS_OP` leads with `numel`/`num_dims`/`info`, `GATHER_OP` with `numel` only,
//! `IA_OP` takes `ids_dim_size` right after `ids` while `S_OP`/`SA_OP` take no
//! such argument at all.
//!
//! The index dtype and the value dtype are dispatched separately: the index
//! dtype picks the kernel-name prefix (`gather_u32`) and `Map1`/`Map2InPlace`
//! append the value dtype (`gather_u32_f32`).

use std::ffi::c_void;

use super::launch::{launch_config, launch_kernel};
use super::{
    kernels, try_kernel_name, Map1, Map2InPlace, RocmDevice, RocmStorage, RocmStorageSlice,
    SendSyncDeviceMemory,
};
use crate::{Layout, Result};

/// Kernel-name prefix and device pointer for an index buffer.
///
/// `indexing.cu` only instantiates u8/u32/i64 index kernels for the regular
/// dtypes, so anything else is rejected here rather than at launch time.
fn ids_prefix_and_ptr(
    base: &str,
    ids: &RocmStorageSlice,
    ids_offset: usize,
    op: &'static str,
) -> Result<(String, *mut c_void)> {
    let (suffix, ptr) = match ids {
        RocmStorageSlice::U8(s) => ("u8", unsafe { s.ptr_at(ids_offset) }),
        RocmStorageSlice::U32(s) => ("u32", unsafe { s.ptr_at(ids_offset) }),
        RocmStorageSlice::I64(s) => ("i64", unsafe { s.ptr_at(ids_offset) }),
        slice => crate::bail!("{op} ids should be u8/u32/i64, got {:?}", slice.dtype()),
    };
    Ok((format!("{base}_{suffix}"), ptr))
}

/// Element offset of a contiguous view, or `RequiresContiguous`.
fn contiguous_offset(l: &Layout, op: &'static str) -> Result<usize> {
    match l.contiguous_offsets() {
        Some((o1, _)) => Ok(o1),
        None => Err(crate::Error::RequiresContiguous { op }.bt()),
    }
}

/// `index_select`: picks `ids` entries along `dim` of a contiguous source.
pub(super) struct IndexSelect<'a>(pub &'a RocmStorage, pub &'a Layout, pub usize);

impl Map1 for IndexSelect<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        src_l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let ids_l = self.1;
        let dim = self.2;
        let (name, ids_ptr) =
            ids_prefix_and_ptr("is", &self.0.slice, ids_l.start_offset(), "index-select")?;
        let func_name = try_kernel_name::<T>(&name)?;

        let ids_dims = ids_l.shape().dims();
        let num_dims = ids_dims.len();
        // `IS_OP` dereferences `info` unconditionally to test contiguity, so it
        // never accepts a null pointer here the way the elementwise ops do.
        let ds = dev.clone_htod(&[ids_dims, ids_l.stride()].concat())?;

        let src_ptr = unsafe { src.ptr_at(contiguous_offset(src_l, "index-select")?) };

        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_size = src_l.dims()[dim];
        let ids_dim_size = ids_l.shape().elem_count();
        let dst_el = ids_dim_size * left_size * right_size;

        let output = dev.alloc::<T>(dst_el)?;
        let (grid, block) = launch_config(dev, dst_el);
        unsafe {
            let out_ptr = output.as_ptr();
            let ds_ptr = ds.as_ptr() as *const usize;
            launch_kernel(
                dev,
                &kernels::INDEXING,
                &func_name,
                grid,
                block,
                &mut [
                    &dst_el as *const usize as *mut c_void,
                    &num_dims as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&ids_ptr) as *const *mut c_void as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                    &left_size as *const usize as *mut c_void,
                    &src_dim_size as *const usize as *mut c_void,
                    &ids_dim_size as *const usize as *mut c_void,
                    &right_size as *const usize as *mut c_void,
                ],
            )?;
        }
        Ok(output)
    }
}

/// `gather`: one output element per index, `ids` shaped like the output.
pub(super) struct Gather<'a>(pub &'a RocmStorage, pub &'a Layout, pub usize);

impl Map1 for Gather<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        src_l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let ids_l = self.1;
        let dim = self.2;
        let ids_o1 = contiguous_offset(ids_l, "gather")?;
        let (name, ids_ptr) = ids_prefix_and_ptr("gather", &self.0.slice, ids_o1, "gather")?;
        let func_name = try_kernel_name::<T>(&name)?;

        let src_ptr = unsafe { src.ptr_at(contiguous_offset(src_l, "gather")?) };

        let el = ids_l.shape().elem_count();
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];

        let output = dev.alloc::<T>(el)?;
        let (grid, block) = launch_config(dev, el);
        unsafe {
            let out_ptr = output.as_ptr();
            launch_kernel(
                dev,
                &kernels::INDEXING,
                &func_name,
                grid,
                block,
                &mut [
                    &el as *const usize as *mut c_void,
                    (&ids_ptr) as *const *mut c_void as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                    &left_sz as *const usize as *mut c_void,
                    &src_dim_sz as *const usize as *mut c_void,
                    &ids_dim_sz as *const usize as *mut c_void,
                    &right_sz as *const usize as *mut c_void,
                ],
            )?;
        }
        Ok(output)
    }
}

/// Whether a scatter overwrites (`S_OP`) or accumulates (`SA_OP`).
#[derive(Clone, Copy)]
pub(super) enum ScatterKind {
    Set,
    Add,
}

impl ScatterKind {
    /// Kernel base name and the op name used in error messages.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            ScatterKind::Set => ("s", "scatter"),
            ScatterKind::Add => ("sa", "scatter-add"),
        }
    }
}

/// `scatter_set` / `scatter_add_set`.
///
/// `S_OP` and `SA_OP` have identical signatures, so a single launcher covers
/// both. `SA_OP` accumulates with a plain `out[dst_i] += inp[src_i]` — matching
/// the CUDA backend, races and all.
pub(super) struct Scatter<'a> {
    ids: &'a RocmStorage,
    ids_l: &'a Layout,
    dim: usize,
    kind: ScatterKind,
}

impl<'a> Scatter<'a> {
    pub(super) fn new(
        ids: &'a RocmStorage,
        ids_l: &'a Layout,
        dim: usize,
        kind: ScatterKind,
    ) -> Self {
        Self {
            ids,
            ids_l,
            dim,
            kind,
        }
    }
}

impl Map2InPlace for Scatter<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        dst: &mut SendSyncDeviceMemory<T>,
        dst_l: &Layout,
        src: &SendSyncDeviceMemory<T>,
        src_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<()> {
        let (base, op) = self.kind.names();
        // `indexing.cu` instantiates `SA_OP_F8` for fp8 but no `S_OP`, so
        // scatter-add has a kernel and scatter-set does not. Say so rather than
        // let the driver report a missing symbol.
        if matches!(self.kind, ScatterKind::Set) && std::any::type_name::<T>().contains("F8E4M3") {
            crate::bail!(
                "scatter is not available for F8E4M3 on ROCm: \
                 candle-kernels/src/indexing.cu instantiates SA_OP_F8 but no S_OP \
                 for fp8, so only scatter-add has a kernel"
            )
        }
        let dim = self.dim;
        let ids_o1 = contiguous_offset(self.ids_l, op)?;
        let (name, ids_ptr) = ids_prefix_and_ptr(base, &self.ids.slice, ids_o1, op)?;
        let func_name = try_kernel_name::<T>(&name)?;

        let dst_ptr = unsafe { dst.ptr_at(contiguous_offset(dst_l, op)?) };
        let src_ptr = unsafe { src.ptr_at(contiguous_offset(src_l, op)?) };

        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];

        let (grid, block) = launch_config(dev, left_sz * right_sz);
        unsafe {
            launch_kernel(
                dev,
                &kernels::INDEXING,
                &func_name,
                grid,
                block,
                &mut [
                    (&ids_ptr) as *const *mut c_void as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                    &left_sz as *const usize as *mut c_void,
                    &src_dim_sz as *const usize as *mut c_void,
                    &dst_dim_sz as *const usize as *mut c_void,
                    &right_sz as *const usize as *mut c_void,
                ],
            )?;
        }
        Ok(())
    }
}

/// `index_add`: accumulates `src` into `dst` at the positions named by a 1-D
/// `ids`, which is why `IA_OP` needs `ids_dim_size` passed in.
pub(super) struct IndexAdd<'a>(pub &'a RocmStorage, pub &'a Layout, pub usize);

impl Map2InPlace for IndexAdd<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        dst: &mut SendSyncDeviceMemory<T>,
        dst_l: &Layout,
        src: &SendSyncDeviceMemory<T>,
        src_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<()> {
        let ids_l = self.1;
        let dim = self.2;
        let ids_o1 = contiguous_offset(ids_l, "index-add")?;
        let (name, ids_ptr) = ids_prefix_and_ptr("ia", &self.0.slice, ids_o1, "index-add")?;
        let func_name = try_kernel_name::<T>(&name)?;

        let dst_ptr = unsafe { dst.ptr_at(contiguous_offset(dst_l, "index-add")?) };
        let src_ptr = unsafe { src.ptr_at(contiguous_offset(src_l, "index-add")?) };

        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];

        let (grid, block) = launch_config(dev, left_sz * right_sz);
        unsafe {
            launch_kernel(
                dev,
                &kernels::INDEXING,
                &func_name,
                grid,
                block,
                &mut [
                    (&ids_ptr) as *const *mut c_void as *mut c_void,
                    &ids_dim_sz as *const usize as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                    &left_sz as *const usize as *mut c_void,
                    &src_dim_sz as *const usize as *mut c_void,
                    &dst_dim_sz as *const usize as *mut c_void,
                    &right_sz as *const usize as *mut c_void,
                ],
            )?;
        }
        Ok(())
    }
}
