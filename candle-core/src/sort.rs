use crate::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Result<Vec<u32>> {
        let vs = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, o2)) => &vs[o1..o2],
        };
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        Ok(sort_indexes)
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda_backend::cudarc::driver::{
        CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, ValidAsZeroBits,
    };
    use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, WrapErr};
    use crate::{CudaDevice, DType, WithDType};

    const SHARED_ARGSORT_MAX_ITEMS: usize = 8192;
    const LARGE_ARGSORT_MAX_BATCH_ROWS: usize = 256;
    const LARGE_ARGSORT_AUX_BUDGET: usize = 64 * 1024 * 1024;

    fn checked_device_offset(ptr: u64, elements: usize, element_size: usize) -> Result<u64> {
        let byte_offset = elements
            .checked_mul(element_size)
            .ok_or_else(|| crate::Error::msg("CUDA argsort byte offset overflow"))?;
        ptr.checked_add(byte_offset as u64)
            .ok_or_else(|| crate::Error::msg("CUDA argsort device pointer overflow"))
    }

    fn check_runtime_status(status: i32, action: &str) -> Result<()> {
        if status != 0 {
            crate::bail!("CUDA argsort {action} failed with runtime error {status}")
        }
        Ok(())
    }

    fn large_f32_workspace_bytes(
        nrows: usize,
        ncols: i32,
        descending: bool,
        stream: *mut std::ffi::c_void,
    ) -> Result<usize> {
        let nrows = i32::try_from(nrows)
            .map_err(|_| crate::Error::msg("CUDA argsort batch has too many rows"))?;
        let mut temp_storage_bytes = 0usize;
        let status = unsafe {
            candle_kernels::ffi::candle_argsort_f32(
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut temp_storage_bytes,
                nrows,
                ncols,
                i32::from(descending),
                stream,
            )
        };
        check_runtime_status(status, "workspace query")?;
        Ok(temp_storage_bytes)
    }

    fn large_f32_argsort<T, P>(
        src: &P,
        dst: &mut CudaSlice<u32>,
        dev: &CudaDevice,
        nrows: usize,
        ncols: usize,
        descending: bool,
    ) -> Result<()>
    where
        T: DeviceRepr + WithDType,
        P: DevicePtr<T>,
    {
        if T::DTYPE != DType::F32 {
            return Err(crate::Error::UnsupportedDTypeForOp(
                T::DTYPE,
                "CUDA arg_sort_last_dim with more than 8192 elements",
            )
            .bt());
        }
        let ncols_i32 =
            i32::try_from(ncols).map_err(|_| crate::Error::msg("CUDA argsort row is too large"))?;
        let rows_by_item_count = (i32::MAX as usize / ncols).max(1);
        let mut max_batch_rows = nrows
            .min(LARGE_ARGSORT_MAX_BATCH_ROWS)
            .min(rows_by_item_count);

        let stream = dev.cuda_stream();
        let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
        let temp_storage_bytes = loop {
            let max_temp =
                large_f32_workspace_bytes(max_batch_rows, ncols_i32, descending, stream_ptr)?;
            let tail_rows = nrows % max_batch_rows;
            let tail_temp = if tail_rows == 0 {
                0
            } else {
                large_f32_workspace_bytes(tail_rows, ncols_i32, descending, stream_ptr)?
            };
            let temp_bytes = max_temp.max(tail_temp);
            let direct_items = max_batch_rows
                .checked_mul(ncols)
                .ok_or_else(|| crate::Error::msg("CUDA argsort item count overflow"))?;
            let direct_bytes = direct_items
                .checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<u32>())
                .and_then(|bytes| {
                    bytes.checked_add((max_batch_rows + 1) * std::mem::size_of::<i32>())
                })
                .ok_or_else(|| crate::Error::msg("CUDA argsort workspace size overflow"))?;
            let total_bytes = direct_bytes
                .checked_add(temp_bytes)
                .ok_or_else(|| crate::Error::msg("CUDA argsort workspace size overflow"))?;
            if total_bytes <= LARGE_ARGSORT_AUX_BUDGET || max_batch_rows == 1 {
                break temp_bytes;
            }
            let scaled_rows = ((max_batch_rows as u128 * LARGE_ARGSORT_AUX_BUDGET as u128)
                / total_bytes as u128) as usize;
            max_batch_rows = scaled_rows.clamp(1, max_batch_rows - 1);
        };

        let max_batch_items = max_batch_rows
            .checked_mul(ncols)
            .ok_or_else(|| crate::Error::msg("CUDA argsort item count overflow"))?;

        let mut keys_out = unsafe { dev.alloc::<f32>(max_batch_items)? };
        let mut indices_in = unsafe { dev.alloc::<u32>(max_batch_items)? };
        let mut offsets = unsafe { dev.alloc::<i32>(max_batch_rows + 1)? };
        let (src_ptr, _src_guard) = src.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
        let (keys_out_ptr, _keys_out_guard) = keys_out.device_ptr_mut(&stream);
        let (indices_in_ptr, _indices_in_guard) = indices_in.device_ptr_mut(&stream);
        let (offsets_ptr, _offsets_guard) = offsets.device_ptr_mut(&stream);

        let mut temp_storage = unsafe { dev.alloc::<u8>(temp_storage_bytes.max(1))? };
        let (temp_storage_ptr, _temp_storage_guard) = temp_storage.device_ptr_mut(&stream);

        let mut row_start = 0usize;
        while row_start < nrows {
            let batch_rows = (nrows - row_start).min(max_batch_rows);
            let batch_items = batch_rows * ncols;
            let item_offset = row_start * ncols;
            let batch_src_ptr =
                checked_device_offset(src_ptr, item_offset, std::mem::size_of::<f32>())?;
            let batch_dst_ptr =
                checked_device_offset(dst_ptr, item_offset, std::mem::size_of::<u32>())?;
            let mut launch_temp_storage_bytes = temp_storage_bytes;
            let status = unsafe {
                candle_kernels::ffi::candle_argsort_f32(
                    batch_src_ptr as *const f32,
                    keys_out_ptr as *mut f32,
                    indices_in_ptr as *mut u32,
                    batch_dst_ptr as *mut u32,
                    offsets_ptr as *mut i32,
                    temp_storage_ptr as *mut std::ffi::c_void,
                    &mut launch_temp_storage_bytes,
                    i32::try_from(batch_rows)
                        .map_err(|_| crate::Error::msg("CUDA argsort batch has too many rows"))?,
                    ncols_i32,
                    i32::from(descending),
                    stream_ptr,
                )
            };
            check_runtime_status(status, "launch")?;
            row_start += batch_rows;

            debug_assert!(batch_items <= max_batch_items);
        }
        Ok(())
    }

    impl crate::cuda_backend::Map1Any for ArgSort {
        fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
            &self,
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &crate::Layout,
            _wrap: W,
        ) -> Result<S> {
            use cudarc::driver::PushKernelArg;

            let slice = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let mut dst = unsafe { dev.alloc::<u32>(elem_count)? };
            let ncols = self.last_dim;
            if elem_count == 0 {
                return Ok(S::U32(dst));
            }
            let nrows = elem_count / ncols;
            if ncols > SHARED_ARGSORT_MAX_ITEMS {
                large_f32_argsort(&slice, &mut dst, dev, nrows, ncols, !self.asc)?;
                return Ok(S::U32(dst));
            }
            let func = if self.asc {
                dev.get_or_load_func(&kernel_name::<T>("asort_asc"), &kernels::SORT)?
            } else {
                dev.get_or_load_func(&kernel_name::<T>("asort_desc"), &kernels::SORT)?
            };
            let ncols_pad = next_power_of_2(ncols);
            // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
            let block_dim = ncols_pad.min(1024);
            let cfg = LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (block_dim as u32, 1, 1),
                shared_mem_bytes: (ncols_pad * std::mem::size_of::<u32>()) as u32,
            };
            let stream = dev.cuda_stream();
            let mut builder = stream.launch_builder(&func);
            let ncols = ncols as i32;
            let ncols_pad = ncols_pad as i32;
            builder.arg(&slice).arg(&dst).arg(&ncols).arg(&ncols_pad);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(S::U32(dst))
        }
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::U32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout)?,
            // Dummy types don't support sorting
            crate::CpuStorage::F6E2M3(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E2M3, "argsort").bt(),
                )
            }
            crate::CpuStorage::F6E3M2(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E3M2, "argsort").bt(),
                )
            }
            crate::CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt())
            }
            crate::CpuStorage::F8E8M0(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F8E8M0, "argsort").bt(),
                )
            }
        };
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::Map1Any;
        let dev = storage.device();
        let slice = self.map(&storage.slice, dev, layout)?;
        let dst = crate::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I16 => "asort_asc_i16",
                    DType::I32 => "asort_asc_i32",
                    DType::I64 => "asort_asc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I16 => "asort_desc_i16",
                    DType::I32 => "asort_desc_i32",
                    DType::I64 => "asort_desc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_encoder = device.command_encoder()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device
            .new_buffer_builder()
            .with_size_for(el, DType::U32)
            .with_label("asort")
            .build()?;
        let mut ncols_pad = 1;
        while ncols_pad < ncols {
            ncols_pad *= 2;
        }
        candle_metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_encoder,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::Error::wrap)?;
        let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}
