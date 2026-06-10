use crate::backend::BackendStorage;
use crate::{Result, Tensor};

#[cfg(feature = "cuda")]
thread_local! {
    static TOPK_TMP: std::cell::RefCell<
        std::collections::HashMap<(crate::cuda_backend::DeviceId, usize), TopkTmpBufs>,
    > =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(feature = "cuda")]
struct TopkTmpBufs {
    vals: crate::cuda_backend::cudarc::driver::CudaSlice<f32>,
    idx: crate::cuda_backend::cudarc::driver::CudaSlice<u32>,
    cap_elems: usize,
}
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Vec<u32> {
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
        sort_indexes
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda_backend::cudarc::driver::{
        CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits,
    };
    use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, WrapErr};
    use crate::{CudaDevice, WithDType};

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
            let dst = unsafe { dev.alloc::<u32>(elem_count)? };
            let func = if self.asc {
                dev.get_or_load_func(&kernel_name::<T>("asort_asc"), &kernels::SORT)?
            } else {
                dev.get_or_load_func(&kernel_name::<T>("asort_desc"), &kernels::SORT)?
            };
            let ncols = self.last_dim;
            let nrows = elem_count / ncols;
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
            crate::CpuStorage::U8(vs) => self.asort(vs, layout),
            crate::CpuStorage::U32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I16(vs) => self.asort(vs, layout),
            crate::CpuStorage::I32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::CpuStorage::F64(vs) => self.asort(vs, layout),
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout),
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
        let dst = device.new_buffer(el, DType::U32, "asort")?;
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

    pub fn topk_indices(&self, k: usize) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "topk_indices" });
        }
        if self.rank() != 1 {
            crate::bail!("topk_indices expects a 1D tensor")
        }
        if k == 0 {
            crate::bail!("topk_indices expects k > 0")
        }
        if k > 64 {
            crate::bail!("topk_indices currently supports k <= 64")
        }
        let n = self.dims1()?;
        if k > n {
            crate::bail!("topk_indices expects k <= n")
        }
        self.apply_op1_no_bwd(&TopKIndices { k })
    }
}

#[derive(Debug, Clone, Copy)]
struct TopKIndices {
    k: usize,
}

impl crate::CustomOp1 for TopKIndices {
    fn name(&self) -> &'static str {
        "topk_indices"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("topk_indices requires contiguous layout")
        }
        let k = self.k;
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        let end = start + n;

        let mut pairs: Vec<(f32, u32)> = match storage {
            crate::CpuStorage::F32(vs) => vs[start..end]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i as u32))
                .collect(),
            _ => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(storage.dtype(), "topk_indices").bt(),
                )
            }
        };

        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater));

        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Ok((crate::CpuStorage::U32(out), crate::Shape::from((k,))))
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use crate::cuda_backend::{kernel_name, CudaDType, WrapErr};

            if !layout.is_contiguous() {
                crate::bail!("topk_indices requires contiguous layout")
            }
            let k = self.k as u32;
            let n = layout.shape().elem_count() as u32;
            let dev = &storage.device;

            let x = storage.as_cuda_slice::<f32>()?;
            let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
                crate::Error::Msg("topk_indices requires contiguous offsets".into()).bt()
            })?;
            let x = x.slice(o1..o2);

            let block_dim1 = 128u32;
            let block_dim2 = 128u32;
            let items_per_block = (block_dim1 as usize) * 8;
            let mut grid = ((n as usize) + items_per_block - 1) / items_per_block;
            grid = grid.clamp(1, 1024);
            let grid_dim = grid as u32;
            let shared1 = (block_dim1 as usize)
                * (self.k)
                * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
            let shared2 = (block_dim2 as usize)
                * (self.k)
                * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

            let cap_elems = grid * self.k;
            let dev_id = dev.id();
            let (tmp_vals, tmp_idx) = TOPK_TMP.with(|cell| -> Result<_> {
                let mut map = cell.borrow_mut();
                match map.get_mut(&(dev_id, self.k)) {
                    Some(bufs) if bufs.cap_elems >= cap_elems => {
                        Ok((bufs.vals.clone(), bufs.idx.clone()))
                    }
                    _ => {
                        let vals = unsafe { dev.alloc::<f32>(cap_elems)? };
                        let idx = unsafe { dev.alloc::<u32>(cap_elems)? };
                        map.insert(
                            (dev_id, self.k),
                            TopkTmpBufs {
                                vals: vals.clone(),
                                idx: idx.clone(),
                                cap_elems,
                            },
                        );
                        Ok((vals, idx))
                    }
                }
            })?;

            let out_idx = unsafe { dev.alloc::<u32>(self.k)? };

            let items_per_block_u32 = items_per_block as u32;
            let f1 = dev.get_or_load_func(
                &kernel_name::<f32>("topk_stage1"),
                &crate::cuda_backend::kernels::SORT,
            )?;
            let stream = dev.cuda_stream();
            let mut b1 = stream.launch_builder(&f1);
            b1.arg(&x)
                .arg(&n)
                .arg(&k)
                .arg(&items_per_block_u32)
                .arg(&tmp_vals)
                .arg(&tmp_idx);
            unsafe {
                b1.launch(LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim1, 1, 1),
                    shared_mem_bytes: shared1 as u32,
                })
            }
            .w()?;

            let m = (grid_dim * k) as u32;
            let f2 = dev.get_or_load_func(
                &kernel_name::<f32>("topk_stage2"),
                &crate::cuda_backend::kernels::SORT,
            )?;
            let stream = dev.cuda_stream();
            let mut b2 = stream.launch_builder(&f2);
            b2.arg(&tmp_vals)
                .arg(&tmp_idx)
                .arg(&m)
                .arg(&k)
                .arg(&out_idx);
            unsafe {
                b2.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_dim2, 1, 1),
                    shared_mem_bytes: shared2 as u32,
                })
            }
            .w()?;

            let dst = <u32 as CudaDType>::wrap_cuda_slice(out_idx, dev.clone());
            Ok((dst, crate::Shape::from((self.k,))))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = storage;
            let _ = layout;
            Err(crate::Error::NotCompiledWithCudaSupport.bt())
        }
    }
}
