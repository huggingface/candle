use crate::{Result, Tensor};
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
            crate::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::CpuStorage::F64(vs) => self.asort(vs, layout),
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
        use crate::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
        };
        use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, Map1Any, WrapErr};
        use crate::{CudaDevice, WithDType};

        impl Map1Any for ArgSort {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &crate::Layout,
                _wrap: W,
            ) -> Result<S> {
                let slice = match layout.contiguous_offsets() {
                    None => crate::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let elem_count = layout.shape().elem_count();
                let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;
                let func = if self.asc {
                    dev.get_or_load_func(&kernel_name::<T>("asort_asc"), kernels::SORT)?
                } else {
                    dev.get_or_load_func(&kernel_name::<T>("asort_desc"), kernels::SORT)?
                };
                let ncols = self.last_dim;
                let nrows = elem_count / ncols;
                let ncols_pad = next_power_of_2(ncols);
                let params = (&slice, &dst, ncols as i32, ncols_pad as i32);
                let cfg = LaunchConfig {
                    grid_dim: (1, nrows as u32, 1),
                    block_dim: (ncols_pad as u32, 1, 1),
                    shared_mem_bytes: (ncols_pad * std::mem::size_of::<u32>()) as u32,
                };
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(S::U32(dst))
            }
        }

        use crate::backend::BackendStorage;
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
        _storage: &crate::MetalStorage,
        _layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        todo!()
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
}
