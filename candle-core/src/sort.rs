use crate::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort;

fn asort<T: crate::WithDType>(vs: &[T], layout: &crate::Layout) -> Result<Vec<u32>> {
    #[allow(clippy::uninit_vec)]
    // Safety: indexes are set later in the parallelized section.
    let mut sort_indexes = unsafe {
        let el_count = layout.shape().elem_count();
        let mut v = Vec::with_capacity(el_count);
        v.set_len(el_count);
        v
    };
    let last_dim = match layout.dims().last() {
        None => crate::bail!("empty last-dim in arg-sort"),
        Some(last_dim) => *last_dim,
    };
    sort_indexes
        .par_chunks_exact_mut(last_dim)
        .zip(vs.par_chunks_exact(last_dim))
        .for_each(|(indexes, vs)| {
            indexes.sort_by(|&i, &j| {
                vs[i as usize]
                    .partial_cmp(&vs[j as usize])
                    .unwrap_or(std::cmp::Ordering::Greater)
            })
        });
    Ok(sort_indexes)
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
            crate::CpuStorage::U8(vs) => asort(vs, layout),
            crate::CpuStorage::U32(vs) => asort(vs, layout),
            crate::CpuStorage::I64(vs) => asort(vs, layout),
            crate::CpuStorage::BF16(vs) => asort(vs, layout),
            crate::CpuStorage::F16(vs) => asort(vs, layout),
            crate::CpuStorage::F32(vs) => asort(vs, layout),
            crate::CpuStorage::F64(vs) => asort(vs, layout),
        }?;
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _storage: &crate::CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        todo!()
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

impl Tensor {
    pub fn arg_sort_last_dim(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort)
    }
}
