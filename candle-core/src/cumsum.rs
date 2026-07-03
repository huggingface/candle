use crate::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor, WithDType};
use num_traits::Zero;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Cumsum {
    dim: usize,
}

impl Cumsum {
    pub(crate) fn new(dim: usize) -> Self {
        Self { dim }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn is_cuda_dtype_supported(dtype: DType) -> bool {
        matches!(dtype, DType::F32 | DType::F64 | DType::U32 | DType::I64)
    }

    pub(crate) fn is_supported(t: &Tensor, dim: usize) -> bool {
        #[cfg(not(feature = "cuda"))]
        let _ = dim;
        if matches!(
            t.dtype(),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0
        ) {
            return false;
        }
        match &*t.storage() {
            Storage::Cpu(_) => true,
            #[cfg(feature = "cuda")]
            // This is the exact CUDA kernel contract. Tensor::cumsum may adapt
            // other CUDA layouts into this shape before calling the CustomOp.
            Storage::Cuda(_) => {
                Self::is_cuda_dtype_supported(t.dtype())
                    && t.is_contiguous()
                    && dim + 1 == t.rank()
                    && t.shape().elem_count() > 0
                    && t.dims().last().copied().unwrap_or(0) > 0
            }
            #[cfg(not(feature = "cuda"))]
            Storage::Cuda(_) => false,
            Storage::Metal(_) => false,
        }
    }

    // The CPU path honors arbitrary input layouts and returns contiguous storage.
    fn cpu_cumsum<T: WithDType + Zero>(src: &[T], layout: &Layout, dim: usize) -> Vec<T> {
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let rank = dims.len();
        if elem_count == 0 {
            return Vec::new();
        }
        if rank == 0 {
            return vec![src[layout.start_offset()]];
        }

        let mut dst = vec![T::zero(); elem_count];
        let mut dst_stride = vec![1; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            dst_stride[i] = dst_stride[i + 1] * dims[i + 1];
        }
        let outer_count = elem_count / dims[dim];
        for outer_idx in 0..outer_count {
            let mut rem = outer_idx;
            let mut src_base = layout.start_offset();
            let mut dst_base = 0usize;
            for d in (0..rank).rev() {
                if d == dim {
                    continue;
                }
                let coord = rem % dims[d];
                rem /= dims[d];
                src_base += coord * layout.stride()[d];
                dst_base += coord * dst_stride[d];
            }

            let mut acc = T::zero();
            for i in 0..dims[dim] {
                acc += src[src_base + i * layout.stride()[dim]];
                dst[dst_base + i * dst_stride[dim]] = acc;
            }
        }
        dst
    }
}

impl crate::CustomOp1 for Cumsum {
    fn name(&self) -> &'static str {
        "cumsum"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dst = match storage {
            CpuStorage::U8(v) => CpuStorage::U8(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::U32(v) => CpuStorage::U32(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::I16(v) => CpuStorage::I16(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::I32(v) => CpuStorage::I32(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::I64(v) => CpuStorage::I64(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::BF16(v) => CpuStorage::BF16(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::F16(v) => CpuStorage::F16(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::F32(v) => CpuStorage::F32(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::F64(v) => CpuStorage::F64(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::F8E4M3(v) => CpuStorage::F8E4M3(Self::cpu_cumsum(v, layout, self.dim)),
            CpuStorage::F6E2M3(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(DType::F6E2M3, "cumsum").bt())
            }
            CpuStorage::F6E3M2(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(DType::F6E3M2, "cumsum").bt())
            }
            CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(DType::F4, "cumsum").bt())
            }
            CpuStorage::F8E8M0(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(DType::F8E8M0, "cumsum").bt())
            }
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use crate::cuda_backend::{kernel_name, kernels, CudaDevice, CudaStorageSlice, WrapErr};

        // Tile size is also the dispatch threshold: smaller rows avoid the extra
        // multi-kernel launch overhead, larger rows get row-level parallelism.
        const MULTI_BLOCK_TILE_SIZE: usize = 4096;
        const MULTI_BLOCK_THREADS: usize = 256;
        // Keep the first version bounded; larger rows remain correct via the
        // single-block fallback and can use recursive tile-sum scan later.
        const MAX_TILES_PER_ROW: usize = 1024;

        fn launch_single_block<T>(
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &Layout,
            start: usize,
            end: usize,
        ) -> Result<CudaSlice<T>>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            let elem_count = layout.shape().elem_count();
            let last_dim = layout.dims().last().copied().unwrap_or(1);
            let rows = elem_count / last_dim;
            let dst = unsafe { dev.alloc::<T>(elem_count)? };
            let block_dim = last_dim.next_power_of_two().min(1024);
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (block_dim as u32, 1, 1),
                shared_mem_bytes: (block_dim * std::mem::size_of::<T>()) as u32,
            };
            let func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_last_dim"), &kernels::CUMSUM)?;
            let src = src.slice(start..end);
            let mut builder = func.builder();
            let last_dim = last_dim as i32;
            let block_dim = block_dim as i32;
            builder.arg(&src);
            builder.arg(&dst);
            builder.arg(&last_dim);
            builder.arg(&block_dim);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        // First pass writes local prefix sums into dst and one total per tile into
        // block_sums. The scanned tile totals are then added back as per-tile offsets.
        fn launch_multi_block<T>(
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &Layout,
            start: usize,
            end: usize,
        ) -> Result<CudaSlice<T>>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            let elem_count = layout.shape().elem_count();
            let last_dim = layout.dims().last().copied().unwrap_or(1);
            let rows = elem_count / last_dim;
            let ntiles = last_dim.div_ceil(MULTI_BLOCK_TILE_SIZE);
            if ntiles > MAX_TILES_PER_ROW {
                return launch_single_block(src, dev, layout, start, end);
            }

            let dst = unsafe { dev.alloc::<T>(elem_count)? };
            let block_sums = unsafe { dev.alloc::<T>(rows * ntiles)? };
            let scanned_block_sums = unsafe { dev.alloc::<T>(rows * ntiles)? };
            let src = src.slice(start..end);
            let tile_func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_tile"), &kernels::CUMSUM)?;
            let tile_cfg = LaunchConfig {
                grid_dim: (rows as u32, ntiles as u32, 1),
                block_dim: (MULTI_BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: (MULTI_BLOCK_THREADS * std::mem::size_of::<T>()) as u32,
            };
            let mut builder = tile_func.builder();
            let last_dim_i32 = last_dim as i32;
            let tile_size_i32 = MULTI_BLOCK_TILE_SIZE as i32;
            let ntiles_i32 = ntiles as i32;
            let threads_i32 = MULTI_BLOCK_THREADS as i32;
            builder.arg(&src);
            builder.arg(&dst);
            builder.arg(&block_sums);
            builder.arg(&last_dim_i32);
            builder.arg(&tile_size_i32);
            builder.arg(&ntiles_i32);
            builder.arg(&threads_i32);
            unsafe { builder.launch(tile_cfg) }.w()?;

            // Scan the per-tile sums once per row. The first multi-block version
            // keeps this bounded to one CUDA block per row.
            let scan_threads = ntiles.next_power_of_two().min(1024);
            let scan_func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_last_dim"), &kernels::CUMSUM)?;
            let scan_cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (scan_threads as u32, 1, 1),
                shared_mem_bytes: (scan_threads * std::mem::size_of::<T>()) as u32,
            };
            let mut builder = scan_func.builder();
            let ntiles_i32 = ntiles as i32;
            let scan_threads_i32 = scan_threads as i32;
            builder.arg(&block_sums);
            builder.arg(&scanned_block_sums);
            builder.arg(&ntiles_i32);
            builder.arg(&scan_threads_i32);
            unsafe { builder.launch(scan_cfg) }.w()?;

            let add_func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_add_offsets"), &kernels::CUMSUM)?;
            let add_cfg = LaunchConfig {
                grid_dim: (rows as u32, ntiles as u32, 1),
                block_dim: (MULTI_BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = add_func.builder();
            builder.arg(&dst);
            builder.arg(&scanned_block_sums);
            builder.arg(&last_dim_i32);
            builder.arg(&tile_size_i32);
            builder.arg(&ntiles_i32);
            unsafe { builder.launch(add_cfg) }.w()?;
            Ok(dst)
        }

        fn launch<T>(
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &Layout,
            start: usize,
            end: usize,
        ) -> Result<CudaSlice<T>>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            let last_dim = layout.dims().last().copied().unwrap_or(1);
            if last_dim > MULTI_BLOCK_TILE_SIZE {
                launch_multi_block(src, dev, layout, start, end)
            } else {
                launch_single_block(src, dev, layout, start, end)
            }
        }

        if !layout.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "cumsum" }.bt());
        }
        if self.dim + 1 != layout.dims().len() {
            crate::bail!("cuda cumsum only supports the last dimension")
        }

        let dev = storage.device();
        let (start, end) = layout.contiguous_offsets().unwrap();
        let slice = match &storage.slice {
            CudaStorageSlice::F32(src) => {
                CudaStorageSlice::F32(launch(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::F64(src) => {
                CudaStorageSlice::F64(launch(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::U32(src) => {
                CudaStorageSlice::U32(launch(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::I64(src) => {
                CudaStorageSlice::I64(launch(src, dev, layout, start, end)?)
            }
            _ => return Err(crate::Error::UnsupportedDTypeForOp(storage.dtype(), "cumsum").bt()),
        };
        let dst = crate::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let grad = grad_res
            .flip(&[self.dim])?
            .cumsum(self.dim)?
            .flip(&[self.dim])?;
        Ok(Some(grad))
    }
}
