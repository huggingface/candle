use crate::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor, WithDType};
use num_traits::Zero;

#[cfg(any(feature = "cuda", test))]
fn cuda_grid_rows(rows: usize) -> Result<u32> {
    // CUDA grid.x is limited to 2^31 - 1 blocks.
    if rows > i32::MAX as usize {
        crate::bail!("cuda cumsum has too many rows: {rows}")
    }
    Ok(rows as u32)
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, PartialEq, Eq)]
struct CudaTileGrid {
    rows: u32,
    tasks: usize,
    blocks: u32,
    row_reciprocal: u32,
    two_tiles: bool,
}

#[cfg(any(feature = "cuda", test))]
impl CudaTileGrid {
    fn with_two_tile_order(mut self, ntiles: usize, multiprocessors: usize) -> Self {
        // Interleave each row's two tiles only while all tile tasks fit in
        // one block per SM. This heuristic is measured on GB10; SM counts
        // come from the device, never a fixed GPU-specific row threshold.
        self.two_tiles = ntiles == 2
            && self.rows > 1
            && self.rows as usize <= multiprocessors / 2
            && self.tasks == self.blocks as usize;
        self
    }

    fn kernel_name<T: WithDType>(&self, name: &str) -> String {
        let suffix = if self.tasks > self.blocks as usize {
            "_grid_stride"
        } else if self.rows == 1 {
            // Still one block per tile and three passes; only coordinate
            // decoding is specialized. Keep the wide grid-stride path first.
            "_single_row"
        } else if self.two_tiles {
            "_two_tiles"
        } else {
            ""
        };
        let dtype = T::DTYPE.as_str();
        format!("{name}{suffix}_{dtype}")
    }
}

#[cfg(any(feature = "cuda", test))]
fn cuda_tile_grid(rows: usize, ntiles: usize) -> Result<CudaTileGrid> {
    let rows = cuda_grid_rows(rows)?;
    let Some(tasks) = (rows as usize).checked_mul(ntiles) else {
        crate::bail!("cuda cumsum tile task count overflows: {rows} rows, {ntiles} tiles")
    };
    if tasks == 0 {
        crate::bail!("cuda cumsum requires nonempty tile tasks")
    }
    // Flatten task = tile * rows + row, preserving the existing block order.
    // Kernels stride over tasks above grid.x capacity, keeping three passes.
    let blocks = tasks.min(i32::MAX as usize) as u32;
    let row_reciprocal = if tasks <= blocks as usize && rows > 1 {
        (1u64 << 32).div_ceil(rows as u64) as u32
    } else {
        0
    };
    Ok(CudaTileGrid {
        rows,
        tasks,
        blocks,
        row_reciprocal,
        two_tiles: false,
    })
}

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
        matches!(
            dtype,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::U32 | DType::I64
        )
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
    fn cpu_cumsum<T: WithDType + Zero>(
        src: &[T],
        layout: &Layout,
        dim: usize,
        add: impl Fn(T, T) -> T,
    ) -> Vec<T> {
        Self::cpu_cumsum_with_acc(
            src,
            layout,
            dim,
            std::convert::identity,
            std::convert::identity,
            add,
        )
    }

    fn cpu_cumsum_with_acc<T: WithDType + Zero, A: WithDType + Zero>(
        src: &[T],
        layout: &Layout,
        dim: usize,
        to_acc: impl Fn(T) -> A,
        to_dst: impl Fn(A) -> T,
        add: impl Fn(A, A) -> A,
    ) -> Vec<T> {
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

            let mut acc = A::zero();
            for i in 0..dims[dim] {
                acc = add(acc, to_acc(src[src_base + i * layout.stride()[dim]]));
                dst[dst_base + i * dst_stride[dim]] = to_dst(acc);
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
            CpuStorage::U8(v) => {
                CpuStorage::U8(Self::cpu_cumsum(v, layout, self.dim, u8::wrapping_add))
            }
            CpuStorage::U32(v) => {
                CpuStorage::U32(Self::cpu_cumsum(v, layout, self.dim, u32::wrapping_add))
            }
            CpuStorage::I16(v) => {
                CpuStorage::I16(Self::cpu_cumsum(v, layout, self.dim, i16::wrapping_add))
            }
            CpuStorage::I32(v) => {
                CpuStorage::I32(Self::cpu_cumsum(v, layout, self.dim, i32::wrapping_add))
            }
            CpuStorage::I64(v) => {
                CpuStorage::I64(Self::cpu_cumsum(v, layout, self.dim, i64::wrapping_add))
            }
            // Keep low-precision rounding out of the running prefix sum.
            CpuStorage::BF16(v) => CpuStorage::BF16(Self::cpu_cumsum_with_acc(
                v,
                layout,
                self.dim,
                half::bf16::to_f32,
                half::bf16::from_f32,
                std::ops::Add::add,
            )),
            CpuStorage::F16(v) => CpuStorage::F16(Self::cpu_cumsum_with_acc(
                v,
                layout,
                self.dim,
                half::f16::to_f32,
                half::f16::from_f32,
                std::ops::Add::add,
            )),
            CpuStorage::F32(v) => {
                CpuStorage::F32(Self::cpu_cumsum(v, layout, self.dim, std::ops::Add::add))
            }
            CpuStorage::F64(v) => {
                CpuStorage::F64(Self::cpu_cumsum(v, layout, self.dim, std::ops::Add::add))
            }
            CpuStorage::F8E4M3(v) => CpuStorage::F8E4M3(Self::cpu_cumsum_with_acc(
                v,
                layout,
                self.dim,
                |value| value.to_f32(),
                float8::F8E4M3::from_f32,
                std::ops::Add::add,
            )),
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

        fn launch_single_block<T, A>(
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &Layout,
            start: usize,
            end: usize,
        ) -> Result<CudaSlice<T>>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
            A: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            let elem_count = layout.shape().elem_count();
            let last_dim = layout.dims().last().copied().unwrap_or(1);
            let rows = cuda_grid_rows(elem_count / last_dim)?;
            let mut dst = unsafe { dev.alloc::<T>(elem_count)? };
            let block_dim = last_dim.min(1024).next_power_of_two();
            let cfg = LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (block_dim as u32, 1, 1),
                shared_mem_bytes: (block_dim * std::mem::size_of::<A>()) as u32,
            };
            let func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_last_dim"), &kernels::CUMSUM)?;
            let src = src.slice(start..end);
            let mut builder = func.builder();
            let block_dim = block_dim as i32;
            builder.arg(&src);
            builder.arg(&mut dst);
            builder.arg(&last_dim);
            builder.arg(&block_dim);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        fn scan_block_totals<T>(
            totals: &CudaSlice<T>,
            scanned: &mut CudaSlice<T>,
            dev: &CudaDevice,
            rows: u32,
            ntiles: usize,
        ) -> Result<()>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            // A thread scans multiple totals when there are more than 1024 tiles.
            let threads = ntiles.min(1024).next_power_of_two();
            let func =
                dev.get_or_load_func(&kernel_name::<T>("cumsum_last_dim"), &kernels::CUMSUM)?;
            let cfg = LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: (threads * std::mem::size_of::<T>()) as u32,
            };
            let mut builder = func.builder();
            let threads = threads as i32;
            builder.arg(totals);
            builder.arg(scanned);
            builder.arg(&ntiles);
            builder.arg(&threads);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(())
        }

        fn launch_multi_block_lowp<T>(
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
            let last_dim = *layout.dims().last().unwrap();
            let ntiles = last_dim.div_ceil(MULTI_BLOCK_TILE_SIZE);
            let grid = cuda_tile_grid(elem_count / last_dim, ntiles)?
                .with_two_tile_order(ntiles, dev.multiprocessor_count());
            let mut dst = unsafe { dev.alloc::<T>(elem_count)? };
            let mut totals = unsafe { dev.alloc::<f32>(grid.tasks)? };
            let mut scanned = unsafe { dev.alloc::<f32>(grid.tasks)? };
            let src = src.slice(start..end);
            let cfg = LaunchConfig {
                grid_dim: (grid.blocks, 1, 1),
                block_dim: (MULTI_BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_func(
                &grid.kernel_name::<T>("cumsum_tile_totals"),
                &kernels::CUMSUM,
            )?;
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&mut totals);
            builder.arg(&last_dim);
            builder.arg(&ntiles);
            builder.arg(&grid.tasks);
            builder.arg(&grid.rows);
            builder.arg(&grid.row_reciprocal);
            unsafe { builder.launch(cfg) }.w()?;

            scan_block_totals(&totals, &mut scanned, dev, grid.rows, ntiles)?;

            let func =
                dev.get_or_load_func(&grid.kernel_name::<T>("cumsum_rescan"), &kernels::CUMSUM)?;
            let mut builder = func.builder();
            // The final pass reads src again: pass the tracked view, keeping
            // temporary layout copies alive until this read has completed.
            builder.arg(&src);
            builder.arg(&mut dst);
            builder.arg(&scanned);
            builder.arg(&last_dim);
            builder.arg(&ntiles);
            builder.arg(&grid.tasks);
            builder.arg(&grid.rows);
            builder.arg(&grid.row_reciprocal);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        // First pass writes local prefix sums and one total per tile into
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
            let ntiles = last_dim.div_ceil(MULTI_BLOCK_TILE_SIZE);
            let grid = cuda_tile_grid(elem_count / last_dim, ntiles)?
                .with_two_tile_order(ntiles, dev.multiprocessor_count());
            let rows = grid.rows;

            let mut dst = unsafe { dev.alloc::<T>(elem_count)? };
            let mut block_sums = unsafe { dev.alloc::<T>(grid.tasks)? };
            let mut scanned_block_sums = unsafe { dev.alloc::<T>(grid.tasks)? };
            let src = src.slice(start..end);
            let tile_func =
                dev.get_or_load_func(&grid.kernel_name::<T>("cumsum_tile"), &kernels::CUMSUM)?;
            let tile_cfg = LaunchConfig {
                grid_dim: (grid.blocks, 1, 1),
                block_dim: (MULTI_BLOCK_THREADS as u32, 1, 1),
                // Stage one uses CUB static shared storage.
                shared_mem_bytes: 0,
            };
            let mut builder = tile_func.builder();
            let threads_i32 = MULTI_BLOCK_THREADS as i32;
            builder.arg(&src);
            builder.arg(&mut dst);
            builder.arg(&mut block_sums);
            builder.arg(&last_dim);
            builder.arg(&MULTI_BLOCK_TILE_SIZE);
            builder.arg(&ntiles);
            builder.arg(&threads_i32);
            builder.arg(&grid.tasks);
            builder.arg(&grid.rows);
            builder.arg(&grid.row_reciprocal);
            unsafe { builder.launch(tile_cfg) }.w()?;

            scan_block_totals(&block_sums, &mut scanned_block_sums, dev, rows, ntiles)?;

            let add_func = dev.get_or_load_func(
                &grid.kernel_name::<T>("cumsum_add_offsets"),
                &kernels::CUMSUM,
            )?;
            let add_cfg = LaunchConfig {
                grid_dim: (grid.blocks, 1, 1),
                block_dim: (MULTI_BLOCK_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = add_func.builder();
            builder.arg(&mut dst);
            builder.arg(&scanned_block_sums);
            builder.arg(&last_dim);
            builder.arg(&MULTI_BLOCK_TILE_SIZE);
            builder.arg(&ntiles);
            builder.arg(&grid.tasks);
            builder.arg(&grid.rows);
            builder.arg(&grid.row_reciprocal);
            unsafe { builder.launch(add_cfg) }.w()?;
            Ok(dst)
        }

        fn launch<T, A>(
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &Layout,
            start: usize,
            end: usize,
        ) -> Result<CudaSlice<T>>
        where
            T: DeviceRepr + ValidAsZeroBits + WithDType,
            A: DeviceRepr + ValidAsZeroBits + WithDType,
        {
            let last_dim = layout.dims().last().copied().unwrap_or(1);
            if last_dim > MULTI_BLOCK_TILE_SIZE {
                if T::DTYPE != A::DTYPE {
                    launch_multi_block_lowp(src, dev, layout, start, end)
                } else {
                    launch_multi_block(src, dev, layout, start, end)
                }
            } else {
                launch_single_block::<T, A>(src, dev, layout, start, end)
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
            CudaStorageSlice::F16(src) => {
                CudaStorageSlice::F16(launch::<_, f32>(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::BF16(src) => {
                CudaStorageSlice::BF16(launch::<_, f32>(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::F32(src) => {
                CudaStorageSlice::F32(launch::<_, f32>(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::F64(src) => {
                CudaStorageSlice::F64(launch::<_, f64>(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::U32(src) => {
                CudaStorageSlice::U32(launch::<_, u32>(src, dev, layout, start, end)?)
            }
            CudaStorageSlice::I64(src) => {
                CudaStorageSlice::I64(launch::<_, i64>(src, dev, layout, start, end)?)
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

#[cfg(test)]
mod tests {
    use super::{cuda_grid_rows, cuda_tile_grid, CudaTileGrid};

    #[test]
    fn cuda_two_tile_order_follows_sm_capacity() {
        for sms in [0, 1, 2, 3, 16, 48, 80, 128] {
            for rows in [1, 2, 3, 23, 24, 25, 39, 40, 41, 64, 65] {
                for tiles in [1, 2, 3, 65536] {
                    let grid = cuda_tile_grid(rows, tiles)
                        .unwrap()
                        .with_two_tile_order(tiles, sms);
                    let expected = if rows == 1 {
                        "cumsum_tile_single_row_f32"
                    } else if tiles == 2 && rows <= sms / 2 {
                        "cumsum_tile_two_tiles_f32"
                    } else {
                        "cumsum_tile_f32"
                    };
                    assert_eq!(grid.kernel_name::<f32>("cumsum_tile"), expected);
                }
            }
        }
        let grid = cuda_tile_grid(i32::MAX as usize, 2)
            .unwrap()
            .with_two_tile_order(2, usize::MAX);
        assert_eq!(
            grid.kernel_name::<f32>("cumsum_tile"),
            "cumsum_tile_grid_stride_f32"
        );
    }

    #[test]
    fn cuda_tile_grid_selects_single_row_entry() {
        for tiles in [2, 1025, 65535, 65536, i32::MAX as usize] {
            let grid = cuda_tile_grid(1, tiles).unwrap();
            assert_eq!(
                grid.kernel_name::<half::f16>("cumsum_tile_totals"),
                "cumsum_tile_totals_single_row_f16"
            );
            assert_eq!(
                grid.kernel_name::<f64>("cumsum_add_offsets"),
                "cumsum_add_offsets_single_row_f64"
            );
        }
        let grid = cuda_tile_grid(1, i32::MAX as usize + 1).unwrap();
        assert_eq!(
            grid.kernel_name::<half::bf16>("cumsum_rescan"),
            "cumsum_rescan_grid_stride_bf16"
        );
        assert_eq!(
            cuda_tile_grid(2, 2)
                .unwrap()
                .kernel_name::<u32>("cumsum_tile"),
            "cumsum_tile_u32"
        );
    }

    #[test]
    fn cuda_tile_grid_flattens_tile_and_row_boundaries() {
        for rows in [1, 2, 65535, 65536] {
            for tiles in [1023, 1024, 1025, 2049, 65535, 65536, 65537] {
                let tasks = rows * tiles;
                assert_eq!(
                    cuda_tile_grid(rows, tiles).unwrap(),
                    CudaTileGrid {
                        rows: rows as u32,
                        tasks,
                        blocks: tasks.min(i32::MAX as usize) as u32,
                        row_reciprocal: if tasks <= i32::MAX as usize && rows > 1 {
                            (1u64 << 32).div_ceil(rows as u64) as u32
                        } else {
                            0
                        },
                        two_tiles: false,
                    }
                );
            }
        }
    }

    #[test]
    fn cuda_tile_grid_caps_blocks_for_grid_stride() {
        let limit = i32::MAX as usize;
        for tasks in [limit - 1, limit, limit + 1, 2 * limit + 1] {
            let grid = cuda_tile_grid(1, tasks).unwrap();
            assert_eq!(grid.tasks, tasks);
            assert_eq!(grid.blocks, tasks.min(limit) as u32);
            assert_eq!(
                grid.kernel_name::<f32>("cumsum_tile"),
                if tasks <= limit {
                    "cumsum_tile_single_row_f32"
                } else {
                    "cumsum_tile_grid_stride_f32"
                }
            );
        }
        assert!(cuda_tile_grid(2, usize::MAX).is_err());
        assert!(cuda_tile_grid(0, 1).is_err());
        assert!(cuda_tile_grid(1, 0).is_err());
    }

    #[test]
    fn cuda_row_reciprocal_decodes_task_boundaries() {
        let limit = i32::MAX as usize;
        for rows in [
            1,
            2,
            3,
            7,
            31,
            32,
            33,
            1025,
            65535,
            65536,
            65537,
            limit - 1,
            limit,
        ] {
            let grid = cuda_tile_grid(rows, 1).unwrap();
            let mut state = 0x12345678u32;
            let mut tasks = vec![0, 1, rows - 1, rows, limit - 1];
            for _ in 0..1000 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                tasks.push(state as usize % limit);
            }
            for task in tasks {
                let row = if rows == 1 {
                    task
                } else {
                    let row = ((task as u64 * grid.row_reciprocal as u64) >> 32) as usize;
                    row - usize::from(row * rows > task)
                };
                assert_eq!(row, task / rows, "task {task}, rows {rows}");
                assert_eq!(task - row * rows, task % rows);
            }
        }
    }

    #[test]
    fn cuda_grid_checks_rows_before_narrowing() {
        let max_rows = i32::MAX as usize;
        assert_eq!(cuda_grid_rows(max_rows).unwrap(), i32::MAX as u32);
        assert_eq!(
            cuda_tile_grid(max_rows, 1025).unwrap().rows,
            max_rows as u32
        );
        assert!(cuda_tile_grid(max_rows + 1, 1025).is_err());
        assert!(cuda_tile_grid(max_rows + 1, 65536).is_err());
        assert!(cuda_grid_rows(usize::MAX).is_err());
    }
}
