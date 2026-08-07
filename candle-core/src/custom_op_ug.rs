//! [`UgIOp1`], an in-place unary op backed by a `ug` micro-kernel.
//!
//! Split out of `custom_op.rs`, which is at its size cap, and because each
//! backend needs its own compilation path: CUDA goes through nvrtc, Metal
//! through the shader compiler, and ROCm through `hipcc`.

use crate::custom_op::InplaceOp1;
use crate::{CpuStorage, Layout, Result};

#[cfg(feature = "cuda")]
use crate::CudaStorage;
#[cfg(feature = "metal")]
use crate::MetalStorage;
#[cfg(feature = "rocm")]
use crate::RocmStorage;

pub struct UgIOp1 {
    name: &'static str,
    #[cfg(feature = "cuda")]
    func: cudarc::driver::CudaFunction,
    #[cfg(feature = "metal")]
    func: candle_metal_kernels::metal::ComputePipeline,
    #[cfg(feature = "rocm")]
    func: rocm_rs::hip::Function,
    /// The geometry `ug` lowered the kernel for.
    ///
    /// The CUDA path above derives its own from the element count, which is only
    /// `LaunchConfig::for_num_elems` restated and does not always agree with
    /// what the kernel was lowered for — the `exp` sample lowers to a serial
    /// loop over the whole tensor, i.e. grid 1 / block 1, and launching it over
    /// a grid of `n / 32` would have every block redo the loop and race the
    /// others on the same buffer. `ug`'s own runtimes launch this config, so so
    /// does this one.
    #[cfg(feature = "rocm")]
    cfg: candle_ug::lang::LaunchConfig,
}

impl UgIOp1 {
    #[allow(unused)]
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn new(
        name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
        device: &crate::Device,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let device = device.as_cuda_device()?;
            let func = device.compile(name, kernel)?;
            Ok(Self {
                name,
                func: func.into_cuda_function(),
            })
        }
        #[cfg(feature = "metal")]
        {
            let device = device.as_metal_device()?;
            let func = device.compile(name, kernel)?;
            Ok(Self { name, func })
        }
        #[cfg(feature = "rocm")]
        {
            let device = device.as_rocm_device()?;
            let cfg = *kernel.launch_config();
            let func = device.compile(name, kernel)?;
            Ok(Self { name, func, cfg })
        }
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "rocm")))]
        {
            Ok(Self { name })
        }
    }
}

impl InplaceOp1 for UgIOp1 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        crate::bail!("ug ops are only supported on metal/cuda/rocm at the moment")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, sto: &mut MetalStorage, layout: &Layout) -> Result<()> {
        use crate::backend::BackendStorage;
        use objc2_metal;

        let elem_count = layout.shape().elem_count();
        if sto.dtype() != crate::DType::F32 {
            // TODO: support more dtypes.
            crate::bail!("input is not a f32 tensor")
        }
        let device = sto.device();
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.func);
        candle_metal_kernels::debug_group!(encoder, "{}", self.name);
        let (g, b) = if elem_count.is_multiple_of(32) {
            (elem_count / 32, 32)
        } else {
            (elem_count, 1)
        };
        let grid_dims = objc2_metal::MTLSize {
            width: g,
            height: 1,
            depth: 1,
        };
        let group_dims = candle_metal_kernels::utils::get_block_dims(b, 1, 1);
        encoder.set_output_buffer(0, Some(sto.buffer()), 0);
        encoder.dispatch_threads(grid_dims, group_dims);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, sto: &mut CudaStorage, layout: &Layout) -> Result<()> {
        use crate::cuda_backend::WrapErr;
        use cudarc::driver::PushKernelArg;

        let elem_count = layout.shape().elem_count();
        let stream = sto.device.cuda_stream();
        // TODO: support more dtypes.
        let sto = sto.as_cuda_slice::<f32>()?;
        let sto = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, o2)) => sto.slice(o1..o2),
        };
        let (g, b) = if elem_count % 32 == 0 {
            (elem_count / 32, 32)
        } else {
            (elem_count, 1)
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (g as u32, 1, 1),
            block_dim: (b as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.func);
        builder.arg(&sto);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(&self, sto: &mut RocmStorage, layout: &Layout) -> Result<()> {
        use crate::rocm_backend::{rocm_error, RocmStorageSlice};

        // TODO: support more dtypes.
        let RocmStorageSlice::F32(slice) = &sto.slice else {
            crate::bail!("input is not a f32 tensor")
        };
        let offset = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, _)) => o1,
        };
        let (grid, block) = (self.cfg.grid_dim, self.cfg.block_dim);
        // hipModuleLaunchKernel rejects a zero dimension with a bare error code;
        // say which kernel and which dimension instead.
        if grid == 0 || block == 0 {
            return Err(rocm_error(format!(
                "ug kernel {}: lowered to an empty launch config (grid {grid}, block {block})",
                self.name
            )));
        }

        // SAFETY: `contiguous_offsets` has just confirmed `offset` addresses an
        // element of this allocation.
        let ptr = unsafe { slice.ptr_at(offset) };
        let mut args = [&ptr as *const _ as *mut std::ffi::c_void];
        self.func
            .launch(
                rocm_rs::hip::Dim3::from(grid),
                rocm_rs::hip::Dim3::from(block),
                self.cfg.shared_mem,
                Some(sto.device.stream()),
                &mut args,
            )
            .map_err(|e| rocm_error(format!("ug kernel {} launch failed: {e}", self.name)))?;
        Ok(())
    }
}
