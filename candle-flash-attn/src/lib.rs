mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, Error, Layout, Result, Shape};
use half::f16;

pub struct FlashHdim32Sm80;

impl candle::CustomOp3 for FlashHdim32Sm80 {
    fn name(&self) -> &'static str {
        "flash-hdim32-sm80"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Wrapped("no cpu support for flash-attn".into()))
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        _q_l: &Layout,
        k: &candle::CudaStorage,
        _k_l: &Layout,
        v: &candle::CudaStorage,
        _v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device();
        let out_shape = Shape::from(&[1]);
        let q = q.as_cuda_slice::<f16>()?;
        let k = k.as_cuda_slice::<f16>()?;
        let v = v.as_cuda_slice::<f16>()?;
        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            ffi::run_mha(
                q_ptr, k_ptr, v_ptr, dst_ptr, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 1, 1,
                1, 1, 1,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}
