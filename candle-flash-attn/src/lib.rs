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
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/b252072409e69c25f2b9d473cc534e49b24decd2/csrc/flash_attn/flash_api.cpp#L187
        let dev = q.device();
        let out_shape = Shape::from(&[1]);
        let q = q.as_cuda_slice::<f16>()?;
        let k = k.as_cuda_slice::<f16>()?;
        let v = v.as_cuda_slice::<f16>()?;

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            Err(Error::Wrapped("the last dim of q must be contiguous".into()).bt())?
        }
        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* q_row_stride */ q_stride[q_rank - 3] as u32,
                /* k_row_stride */ k_stride[k_rank - 3] as u32,
                /* v_row_stride  */ v_stride[v_rank - 3] as u32,
                /* q_head_stride */ q_stride[q_rank - 2] as u32,
                /* k_head_stride */ k_stride[k_rank - 2] as u32,
                /* v_head_stride */ v_stride[v_rank - 2] as u32,
                /* b */ 1,
                /* h */ 1,
                /* k */ 1,
                /* d */ 1,
                /* d_rounded */ 1,
                /* softmax_scale*/ 1.0,
                /* seqlen_q */ 1,
                /* seqlen_k */ 1,
                /* seqlen_q_rounded */ 1,
                /* seqlen_k_rounded */ 1,
                /* is_causal */ 1,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}
