mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, Error, Layout, Result, Shape, Tensor};
use half::f16;

pub struct FlashHdim32Sm80 {
    pub softmax_scale: f32,
    pub causal: bool,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

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
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<f16>()?;
        let k = k.as_cuda_slice::<f16>()?;
        let v = v.as_cuda_slice::<f16>()?;

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;
        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        let softmax_lse = dev.alloc_zeros::<f32>(b_sz * num_heads * seqlen_q).w()?;

        let causal = if self.causal { 1 } else { 0 };

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_stride[0] as u32,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ b_sz as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_causal */ causal,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

pub fn flash_attn(softmax_scale: f32, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    q.custom_op3(
        k,
        v,
        FlashHdim32Sm80 {
            softmax_scale,
            causal: true,
        },
    )
}
