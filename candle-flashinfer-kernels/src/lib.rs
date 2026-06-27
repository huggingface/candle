//! Optional, feature-gated decode-attention backend for the autoregressive decode path,
//! alongside `candle-flash-attn`. See https://github.com/huggingface/candle/issues/3651.
//!
//! Unlike `candle-flash-attn`, which targets the prefill (multi-token) case, this crate
//! targets the decode case: each sequence in the batch contributes exactly one new query
//! token, attending over an arbitrarily long key/value cache. The kernel here is a reference,
//! numerically-stable implementation of that workload (the one FlashInfer's batch-decode
//! kernels optimize); it is not a port of FlashInfer's own tensor-core/split-KV kernels.
mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

pub struct DecodeAttention {
    pub softmax_scale: f32,
}

impl DecodeAttention {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        dtype_tag: i32,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device();
        let (b_sz, num_heads, head_dim) = q_l.shape().dims3()?;
        let (b_sz_k, num_heads_k, seqlen_k, head_dim_k) = k_l.shape().dims4()?;
        if k_l.shape() != v_l.shape() {
            candle::bail!(
                "shape mismatch between k {:?} and v {:?}",
                k_l.shape(),
                v_l.shape()
            );
        }
        if b_sz_k != b_sz || head_dim_k != head_dim {
            candle::bail!(
                "shape mismatch between q {:?} and k {:?}",
                q_l.shape(),
                k_l.shape()
            );
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!(
                "number of kv heads {num_heads_k} must divide the number of query heads {num_heads}"
            )
        }

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        if q_stride[2] != 1 {
            candle::bail!("the head dimension of q must be contiguous {q_stride:?}")
        }
        if k_stride[3] != 1 {
            candle::bail!("the head dimension of k must be contiguous {k_stride:?}")
        }
        if v_stride[3] != 1 {
            candle::bail!("the head dimension of v must be contiguous {v_stride:?}")
        }

        let out_shape = q_l.shape().clone();
        let elem_count = out_shape.elem_count();
        let stream = dev.cuda_stream();
        let mut dst = unsafe { dev.alloc::<T>(elem_count)? };

        let q = q.as_cuda_slice::<T>()?.slice(q_l.start_offset()..);
        let k = k.as_cuda_slice::<T>()?.slice(k_l.start_offset()..);
        let v = v.as_cuda_slice::<T>()?.slice(v_l.start_offset()..);

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_decode_attention(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                dtype_tag,
                b_sz as i32,
                num_heads as i32,
                num_heads_k as i32,
                seqlen_k as i32,
                head_dim as i32,
                q_stride[0] as i64,
                q_stride[1] as i64,
                k_stride[0] as i64,
                k_stride[1] as i64,
                k_stride[2] as i64,
                v_stride[0] as i64,
                v_stride[1] as i64,
                v_stride[2] as i64,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                self.softmax_scale,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for DecodeAttention {
    fn name(&self) -> &'static str {
        "flashinfer-decode-attention"
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
        candle::bail!("no cpu support for flashinfer-decode-attention")
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
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l, k, k_l, v, v_l, 0),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, 1),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, 2),
            dt => candle::bail!("flashinfer-decode-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// Single-token decode attention: `softmax(q @ k^T * softmax_scale) @ v`, where each sequence
/// in the batch contributes exactly one query token attending over its key/value cache.
///
/// Grouped-query attention is supported: `k`/`v` may use fewer heads than `q`, as long as
/// `num_heads_q` is a multiple of `num_heads_kv`.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, num_heads_q, head_dim)`, i.e. one token per sequence.
/// * `k` - Key cache with shape `(batch, num_heads_kv, seqlen_k, head_dim)`.
/// * `v` - Value cache with shape `(batch, num_heads_kv, seqlen_k, head_dim)`.
///
/// The resulting tensor has shape `(batch, num_heads_q, head_dim)`.
pub fn flashinfer_decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let op = DecodeAttention { softmax_scale };
    q.apply_op3(k, v, op)
}
