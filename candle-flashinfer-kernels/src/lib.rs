//! Optional, feature-gated decode-attention backend for the autoregressive decode path,
//! alongside `candle-flash-attn`. See https://github.com/huggingface/candle/issues/3651.
//!
//! Unlike `candle-flash-attn`, which targets the prefill (multi-token) case, this crate
//! targets the decode case: each sequence in the batch contributes exactly one new query
//! token, attending over an arbitrarily long key/value cache. The kernel here is a reference,
//! numerically-stable implementation of that workload (the one FlashInfer's batch-decode
//! kernels optimize); it is not a port of FlashInfer's own tensor-core/split-KV kernels.
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
use candle::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use candle::DType;
use candle::{CpuStorage, Layout, Result, Shape, Tensor};
use half::{bf16, f16};
use rayon::prelude::*;

pub struct DecodeAttention {
    pub softmax_scale: f32,
}

/// Reference CPU decode attention: `softmax(q @ k^T * scale) @ v` for a single query token
/// per sequence, with grouped-query attention (`num_heads` a multiple of `num_heads_kv`).
///
/// Computes in f32 regardless of the storage type for numerical stability, honoring the
/// arbitrary strides/offsets of the input layouts. The independent `(batch, head)` outputs
/// are computed in parallel with rayon, which matters on the many-core CPUs (incl. Apple
/// Silicon) this serves as a fallback for.
fn cpu_decode_attention<T: Copy + Send + Sync>(
    softmax_scale: f32,
    q: &[T],
    q_l: &Layout,
    k: &[T],
    k_l: &Layout,
    v: &[T],
    v_l: &Layout,
    to_f32: impl Fn(T) -> f32 + Sync,
    from_f32: impl Fn(f32) -> T + Sync,
) -> Result<(Vec<T>, Shape)> {
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
    let group = num_heads / num_heads_k;
    let (q_o, q_s) = (q_l.start_offset(), q_l.stride());
    let (k_o, k_s) = (k_l.start_offset(), k_l.stride());
    let (v_o, v_s) = (v_l.start_offset(), v_l.stride());

    let mut out = vec![from_f32(0f32); b_sz * num_heads * head_dim];
    // Each `(batch, head)` writes a disjoint `head_dim`-sized slice of `out`, so they can run
    // concurrently. Chunk index `c` maps back to `bi = c / num_heads`, `h = c % num_heads`.
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(c, out_chunk)| {
            let bi = c / num_heads;
            let h = c % num_heads;
            let h_kv = h / group;
            // scores[l] = scale * dot(q[bi, h], k[bi, h_kv, l]), tracking the max for a
            // numerically-stable softmax.
            let mut scores = vec![0f32; seqlen_k];
            let mut max = f32::NEG_INFINITY;
            for (l, score) in scores.iter_mut().enumerate() {
                let mut dot = 0f32;
                for i in 0..head_dim {
                    let qv = to_f32(q[q_o + bi * q_s[0] + h * q_s[1] + i * q_s[2]]);
                    let kv = to_f32(k[k_o + bi * k_s[0] + h_kv * k_s[1] + l * k_s[2] + i * k_s[3]]);
                    dot += qv * kv;
                }
                let s = dot * softmax_scale;
                *score = s;
                if s > max {
                    max = s;
                }
            }
            let mut denom = 0f32;
            for s in scores.iter_mut() {
                let e = (*s - max).exp();
                *s = e;
                denom += e;
            }
            let inv = if denom > 0f32 { 1f32 / denom } else { 0f32 };
            // out[bi, h, i] = sum_l softmax[l] * v[bi, h_kv, l, i]
            for (i, out_i) in out_chunk.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (l, &p) in scores.iter().enumerate() {
                    let vv = to_f32(v[v_o + bi * v_s[0] + h_kv * v_s[1] + l * v_s[2] + i * v_s[3]]);
                    acc += p * vv;
                }
                *out_i = from_f32(acc * inv);
            }
        });
    Ok((out, (b_sz, num_heads, head_dim).into()))
}

impl DecodeAttention {
    #[cfg(feature = "cuda")]
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
        q: &CpuStorage,
        q_l: &Layout,
        k: &CpuStorage,
        k_l: &Layout,
        v: &CpuStorage,
        v_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let scale = self.softmax_scale;
        match (q, k, v) {
            (CpuStorage::F32(q), CpuStorage::F32(k), CpuStorage::F32(v)) => {
                let (out, shape) =
                    cpu_decode_attention(scale, q, q_l, k, k_l, v, v_l, |x| x, |x| x)?;
                Ok((CpuStorage::F32(out), shape))
            }
            (CpuStorage::F16(q), CpuStorage::F16(k), CpuStorage::F16(v)) => {
                let (out, shape) = cpu_decode_attention(
                    scale,
                    q,
                    q_l,
                    k,
                    k_l,
                    v,
                    v_l,
                    |x: f16| x.to_f32(),
                    f16::from_f32,
                )?;
                Ok((CpuStorage::F16(out), shape))
            }
            (CpuStorage::BF16(q), CpuStorage::BF16(k), CpuStorage::BF16(v)) => {
                let (out, shape) = cpu_decode_attention(
                    scale,
                    q,
                    q_l,
                    k,
                    k_l,
                    v,
                    v_l,
                    |x: bf16| x.to_f32(),
                    bf16::from_f32,
                )?;
                Ok((CpuStorage::BF16(out), shape))
            }
            _ => candle::bail!(
                "flashinfer-decode-attention cpu: q/k/v must share dtype (f32/f16/bf16)"
            ),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        metal::decode_attention_metal_fwd(self, q, q_l, k, k_l, v, v_l)
    }

    #[cfg(feature = "cuda")]
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
            dt => candle::bail!(
                "flashinfer-decode-attention is only supported for f32/f16/bf16 ({dt:?})"
            ),
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
