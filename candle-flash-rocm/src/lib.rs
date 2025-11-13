//! Flash Attention for ROCm backend
//!
//! Mirrors candle-flash-attn but uses AMD's Composable Kernel (CK) implementation
//! 
//! Created by: TEAM-509 (ROCm Flash Attention implementation)
//! CUDA parity: candle-flash-attn/src/lib.rs

mod ffi;

use candle::backend::BackendStorage;
use candle::rocm_backend::rocm_rs::hip::DevicePtr;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

pub struct FlashAttn {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

impl FlashAttn {
    fn rocm_fwd_t<
        T: candle::rocm_backend::RocmDType + candle::rocm_backend::rocm_rs::hip::DeviceRepr,
    >(
        &self,
        q: &candle::RocmStorage,
        q_l: &Layout,
        k: &candle::RocmStorage,
        k_l: &Layout,
        v: &candle::RocmStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle::RocmStorage, Shape)> {
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_hip_slice::<T>()?;
        let k = k.as_hip_slice::<T>()?;
        let v = v.as_hip_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

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
        if head_size_og % 8 != 0 {
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let stream = dev.hip_stream();
        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle::Storage::Rocm(r) => r.as_hip_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a rocm tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            let (ptr, _guard) = alibi_slopes.device_ptr(&stream);
            ptr as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count)? };
        let softmax_lse = dev.alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr(&stream);
            let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr(&stream);
            ffi::run_mha_rocm(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                softmax_lse_ptr as *const core::ffi::c_void,
                alibi_slopes_ptr,
                std::ptr::null(),
                std::ptr::null(),
                q_stride[0] as u32,
                k_stride[0] as u32,
                v_stride[0] as u32,
                o_stride[0] as u32,
                0,
                q_stride[q_rank - 3] as u32,
                k_stride[k_rank - 3] as u32,
                v_stride[v_rank - 3] as u32,
                o_stride[o_rank - 3] as u32,
                q_stride[q_rank - 2] as u32,
                k_stride[k_rank - 2] as u32,
                v_stride[v_rank - 2] as u32,
                o_stride[o_rank - 2] as u32,
                b_sz as u32,
                num_heads as u32,
                num_heads_k as u32,
                head_size as u32,
                head_size_rounded as u32,
                self.softmax_scale,
                seqlen_q as u32,
                seqlen_k as u32,
                seqlen_q_rounded as u32,
                seqlen_k_rounded as u32,
                is_bf16,
                is_causal,
                0,
                window_size_left,
                window_size_right,
                self.softcap.unwrap_or(0f32),
            )
        }

        let dst = candle::RocmStorage::wrap_rocm_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn-rocm"
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
        candle::bail!("no cpu support for flash-attn-rocm")
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        q: &candle::RocmStorage,
        q_l: &Layout,
        k: &candle::RocmStorage,
        k_l: &Layout,
        v: &candle::RocmStorage,
        v_l: &Layout,
    ) -> Result<(candle::RocmStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.rocm_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.rocm_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn-rocm is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention v2 layer for ROCm (AMD GPUs).
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 with windowing for ROCm.
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 with ALiBi for ROCm.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    alibi_slopes: &Tensor,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}
