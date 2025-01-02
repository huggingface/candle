mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
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
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/b252072409e69c25f2b9d473cc534e49b24decd2/csrc/flash_attn/flash_api.cpp#L187
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
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
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

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
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            *alibi_slopes.device_ptr() as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
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
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
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
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_stride[0] as u32,
                /* alibi_slopes_batch_stride */ 0,
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
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* upadded_lse */ 0,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ self.softcap.unwrap_or(0f32),
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn"
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
        candle::bail!("no cpu support for flash-attn")
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
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention v2 layer.
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

/// Flash-attention v2 layer.
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
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
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

/// Flash-attention v2 layer.
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
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
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

/// Flash-attention v2 layer.
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
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors `k` and `v` with fewer heads
/// than `q`. The number of heads in `k` and `v` must be divisible by the number of heads in `q`.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Optional alibi slopes tensor with shape `(num_heads_q)`.
/// * `softmax_scale` - Scaling factor for the softmax operation.
/// * `window_size_left` - Optional limit on left attention to value tokens.
/// * `window_size_right` - Optional limit on right attention to value tokens.
/// * `softcap` - Gemma style softcap the attention logits before the softmax.
///
/// # Causal Mask
///
/// Setting `window_size_left=None` and `window_size_right=Some(0)` applies a causal mask to the result
/// of `Q @ K^T`.
///
/// # Returns
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        softcap: Some(softcap),
    };
    q.apply_op3(k, v, op)
}

struct FlashAttnVarLen {
    pub softmax_scale: f32,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttnVarLen {
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
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/csrc/flash_attn/flash_api.cpp#L327
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle::bail!("seqlens_q has to be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle::bail!("seqlens_k has to be contiguous"),
        };

        let q = q.as_cuda_slice::<f16>()?;
        let k = k.as_cuda_slice::<f16>()?;
        let v = v.as_cuda_slice::<f16>()?;
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

        if q_rank != 3 || k_rank != 3 || v_rank != 3 {
            candle::bail!(
                "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
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

        let (total_q, num_heads, head_size_og) = q_l.shape().dims3()?;
        let (total_k, num_heads_k, _head_size_og) = k_l.shape().dims3()?;
        let expected_kv = (total_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims3()? {
            candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims3()? {
            candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        if nseqlens_q < 2 {
            candle::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
        }
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_k != nseqlens_q {
            candle::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
        }

        let batch_size = nseqlens_q - 1;

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
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            *alibi_slopes.device_ptr() as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(self.max_seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(self.max_seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        let softmax_lse = dev.alloc_zeros::<f32>(num_heads * total_q).w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = self.max_seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = self.max_seqlen_k as i32;
        }

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            let seqlens_q_ptr = *seqlens_q.device_ptr() as *const core::ffi::c_int;
            let seqlens_k_ptr = *seqlens_k.device_ptr() as *const core::ffi::c_int;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ seqlens_q_ptr,
                /* cu_seqlens_k_ptr */ seqlens_k_ptr,
                /* q_batch_stride */ 0,
                /* k_batch_stride */ 0,
                /* v_batch_stride */ 0,
                /* o_batch_stride */ 0,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ batch_size as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ self.max_seqlen_q as u32,
                /* seqlen_k */ self.max_seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* upadded_lse */ 1,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ self.softcap.unwrap_or(0.0),
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttnVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen"
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
        candle::bail!("no cpu support for flash-attn")
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
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Option, alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Option, limit left attention to value tokens.
/// * `window_size_right` - Option, limit right attention to value tokens.
/// * `softcap` - Gemma style softcap the attention logits before the softmax.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: f32,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: alibi_slopes.cloned(),
        window_size_left,
        window_size_right,
        softcap: Some(softcap),
    };
    q.apply_op3(k, v, op)
}
