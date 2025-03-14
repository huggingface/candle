mod ffi;

use std::f32;

use candle::backend::BackendStorage;
use candle::cuda::cudarc;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::bf16;

pub struct FlashAttn {
    pub softmax_scale: f32,
    pub block_table: Tensor,
    pub cache_seqlens: Tensor,
    pub head_size_v: usize,
    pub seqlen_q_ori: usize,
    pub ngroups: usize,
    pub num_heads_per_head_k: usize,
}

impl FlashAttn {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k_c_k_pe_cache: &candle::CudaStorage,
        k_c_k_pe_cache_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device();
        let (b_sz, seqlen_q, num_heads, head_size_q) = q_l.shape().dims4()?;

        let out_shape = Shape::from_dims(&[b_sz, seqlen_q, num_heads, self.head_size_v]);
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<T>()?;
        let k_c_k_pe_cache = k_c_k_pe_cache.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k_c_k_pe_cache = k_c_k_pe_cache.slice(k_c_k_pe_cache_l.start_offset()..);

        let v_l = k_c_k_pe_cache_l;
        let v = &k_c_k_pe_cache;

        let q_stride = q_l.stride();
        let k_stride = k_c_k_pe_cache_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank})"
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

        if self.block_table.dtype() != DType::I32 {
            candle::bail!("block_table must be i32");
        }

        if self.block_table.stride()[self.block_table.stride().len() - 1] != 1 {
            candle::bail!("block_table must have contiguous last dim");
        }

        let max_num_blocks_per_seq = self.block_table.dim(1)?;
        let num_blocks = k_c_k_pe_cache_l.dim(0)?;
        let page_block_size = k_c_k_pe_cache_l.dim(1)?;
        let num_heads_k = k_c_k_pe_cache_l.dim(2)?;

        if head_size_q % 8 != 0 {
            candle::bail!("only supports q/k head sizes that are a multiple of 8")
        }
        if self.head_size_v % 32 != 0 {
            candle::bail!("only supports v head sizes that are a multiple of 32")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let candle::Storage::Cuda(block_table) = &*self.block_table.storage_and_layout().0 else {
            candle::bail!("block_table must be CUDA")
        };
        let block_table = block_table
            .as_cuda_slice::<i32>()?
            .slice(self.block_table.layout().start_offset()..);

        let candle::Storage::Cuda(cache_seqlens) = &*self.cache_seqlens.storage_and_layout().0
        else {
            candle::bail!("cache_seqlens must be CUDA")
        };
        let cache_seqlens = cache_seqlens
            .as_cuda_slice::<i32>()?
            .slice(self.cache_seqlens.layout().start_offset()..);

        let is_causal = self.seqlen_q_ori != 1;

        let num_heads = num_heads_k;
        let head_size_k = head_size_q;

        if q_l.dims() != [b_sz, seqlen_q, num_heads, head_size_q] {
            candle::bail!(
                "Expected q shape {:?}, got {:?} instead.",
                [b_sz, seqlen_q, num_heads, head_size_q],
                q_l.dims()
            );
        }
        if k_c_k_pe_cache_l.dims() != [num_blocks, page_block_size, num_heads_k, head_size_k] {
            candle::bail!(
                "Expected k shape {:?}, got {:?} instead.",
                [num_blocks, page_block_size, num_heads_k, head_size_k],
                k_c_k_pe_cache_l.dims()
            );
        }
        if self.block_table.dims() != [b_sz, max_num_blocks_per_seq] {
            candle::bail!(
                "Expected block_table shape {:?}, got {:?} instead.",
                [b_sz, max_num_blocks_per_seq],
                self.block_table.dims()
            );
        }
        if self.cache_seqlens.dims() != [b_sz] {
            candle::bail!(
                "Expected cache_seqlens shape {:?}, got {:?} instead.",
                [b_sz],
                self.cache_seqlens.dims()
            );
        }

        // This should match the logic in the MLA kernel.
        let block_size_m = 64usize;
        let sm_count = dev
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .w()? as usize;
        let num_sm_parts = sm_count
            / num_heads_k
            / (self.seqlen_q_ori * self.num_heads_per_head_k).div_ceil(block_size_m);

        let tile_scheduler_metadata =
            unsafe { dev.alloc::<i32>(num_sm_parts * ffi::TILE_SCHEDULER_METADATA_SIZE) }.w()?;
        let num_splits = unsafe { dev.alloc::<i32>(b_sz + 1) }.w()?;

        unsafe {
            ffi::get_mla_metadata(
                (*cache_seqlens.device_ptr()) as *mut core::ffi::c_int,
                (*tile_scheduler_metadata.device_ptr()) as *mut core::ffi::c_int,
                (*num_splits.device_ptr()) as *mut core::ffi::c_int,
                b_sz as i32,
                num_sm_parts as i32,
                *dev.cu_stream(),
            );
        }

        let dst = unsafe { dev.alloc::<T>(b_sz * seqlen_q * num_heads * self.head_size_v) }.w()?;
        let softmax_lse = unsafe { dev.alloc::<f32>(b_sz * num_heads * seqlen_q) }.w()?;

        let dst_accum = unsafe {
            dev.alloc::<f32>((b_sz + num_sm_parts) * seqlen_q * num_heads * self.head_size_v)
        }
        .w()?;
        let softmax_lse_accum =
            unsafe { dev.alloc::<f32>((b_sz + num_sm_parts) * num_heads * seqlen_q) }.w()?;

        // Expect:
        if head_size_q != 576 {
            candle::bail!("Expected head_size_q to be 576, got {head_size_q}");
        }
        if self.head_size_v != 512 {
            candle::bail!("Expected head_size_v to be 512, got {}", self.head_size_v);
        }
        if page_block_size != 64 {
            candle::bail!("Expected page_block_size to be 64, got {page_block_size}");
        }

        let params = ffi::FlashFwdMlaParams {
            b: b_sz as i32,
            seqlen_q: seqlen_q as i32,
            cu_seqlens_k: (*cache_seqlens.device_ptr()) as *mut core::ffi::c_int,
            h: num_heads as i32,
            h_h_k_ratio: (num_heads / num_heads_k) as i32,
            ngroups: self.ngroups as i32,
            is_causal,
            d: head_size_q as i32,
            d_v: self.head_size_v as i32,
            scale_softmax: self.softmax_scale,
            scale_softmax_log2: self.softmax_scale * f32::consts::LOG2_E,
            q_ptr: (*q.device_ptr()) as *mut core::ffi::c_void,
            k_ptr: (*k_c_k_pe_cache.device_ptr()) as *mut core::ffi::c_void,
            v_ptr: (*v.device_ptr()) as *mut core::ffi::c_void,
            o_ptr: (*dst.device_ptr()) as *mut core::ffi::c_void,
            softmax_lse_ptr: (*softmax_lse.device_ptr()) as *mut core::ffi::c_void,
            q_batch_stride: q_stride[0] as i64,
            k_batch_stride: k_stride[0] as i64,
            v_batch_stride: v_stride[0] as i64,
            o_batch_stride: o_stride[0] as i64,
            q_row_stride: q_stride[q_stride.len() - 3] as i64,
            k_row_stride: k_stride[k_stride.len() - 3] as i64,
            v_row_stride: v_stride[v_stride.len() - 3] as i64,
            o_row_stride: o_stride[o_stride.len() - 3] as i64,
            q_head_stride: q_stride[q_stride.len() - 2] as i64,
            k_head_stride: k_stride[k_stride.len() - 2] as i64,
            v_head_stride: v_stride[v_stride.len() - 2] as i64,
            o_head_stride: o_stride[o_stride.len() - 2] as i64,
            block_table: (*block_table.device_ptr()) as *mut core::ffi::c_int,
            block_table_batch_stride: self.block_table.stride()[0] as i64,
            page_block_size: page_block_size as i32,
            tile_scheduler_metadata_ptr: (*tile_scheduler_metadata.device_ptr())
                as *mut core::ffi::c_int,
            num_sm_parts: num_sm_parts as i32,
            num_splits_ptr: (*num_splits.device_ptr()) as *mut core::ffi::c_int,
            oaccum_ptr: (*dst_accum.device_ptr()) as *mut core::ffi::c_void,
            softmax_lseaccum_ptr: (*softmax_lse_accum.device_ptr()) as *mut core::ffi::c_void,
        };

        unsafe { ffi::mha_fwd_kvcache_mla(params, *dev.cu_stream()) }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp2 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
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
        k_c_k_pe_cache: &candle::CudaStorage,
        k_c_k_pe_cache_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::BF16 => {
                self.cuda_fwd_t::<bf16>(q, q_l, k_c_k_pe_cache, k_c_k_pe_cache_l)
            }
            dt => candle::bail!("flash-mla is only supported for bf16 ({dt:?})"),
        }
    }
}

/// FlashMLA layer.
///
/// This implements MLA attention, `softmax(Q @ K^T . softmax_scale) @ V`.
///
/// # Arguments
///
/// * `q`: (batch_size, seq_len_q, num_heads_q, head_dim).
/// * `k_c_k_pe_cache`: (num_blocks, page_block_size, num_heads_k, head_dim).
/// * `block_table`: (batch_size, max_num_blocks_per_seq), i32.
/// * `cache_seqlens`: (batch_size), i32
/// * `softmax_scale: The scale of QK^T before applying softmax.
/// * `head_size_v`: v_head_dim in the config
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size_v)`.
pub fn flash_attn_mla(
    q: &Tensor,
    k_c_k_pe_cache: &Tensor,
    block_table: Tensor,
    cache_seqlens: Tensor,
    softmax_scale: f32,
    head_size_v: usize,
) -> Result<Tensor> {
    let (b_sz, seqlen_q_ori, num_heads, head_size) = q.shape().dims4()?;

    let num_heads_k = k_c_k_pe_cache.dim(2)?;
    let ngroups = num_heads / num_heads_k;

    let seqlen_q = seqlen_q_ori * ngroups;
    let num_heads_per_head_k = num_heads / num_heads_k;

    let q = q
        .reshape((b_sz, seqlen_q_ori, num_heads_k, ngroups, head_size))?
        .transpose(2, 3)?
        .reshape((b_sz, seqlen_q, num_heads_k, head_size))?;

    let op = FlashAttn {
        softmax_scale,
        block_table,
        cache_seqlens,
        head_size_v,
        seqlen_q_ori,
        ngroups,
        num_heads_per_head_k,
    };

    let out = q.apply_op2(k_c_k_pe_cache, op)?;

    out.reshape((b_sz, seqlen_q_ori, ngroups, num_heads_k, head_size_v))?
        .transpose(2, 3)?
        .reshape((b_sz, seqlen_q_ori, num_heads, head_size_v))
}
