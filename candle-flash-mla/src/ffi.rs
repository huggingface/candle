use core::ffi::{c_int, c_void};

use candle::cuda::cudarc::driver::sys::CUstream;

#[repr(C)]
pub struct FlashFwdMlaParams {
    pub b: c_int,
    pub seqlen_q: c_int,
    pub d: c_int,
    pub d_v: c_int,
    pub h: c_int,
    pub h_h_k_ratio: c_int,
    pub ngroups: c_int,
    pub is_causal: bool,
    pub scale_softmax: f32,
    pub scale_softmax_log2: f32,
    pub cu_seqlens_k: *mut c_int,

    pub q_ptr: *mut c_void,
    pub k_ptr: *mut c_void,
    pub v_ptr: *mut c_void,
    pub o_ptr: *mut c_void,
    pub softmax_lse_ptr: *mut c_void,

    pub q_batch_stride: i64,
    pub k_batch_stride: i64,
    pub v_batch_stride: i64,
    pub o_batch_stride: i64,
    pub q_row_stride: i64,
    pub k_row_stride: i64,
    pub v_row_stride: i64,
    pub o_row_stride: i64,
    pub q_head_stride: i64,
    pub k_head_stride: i64,
    pub v_head_stride: i64,
    pub o_head_stride: i64,

    pub block_table: *mut c_int,
    pub block_table_batch_stride: i64,
    pub page_block_size: c_int,

    pub tile_scheduler_metadata_ptr: *mut c_int,
    pub num_sm_parts: c_int,
    pub num_splits_ptr: *mut c_int,

    pub softmax_lseaccum_ptr: *mut c_void,
    pub oaccum_ptr: *mut c_void,
}

pub const TILE_SCHEDULER_METADATA_SIZE: usize = 8;

extern "C" {
    pub(crate) fn get_mla_metadata(
        seqlens_k_ptr: *mut c_int,
        tile_scheduler_metadata_ptr: *mut c_int,
        num_splits_ptr: *mut c_int,
        batch_size: c_int,
        num_sm_parts: c_int,
        stream: CUstream,
    );

    pub(crate) fn mha_fwd_kvcache_mla(params: FlashFwdMlaParams, stream: CUstream);
}
