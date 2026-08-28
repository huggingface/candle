//! Raw `extern "C"` declarations matching `csrc/candle_sycl.h`.
use std::ffi::{c_int, c_void};

#[repr(C)]
pub struct CandleSyclQueue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CandleSyclDeviceInfo {
    pub name: [i8; 256],
    pub global_mem_bytes: u64,
    pub max_compute_units: u32,
    pub max_clock_khz: u32,
    pub is_integrated: i32,
    pub supports_fp16: i32,
    pub supports_fp64: i32,
}

// The wrapper module passes `&crate::Layout` (a `#[repr(C)]` type with the same
// fields) where the C side expects `const CandleSyclLayout *`.
pub use crate::Layout as CLayout;

extern "C" {
    pub fn candle_sycl_device_count() -> c_int;
    pub fn candle_sycl_queue_new(ordinal: c_int) -> *mut CandleSyclQueue;
    pub fn candle_sycl_queue_free(q: *mut CandleSyclQueue);
    pub fn candle_sycl_queue_native(q: *mut CandleSyclQueue) -> *mut c_void;
    pub fn candle_sycl_synchronize(q: *mut CandleSyclQueue) -> c_int;
    pub fn candle_sycl_device_info(
        q: *mut CandleSyclQueue,
        out: *mut CandleSyclDeviceInfo,
    ) -> c_int;

    pub fn candle_sycl_malloc(q: *mut CandleSyclQueue, bytes: usize) -> *mut c_void;
    pub fn candle_sycl_free(q: *mut CandleSyclQueue, ptr: *mut c_void);
    pub fn candle_sycl_memcpy_htod(
        q: *mut CandleSyclQueue,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub fn candle_sycl_memcpy_dtoh(
        q: *mut CandleSyclQueue,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub fn candle_sycl_memcpy_dtod(
        q: *mut CandleSyclQueue,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub fn candle_sycl_memset(
        q: *mut CandleSyclQueue,
        dst: *mut c_void,
        value: c_int,
        bytes: usize,
    ) -> c_int;

    pub fn candle_sycl_fill(
        q: *mut CandleSyclQueue,
        dt: u32,
        dst: *mut c_void,
        numel: usize,
        value: f64,
    ) -> c_int;
    pub fn candle_sycl_fill_strided(
        q: *mut CandleSyclQueue,
        dt: u32,
        lin: *const CLayout,
        dst: *mut c_void,
        numel: usize,
        value: f64,
    ) -> c_int;
    pub fn candle_sycl_affine(
        q: *mut CandleSyclQueue,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
        mul: f64,
        add: f64,
    ) -> c_int;
    pub fn candle_sycl_elu(
        q: *mut CandleSyclQueue,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
        alpha: f64,
    ) -> c_int;
    pub fn candle_sycl_powf(
        q: *mut CandleSyclQueue,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
        exponent: f64,
    ) -> c_int;
    pub fn candle_sycl_unary(
        q: *mut CandleSyclQueue,
        op: u32,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_binary(
        q: *mut CandleSyclQueue,
        op: u32,
        dt: u32,
        lhs_l: *const CLayout,
        lhs: *const c_void,
        rhs_l: *const CLayout,
        rhs: *const c_void,
        out: *mut c_void,
        numel: usize,
    ) -> c_int;
    pub fn candle_sycl_cast(
        q: *mut CandleSyclQueue,
        src_dt: u32,
        dst_dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        numel: usize,
    ) -> c_int;
    pub fn candle_sycl_copy_strided(
        q: *mut CandleSyclQueue,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        dst_offset: usize,
        numel: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_copy2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> c_int;

    pub fn candle_sycl_reduce(
        q: *mut CandleSyclQueue,
        op: u32,
        dt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        out: *mut c_void,
        out_el: usize,
        reduce_el: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_cmp(
        q: *mut CandleSyclQueue,
        op: u32,
        dt: u32,
        lhs_l: *const CLayout,
        lhs: *const c_void,
        rhs_l: *const CLayout,
        rhs: *const c_void,
        out: *mut c_void,
        numel: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_where(
        q: *mut CandleSyclQueue,
        cond_dt: u32,
        val_dt: u32,
        cond_l: *const CLayout,
        cond: *const c_void,
        t_l: *const CLayout,
        t_vals: *const c_void,
        f_l: *const CLayout,
        f_vals: *const c_void,
        out: *mut c_void,
        numel: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_index_select(
        q: *mut CandleSyclQueue,
        dt: u32,
        idt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        ids: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_gather(
        q: *mut CandleSyclQueue,
        dt: u32,
        idt: u32,
        lin: *const CLayout,
        inp: *const c_void,
        ids: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_scatter(
        q: *mut CandleSyclQueue,
        add: c_int,
        dt: u32,
        idt: u32,
        out: *mut c_void,
        ids: *const c_void,
        src: *const c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_index_add(
        q: *mut CandleSyclQueue,
        dt: u32,
        idt: u32,
        out: *mut c_void,
        ids: *const c_void,
        src: *const c_void,
        left_size: usize,
        ids_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    ) -> c_int;
    pub fn candle_sycl_argsort(
        q: *mut CandleSyclQueue,
        dt: u32,
        ascending: c_int,
        inp: *const c_void,
        out: *mut u32,
        nrows: usize,
        ncols: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_avg_pool2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        src: *const i64,
        k_h: usize,
        k_w: usize,
        s_h: usize,
        s_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_max_pool2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        src: *const i64,
        k_h: usize,
        k_w: usize,
        s_h: usize,
        s_w: usize,
        h_out: usize,
        w_out: usize,
    ) -> c_int;
    pub fn candle_sycl_upsample_nearest2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        src: *const i64,
        dst_h: usize,
        dst_w: usize,
    ) -> c_int;
    pub fn candle_sycl_upsample_nearest1d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        src: *const i64,
        dst_w: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_upsample_bilinear2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        src: *const i64,
        dst_h: usize,
        dst_w: usize,
        align_corners: c_int,
        scale_h: f64,
        scale_w: f64,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_im2col2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        col: *mut c_void,
        meta: *const i64,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_im2col1d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        col: *mut c_void,
        meta: *const i64,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out_l: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_conv_transpose2d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        im: *const i64,
        ker: *const c_void,
        out: *mut c_void,
        b: usize,
        c_in: usize,
        c_out: usize,
        ih: usize,
        iw: usize,
        kh: usize,
        kw: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_conv_transpose1d(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        im: *const i64,
        ker: *const c_void,
        out: *mut c_void,
        b: usize,
        c_in: usize,
        c_out: usize,
        il: usize,
        kl: usize,
        out_l: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> c_int;
    pub fn candle_sycl_dequantize(
        q: *mut CandleSyclQueue,
        ggml_dtype: u32,
        src: *const c_void,
        dst_f32: *mut c_void,
        n_blocks: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_mmvq(
        q: *mut CandleSyclQueue,
        dt: u32,
        w: *const c_void,
        act: *const f32,
        out: *mut f32,
        n: usize,
        k: usize,
        m: usize,
    ) -> c_int;
    pub fn candle_sycl_softmax_lastdim(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        out: *mut c_void,
        rows: usize,
        d: usize,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_rms_norm(
        q: *mut CandleSyclQueue,
        dt: u32,
        inp: *const c_void,
        alpha: *const c_void,
        out: *mut c_void,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_rope(
        q: *mut CandleSyclQueue,
        mode: u32,
        dt: u32,
        inp: *const c_void,
        cosb: *const c_void,
        sinb: *const c_void,
        out: *mut c_void,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        cos_batched: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn candle_sycl_gemm(
        q: *mut CandleSyclQueue,
        dt: u32,
        transa: c_int,
        transb: c_int,
        m: i64,
        n: i64,
        k: i64,
        alpha: f64,
        beta: f64,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        batch: i64,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        off_a: i64,
        off_b: i64,
    ) -> c_int;
}
