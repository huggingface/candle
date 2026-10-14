//! Launchers from the `candle-offload-kernels` cdylib, over the C ABI. A cdylib
//! because offload lowering needs fat LTO on a final artifact, not an rlib; this
//! keeps offload flags out of candle. build-offload.sh sets link path + rpath.

#[link(name = "candle_offload_kernels")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn candle_offload_gemm_f32(
        pa: usize,
        pb: usize,
        pc: usize,
        nb: usize,
        m: usize,
        n: usize,
        k: usize,
        brs: usize,
        bcs: usize,
        a_bs: usize,
        b_bs: usize,
    );
    pub fn candle_offload_binary_f32(op: usize, pa: usize, pb: usize, pc: usize, n: usize);
    pub fn candle_offload_unary_f32(op: usize, pa: usize, pc: usize, n: usize);
    pub fn candle_offload_affine_f32(pa: usize, pc: usize, n: usize, mul: f32, add: f32);
    pub fn candle_offload_fill_f32(pc: usize, n: usize, v: f32);
    pub fn candle_offload_copy_f32(pa: usize, pc: usize, n: usize);
    pub fn candle_offload_reduce_last_f32(
        op: usize,
        pa: usize,
        pc: usize,
        rows: usize,
        cols: usize,
    );
    pub fn candle_offload_rms_norm_f32(
        px: usize,
        pa: usize,
        pd: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    );
    pub fn candle_offload_softmax_last_f32(px: usize, pd: usize, rows: usize, cols: usize);
    #[allow(clippy::too_many_arguments)]
    pub fn candle_offload_rope_f32(
        ps: usize,
        pcos: usize,
        psin: usize,
        pd: usize,
        bh: usize,
        t: usize,
        d: usize,
        h: usize,
        cs_b: usize,
    );
    pub fn candle_offload_convert(psrc: usize, pdst: usize, n: usize, op: usize);
    #[allow(clippy::too_many_arguments)]
    pub fn candle_offload_index_select_f32(
        psrc: usize,
        pids: usize,
        pdst: usize,
        left: usize,
        n_ids: usize,
        right: usize,
        mid: usize,
        ids_kind: usize,
    );
    pub fn candle_offload_num_devices() -> i32;
}

// Opcodes, kept in sync with candle-offload-kernels BY HAND (not importable
// across the cdylib). Drift = silent wrong results.
pub const OP_ADD: usize = 0;
pub const OP_SUB: usize = 1;
pub const OP_MUL: usize = 2;
pub const OP_DIV: usize = 3;
pub const OP_MAX: usize = 4;
pub const OP_MIN: usize = 5;

pub const U_NEG: usize = 0;
pub const U_SQR: usize = 1;
pub const U_SQRT: usize = 2;
pub const U_EXP: usize = 3;
pub const U_LOG: usize = 4;
pub const U_SIN: usize = 5;
pub const U_COS: usize = 6;
pub const U_ABS: usize = 7;
pub const U_RECIP: usize = 8;
pub const U_RELU: usize = 9;
pub const U_SILU: usize = 10;
pub const U_GELU: usize = 11;
pub const U_TANH: usize = 12;
pub const U_SIGN: usize = 13;

pub const R_SUM: usize = 0;
pub const R_MAX: usize = 1;
pub const R_MIN: usize = 2;

pub const CVT_F16_F32: usize = 0;
pub const CVT_BF16_F32: usize = 1;
pub const CVT_F32_F16: usize = 2;
pub const CVT_F32_BF16: usize = 3;

pub const IDS_U32: usize = 0;
pub const IDS_I64: usize = 1;
