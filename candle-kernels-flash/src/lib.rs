pub const FMHA_BLOCK_DGRAD_FP16_KERNEL_LOOP_SM80: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/fmha_block_dgrad_fp16_kernel_loop.sm80.ptx"
));
pub const FMHA_BLOCK_FPROP_FP16_KERNEL_SM80: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/fmha_block_fprop_fp16_kernel.sm80.ptx"
));
pub const FMHA_BWD_HDIM128: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_bwd_hdim128.ptx"));
pub const FMHA_BWD_HDIM32: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_bwd_hdim32.ptx"));
pub const FMHA_BWD_HDIM64: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_bwd_hdim64.ptx"));
pub const FMHA_FWD_HDIM128: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_fwd_hdim128.ptx"));
pub const FMHA_FWD_HDIM32: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_fwd_hdim32.ptx"));
pub const FMHA_FWD_HDIM64: &str = include_str!(concat!(env!("OUT_DIR"), "/fmha_fwd_hdim64.ptx"));
