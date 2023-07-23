#[rustfmt::skip]
pub const LAYERNORM_KERNELS: &str = include_str!(concat!(env!("OUT_DIR"), "/examples/custom-ops/kernels//layernorm_kernels.ptx"));
