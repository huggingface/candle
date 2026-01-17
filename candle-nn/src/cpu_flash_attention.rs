//! Backward compatibility shim for CPU flash attention.
//!
//! **Deprecated:** Use `candle_nn::attention::{flash_attn, AttnMask}` instead.

#[deprecated(
    since = "0.9.2",
    note = "Use `candle_nn::attention::{flash_attn, AttnMask}` instead"
)]
pub use crate::attention::cpu_flash::standard::run_flash_attn_cpu;
