//! Gemma model family implementations.

pub mod gemma1;
pub mod gemma2;
pub mod gemma3;
pub mod quantized_gemma3;
pub mod translate_gemma;

#[deprecated(
    since = "0.9.2",
    note = "use `models::gemma::gemma1::{Config, Model}` instead"
)]
pub use gemma1::{Config, Model};
