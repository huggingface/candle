mod app;
pub mod model;
pub mod worker;
pub use app::App;
pub use worker::Worker;

pub type Tensor = candle::Tensor<candle::CpuStorage>;
pub type VarBuilder<'a> = candle_nn::VarBuilder<'a, candle::CpuStorage>;
pub type Embedding = candle_nn::Embedding<candle::CpuStorage>;
pub type Linear = candle_nn::Linear<candle::CpuStorage>;
pub type RmsNorm = candle_nn::RmsNorm<candle::CpuStorage>;
