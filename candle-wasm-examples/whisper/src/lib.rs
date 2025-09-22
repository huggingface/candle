pub const WITH_TIMER: bool = true;

mod app;
mod audio;
pub mod languages;
pub mod worker;
pub use app::App;
use candle::{quantized::QCpuStorage, CpuStorage};
pub use worker::Worker;

pub type Tensor = candle::Tensor<CpuStorage>;
pub type Whisper = candle_transformers::models::whisper::model::Whisper<CpuStorage>;
pub type QWhisper = candle_transformers::models::whisper::quantized_model::Whisper<QCpuStorage>;
