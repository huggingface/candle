use candle::{CpuStorage, QCpuStorage};
use wasm_bindgen::prelude::*;
pub mod token_output_stream;
use candle_transformers::models::blip;
use candle_transformers::models::quantized_blip;

pub type Tensor = candle::Tensor<CpuStorage>;
pub type BlipForConditionalGeneration = blip::BlipForConditionalGeneration<CpuStorage>;
pub type QBlipForConditionalGeneration = quantized_blip::BlipForConditionalGeneration<QCpuStorage>;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}
