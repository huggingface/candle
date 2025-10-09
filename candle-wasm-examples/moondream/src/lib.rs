use candle::{CpuStorage, QCpuStorage};
use wasm_bindgen::prelude::*;

pub type Tensor = candle::Tensor<CpuStorage>;
pub type MoonDream = candle_transformers::models::moondream::Model<CpuStorage>;
pub type QMoonDream = candle_transformers::models::quantized_moondream::Model<QCpuStorage>;
pub type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder<QCpuStorage>;

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
