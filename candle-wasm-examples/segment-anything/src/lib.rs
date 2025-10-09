use candle::CpuStorage;
use candle_transformers::models::segment_anything::sam;
use wasm_bindgen::prelude::*;

pub use sam::IMAGE_SIZE;

pub type Tensor = candle::Tensor<CpuStorage>;
pub type VarBuilder<'a> = candle_nn::VarBuilder<'a, CpuStorage>;
pub type Sam = sam::Sam<CpuStorage>;

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
