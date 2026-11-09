use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
}

pub mod m;
pub mod profiler;