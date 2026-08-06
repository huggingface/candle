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

pub mod m;
pub mod profiler;

// Exposes `initThreadPool(numThreads)` to JS (returns a Promise). The page must
// `await` it after `init()` and before the first generation so the rayon worker
// pool backing candle's wasm-threads matmul exists. Needs cross-origin isolation
// (COOP/COEP) for SharedArrayBuffer - serve.py sets those headers.
pub use wasm_bindgen_rayon::init_thread_pool;
