use rayon::prelude::*;
use wasm_bindgen::prelude::*;

pub use wasm_bindgen_rayon::init_thread_pool;

fn test_rayon_multithreading() {
    let numbers: Vec<i32> = (1..=1000).collect();

    let sum: i32 = numbers.par_iter().cloned().sum();

    console_log!("Sum of 1..=1000 using rayon: {}", sum);

    let threads = rayon::current_num_threads();
    console_log!(
        "Multithreading Test - Rayon reports {} threads available",
        threads
    );
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = wasm_bindgen_rayon::init_thread_pool(4);
    test_rayon_multithreading();
}

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
