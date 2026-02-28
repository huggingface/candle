use yew_agent::PublicWorker;
fn main() {
    console_error_panic_hook::set_once();
    console_log::init().expect("could not initialize logger");
    candle_wasm_example_llama2::Worker::register();
}
