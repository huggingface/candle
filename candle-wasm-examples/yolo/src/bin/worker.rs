use yew_agent::PublicWorker;
fn main() {
    console_error_panic_hook::set_once();
    candle_wasm_example_yolo::Worker::register();
}
