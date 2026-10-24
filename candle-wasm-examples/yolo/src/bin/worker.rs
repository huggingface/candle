use yew_agent::Registrable;
fn main() {
    console_error_panic_hook::set_once();
    candle_wasm_example_yolo::Worker::registrar().register();
}
