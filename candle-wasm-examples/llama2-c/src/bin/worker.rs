use yew_agent::PublicWorker;
fn main() {
    console_log::init().expect("could not initialize logger");
    candle_wasm_example_llama2::Worker::register();
}
